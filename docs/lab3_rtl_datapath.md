# Lab 3 — RTL Datapath

This is where the architecture spec becomes hardware. Read `architecture_spec.md`
Sections 3 and 4 again before starting — this lab implements exactly what's described
there, module by module.

## Learning objectives

- Implement a DSP48E1-inferring MAC lane with local BRAM-backed weight storage.
- Implement the 5-step FC1 requantization pipeline in synthesizable RTL, bit-identical
  to `golden_model.py`.
- Implement a reusable, parameterized layer engine (used for both FC1 and FC2) built
  from parallel MAC lanes.
- Implement a top-level FSM that sequences FC1 → requantize → FC2 → argmax and
  produces the board-facing outputs from `architecture_spec.md` Section 5.

## Module map

Five files, in the order you should tackle them (each depends conceptually on the
ones before it):

| File | Role |
|---|---|
| [`rtl/mac_lane.sv`](../rtl/mac_lane.sv) | One MAC lane: local weight memory + DSP48E1-style multiply-accumulate |
| [`rtl/requant_relu.sv`](../rtl/requant_relu.sv) | The 5-step FC1 requantization pipeline, applied per-neuron |
| [`rtl/fc_layer.sv`](../rtl/fc_layer.sv) | 16-lane wrapper + batch FSM; reused for both FC1 and FC2 |
| [`rtl/argmax10.sv`](../rtl/argmax10.sv) | 10-way comparator, produces the predicted digit |
| [`rtl/digit_top.sv`](../rtl/digit_top.sv) | Top-level integration: image ROM, both `fc_layer` instances, requant bank, argmax, board I/O |

## Background: how the batch/addressing scheme actually works

Section 4 of the spec says each of the 16 lanes "owns a disjoint set of output
neurons." Concretely, for FC1 (64 neurons, 16 lanes): each lane owns **4** neurons,
processed one at a time across **4 sequential batches** — lane `L` computes neuron
`L` in batch 0, neuron `L+16` in batch 1, `L+32` in batch 2, `L+48` in batch 3 (or an
equivalent contiguous mapping — pick one, document it, and make sure your weight
memory layout and your Lab 1 export step agree with each other).

The useful consequence of this scheme: **the streamed input index and the current
batch number are identical across all 16 lanes at every cycle** — only the *weight
value* stored at that address differs per lane, because each lane's local memory holds
different neurons' weights. That means you only need **one shared address** and **one
shared broadcast activation value** per cycle, fanned out to all 16 lanes — this is
what makes `mac_lane.sv`'s `weight_addr` and `act_data` ports single (not per-lane)
signals in the skeleton, and it's the direct hardware payoff of the weight-stationary/
broadcast decision from Lab 2.

For FC2 (10 neurons, 16 lanes, single batch): only lanes 0–9 own a real neuron; lanes
10–15 still run (there's no harm in them computing garbage into a memory location
nothing reads), but `fc_layer`'s output-collection logic must not present their output
as one of the 10 FC2 scores. Decide explicitly how you gate this rather than leaving
it implicit.

**Non-negative operands, both layers:** pixels (`0..255`) and FC1's post-ReLU output
`a1` (clamped-then-ReLU'd, so effectively `0..127`) are both *always non-negative* —
the only signed operands in this whole datapath are the weights. That means both
`act_data` inputs (FC1's pixels and FC2's `a1` values) get **zero-extended**, never
sign-extended, into the signed multiplier — the same rule in both layers, which is
worth stating as a single unified insight rather than two separate special cases if
you're asked about it.

**BRAM read latency is not zero.** A registered-address BRAM read (the pattern that
gets you real BRAM inference, per Lab 2) returns its data one cycle *after* the
address is presented, not the same cycle. Your `fc_layer` batch FSM needs to account
for this — either with an explicit wait state, or a shift-register that delays the
"this data is valid" signal by the same number of cycles as your read latency. This is
a classic off-by-one-cycle bug source; get it right early, because it'll otherwise
show up much later as unexplained garbage in your first neuron's accumulation. Decide
where in the pipeline this delay belongs (inside `mac_lane`, inside `fc_layer`, or
both) and keep it consistent.

**Weight file init is an open decision, not solved for you.** Each `mac_lane`
instance needs its own local memory pre-loaded with its slice of the weight matrix.
`train_quantize.py` (Lab 1) exports flat `w1.mem`/`w2.mem` files as the primary
weight/bias artifacts. Getting each lane its own initialization content is a real
design decision you need to make and be able to defend: either (a) extend your Lab 1
export step to also emit 16 small per-lane `.mem` files, one per `mac_lane` instance,
or (b) some other scheme you can justify. Whichever you pick, document it — "how did
you get weights into 16 independent BRAMs from one trained model" is exactly the kind
of question an interviewer asks to check you understand memory architecture, not just
recite it.

**Tie-break bit-exactness.** Python's `max(range(10), key=lambda k: scores[k])` picks
the *smallest index* among tied maximum values. Your `argmax10.sv` comparator must
implement the same tie-break rule (i.e., strictly-greater-than comparisons that only
update the running max on a strict improvement, never on equality) or a tie in the
test data will produce a real, if rare, mismatch between RTL and the golden model in
Lab 4 that has nothing to do with a "real" bug — just an inconsistent tie-break
convention. Decide this now and write it down.

## Your task

Fill in the `TODO`s in all five files in the order listed above. Don't skip straight
to `digit_top.sv` — it depends on the others working, and you won't be able to
meaningfully test it standalone until they do (Lab 4 covers unit-testing each piece
first).

Specific things each file's TODOs ask you to decide and implement:

- **`mac_lane.sv`**: infer the local weight BRAM (`$readmemh`-initialized array,
  registered address for real BRAM inference per Lab 2), implement the DSP48E1 MACC
  coding pattern (registered A/B inputs, registered accumulate) per Lab 2's template,
  and implement `acc_clear` (load the bias instead of accumulating — this is how a
  new neuron's computation starts).
- **`requant_relu.sv`**: implement the exact 5 steps from `architecture_spec.md`
  Section 2 / `golden_model.py`'s `fc1_forward_int`. This is pure combinational/
  pipelined arithmetic — no memory involved.
- **`fc_layer.sv`**: the batch FSM (stream `N_IN` activations per batch, `ceil(N_OUT /
  LANES)` batches total, handle the BRAM-latency delay above), instantiate `LANES`
  `mac_lane`s, read the bias ROM, collect `acc_out[N_OUT]`, and handle FC2's partial
  final batch (only some lanes valid).
- **`argmax10.sv`**: the comparator tree, with the tie-break rule above.
- **`digit_top.sv`**: image ROM + label ROM, one `fc_layer` instance parameterized for
  FC1 and one for FC2 (reusing the same module, different parameters — this reuse is
  the payoff of making `fc_layer` generic), a bank of `requant_relu` instances (decide
  how many — full 64-wide parallel, or fewer reused across a few cycles; the spec
  estimates ~10 cycles of requant overhead, so pick something in that ballpark and be
  able to justify the choice), the top FSM (`IDLE → FC1 → REQUANT → FC2 → ARGMAX →
  DONE`), the cycle counter, and the board-facing outputs (`busy`, `done`,
  `pred_digit`, `correct`, `cycle_count`) from `architecture_spec.md` Section 5.
  `digit_top` itself does **not** debounce `start`/`rst` — assume clean single-cycle
  inputs; a board-level wrapper adding debounce comes in Lab 6.

## Self-check

Before moving to Lab 4:
- Every module in the table above elaborates without errors in simulation (even with
  stub/zero logic — the point here is just that ports and instantiation hierarchy are
  wired up correctly; functional correctness is Lab 4's job).
- You can explain, out loud, why `mac_lane`'s address and activation inputs are single
  shared signals rather than per-lane arrays.
- You've written down (a comment or a short note is fine) your answers to the three
  open decisions flagged above: weight file init scheme, BRAM-latency handling
  location, and requant-bank parallelism.

## Reflection questions

1. Why can every lane use the same broadcast address and activation value each cycle,
   when each lane is computing a *different* neuron's output?
2. What would go wrong, concretely, if your `fc_layer` FSM didn't account for BRAM
   read latency — what would the first accumulated value actually be?
3. Why do both FC1's input (pixels) and FC2's input (`a1`) get zero-extended rather
   than sign-extended, and what's the one thing in this whole datapath that's ever
   actually negative?
4. How does your argmax tie-break rule match Python's `max()` behavior, and why does
   that matching matter for Lab 4?
5. Why is `fc_layer` written as one generic, reusable module instead of two separate
   `fc1_layer`/`fc2_layer` modules (which is what the old version of this project
   did)?
