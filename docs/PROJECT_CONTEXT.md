# Project Context — MNIST Hardware Accelerator (handoff brief)

This document exists to bring a new assistant up to speed on this project without
re-deriving everything. Read it start to finish before answering anything.

---

## 0. How to work with me (read this first — it overrides default behavior)

I am an EE student at Purdue targeting Summer 2027 digital/ASIC design internships.
This project is both a learning vehicle and a portfolio/interview artifact. That shapes
everything about how I want help:

**You are a teacher and reviewer. You are never an implementer.**

- Do **not** write RTL for me. Do not edit my `.sv` files. Do not paste in "here's the
  implementation." I write every line myself — that is the entire point.
- Do explain concepts, derive timing, check my reasoning, and review my work.
- When I paste a block diagram screenshot, review it **signal by signal**: every box,
  every wire, every bit-width label, every missing connection. Be exhaustive and blunt.
  I would rather hear ten real issues than a compliment.
- Rank findings by severity. Tell me *why* something is wrong and *what direction* to
  fix it — not the fixed artifact itself.
- If I push back on something you said, re-examine it honestly. I have caught you being
  wrong before (see §7) and I expect you to say so when it happens.
- Ground claims in the actual files. Read `rtl/*.sv` and `docs/*.md` rather than
  reasoning from memory of what we discussed.

I design block diagrams in an external diagramming tool and paste them as screenshots.
The diagrams are the design; the RTL gets written *from* them afterward. So a bug caught
in a diagram is worth ten caught in code.

---

## 1. What this project is

A digit-classification neural-network inference accelerator on an FPGA.

- **Board:** Digilent Basys3, Xilinx Artix-7 XC7A35T-1CPG236C
  (33,280 logic cells, 225KB BRAM, 90 DSP48E1 slices), Vivado toolchain.
- **Clock:** 100MHz directly off the onboard oscillator, no MMCM/PLL.
- **Network:** 196 → 64 (ReLU) → 10 MLP. No convolution (explicitly out of scope).
- **Input:** real MNIST, 28×28 8-bit grayscale, 2×2-average-pooled to 14×14 = 196
  values. **Pooling happens offline in Python only** — the RTL never sees 28×28.
- **Target accuracy:** ≥95% on the MNIST test set after quantization.
- **Throughput target:** ~900 cycles ≈ 9µs per inference at 100MHz.

This is a **rewrite** of an earlier version of the same project. The old version had a
specific, embarrassing bug: it computed a requantization shift in Python and then never
applied it in hardware, silently turning an "int8 quantized accelerator" into an
unquantized int32 accelerator with int8 weights bolted on the input side. Fixing that —
and being able to explain it in an interview — is a core motivation. `requant_relu.sv`
is the module the old version never built at all.

The project is organized as a series of labs (Lab 1 → Lab 7) with spec docs in `docs/`.
**Currently in Lab 3 (RTL datapath).**

---

## 2. The locked spec (do not renegotiate these)

`docs/architecture_spec.md` is the authoritative reference. If my implementation and
that doc disagree, that's a bug in my implementation. The Python golden model and the
RTL must implement *exactly* the same integer algorithm so they can be compared
bit-for-bit in Lab 4.

### Quantization

| Thing | Representation |
|---|---|
| Weights | symmetric signed int8, `[-127, 127]`, per-layer scale |
| Input pixels | **unsigned** uint8 `[0, 255]`, no scale |
| FC1 activations (`a1`) | int8, effectively `[0, 127]` after clamp+ReLU |
| Accumulators | int32, both layers |
| Biases | int32, pre-scaled into the accumulator's fixed-point domain |

**The single easiest bug to introduce in this datapath:** pixels are *unsigned*. They
must be **zero-extended** to 9 bits going into the signed multiplier, never
sign-extended. Same rule applies to FC2's `a1` inputs. The **only** signed operands in
the entire datapath are the weights.

### FC1 requantization — five steps, this exact order

```
1. acc1       = sum(w1[i] * x[i]) + bias_q_fc1        # int32
2. acc1_round = acc1 + (1 <<< (SHIFT1 - 1))           # round-half-up
3. acc1_shift = acc1_round >>> SHIFT1                 # ARITHMETIC right shift
4. acc1_clamp = clamp(acc1_shift, -128, 127)          # saturate to int8
5. a1         = (acc1_clamp < 0) ? 0 : acc1_clamp     # ReLU
```

`SHIFT1` is a single constant calibrated once offline in Python and baked in as a
`localparam`. Not runtime-configurable. Expect ~10–12; calibrate, don't guess.

### FC2 — deliberately NOT requantized

FC2 feeds an argmax, not another quantized multiply. Requantizing would risk two close
logits saturating to the same value and destroying the ranking, for zero benefit (a
uniform shift can't change which of 10 numbers is largest). Raw int32 straight into
argmax.

### Argmax tie-break (bit-exactness requirement)

Python's `max(range(10), key=...)` picks the **smallest index** among tied maxima. The
comparator must only update the running max on a **strict** improvement (`>`), never on
equality. Otherwise Lab 4's bit-exact comparison shows phantom mismatches on ties.

### Board I/O (spec §5)

| Basys3 I/O | Function |
|---|---|
| `clk` (W5, 100MHz) | system clock, used directly |
| `SW[4:0]` | `img_idx`, selects stored test image 0–19; **values above 19 clamp to 19**, no wraparound |
| `BTNC` | `start` — debounced + rising-edge-detected to a single-cycle pulse |
| `BTNU` | `rst` |
| 7-segment, 1 digit | `pred_digit` (no multiplexing needed — only one digit ever driven) |
| `LD[0]` | `busy` |
| `LD[1]` | `done` — **held until the next `start`** |
| `LD[2]` | `correct` (pred vs. stored label self-check) |
| `LD[15:4]` | `cycle_count` from the last inference |

`digit_top` does **not** debounce. It assumes clean single-cycle inputs; the debounce
wrapper is a Lab 6 deliverable.

**`digit_top` port list:** `clk`, `rst`, `start`, `img_idx[4:0]`, `busy`, `done`,
`pred_digit[3:0]`, `correct`, `cycle_count[15:0]`, `score_debug[319:0] signed`
(= `N_CLASS*32`, sim/ILA debug only).

---

## 3. Project-wide RTL rules I have imposed on myself

These are stricter than the spec requires. They are non-negotiable and apply to every
module. Hold me to them.

1. **All FSMs are purely Moore.** Outputs are a function of state only. Three-block
   structure: `next_state_logic` (comb) → `state_FF` (seq) → `output_logic` (comb).
2. **Every `always_ff` must reduce to exactly:**
   ```systemverilog
   if (rst) x <= reset_value;
   else     x <= next_x;
   ```
   No enables, no clears, no muxes, no arithmetic inside the register block. Any
   decision logic lives in a separate purely-combinational block that produces `next_x`.
   An enable becomes `next_x = en ? new_value : x` in the comb block.
3. **Explicit feedback arrows.** Whenever a comb block references its own downstream
   register's current output (e.g. the "else hold" branch above), the diagram must draw
   that feedback wire. It is real routing, not an implementation detail.

### Diagramming conventions

- Repeated structures collapse into **one box with a multiplicity label**
  (`mac_lane (x16)`, `requant_relu_bank (x64)`). Parallel buses are one labeled wide
  arrow (`acc_out[0:63]`, width `32`), never exploded into N boxes.
- Multiplicity labels do **not** carry into the lower-level diagram of that instance —
  `mac_lane`'s own internal diagram has no `(x16)` anywhere, because at that zoom level
  there is exactly one of everything.
- Every combinational box carries its **pseudocode** inside it, not just a name.
- Every wire gets a **name and a bit-width label**.
- Registers show a `clk` triangle. Async/combinational lookups (e.g. LUTROM) do not.
- **I am currently omitting `rst` from diagrams on purpose**, for readability. Don't
  flag missing `rst` wires. The rule in #2 above still applies in the actual RTL.

---

## 4. Architectural decisions I made that DIVERGE from the spec doc

Important: `docs/architecture_spec.md` has drifted from my actual design in several
places. Where they conflict, the list below is what I'm building. These divergences are
deliberate and defensible, but **the spec doc has not been updated yet** (a TODO).

1. **Two `fc_layer` instances, not one time-shared engine.**
   Spec §3 describes ONE set of 16 per-lane weight memories holding both FC1 and FC2
   regions, "addressed with bit 10 selecting FC1 vs FC2 region" (hence the stale
   `LOCAL_MEM_DEPTH = 1088` = 1024 + 64 default). I instantiate `fc_layer` **twice**
   (once parameterized for FC1, once for FC2), which is what the skeleton's own comments
   call for.
   **Consequences:** 32 `mac_lane`s → **32 DSP48E1s (35.6% of 90), not 16 (17.8%)**;
   32 weight BRAMs, not 16; six of FC2's lanes permanently idle. Spec §4's utilization
   numbers are now wrong. The bit-10 region addressing is dead.
   **Real per-lane weight memory depths:** FC1 = **784** (10-bit addr),
   FC2 = **64** (6-bit addr).

2. **Contiguous lane→neuron mapping.** Lane `L` owns neurons `L*4 .. L*4+3` for FC1.
   The lab doc's *example* is a strided mapping (`L, L+16, L+32, L+48`) but explicitly
   permits "an equivalent contiguous mapping." Contiguous is simpler to export
   (`W1_q[L*4 : L*4+4]`, a plain slice).

3. **Per-lane weight files.** Each `mac_lane` gets its own `.mem` containing only its
   own neurons' weights, concatenated in **batch-major order** (batch 0's N_IN weights,
   then batch 1's, ...). This makes `weight_addr = batch*N_IN + i` a correct **broadcast**
   address into every lane's local ROM simultaneously.
   **Not yet implemented in `train_quantize.py`** — see §6.

4. **Bias storage moved INSIDE `mac_lane`.** The original `bias_in` input port is
   replaced by a narrow `bias_addr` port plus an internal async LUTROM named
   `bias_selector` holding `NUM_BATCHES` int32 entries. Rationale: different lanes need
   genuinely different bias values within the same batch, so one shared fc_layer-level
   bias ROM cannot work; and biases are tiny, so LUTROM (async, zero added latency) is
   the right primitive rather than BRAM.
   **Consequence:** `fc_layer`'s `BIAS_INIT_FILE` parameter is now stale; `mac_lane`
   needs its own per-lane `BIAS_INIT_FILE`, and `train_quantize.py` needs a per-lane
   bias split parallel to the weight split.

5. **The pipeline-stagger "buffer FF" lives inside `mac_lane`**, between the
   `weight_addr` port and the Weights ROM. See §5 for why.

6. **64-wide requant bank (1 cycle)** rather than the spec's suggested 16-wide/~10
   cycles. Faster, more area. Lab doc explicitly leaves this as my call.

---

## 5. Pipeline timing — the derivation everything depends on

This is the subtlest part of the design. Get it wrong and you get silent garbage.

Both multiplier operands must arrive at the multiplier on the **same cycle**, and the
two source paths have different natural latencies:

**Activation path (FC1):** `act_addr(t)` → external image ROM (registered read, 1 cyc)
→ `act_data(t+1)` → `act_data_FF` inside `mac_lane` (1 cyc) → `act_data_r(t+2)`.

**Weight path:** naturally only 1 cycle (`weight_addr(t)` → registered ROM read →
`weight_data(t+1)`). To match, a **buffer FF** is inserted inside `mac_lane` before the
ROM: `weight_addr(t)` → buffer FF `(t+1)` → Weights ROM registered read →
`weight_data(t+2)`. ✓ aligned.

Key insight (I got this backwards once): you can only **add** delay to the naturally
*faster* path. `act_addr` stays raw and undelayed because it feeds the longer, fixed
external path and must be issued as early as possible.

**Activation path (FC2)** is different and needs its own equalizer:
`act_addr(t)` → **combinational** mux over the `requant_r` register bank →
`act_data(t)` → `delay FF` **in `digit_top`** (1 cyc) → `act_data_delay(t+1)` →
`act_data_FF` in `mac_lane` → `act_data_r(t+2)`. ✓ aligned.
The `delay FF` exists purely to make FC2's comb-mux path match FC1's registered-ROM
path. It is not optional.

**Consequence for the `fc_layer` FSM:** the `ACCUMULATE` state lasts **`N_IN + 2`
cycles**, not `N_IN`. The counter runs `0 .. N_IN+1`. The product for input index `i`
lands at cycle `i+2`, so valid accumulates are cycles `2 .. N_IN+1` — exactly `N_IN` of
them. The addressing must be **clamped** at `N_IN-1` once the count reaches `N_IN` so it
doesn't run off the end of memory during those last two cycles.

**Also:** `mac_lane`'s `acc_out` for the final index isn't valid until the clock edge
that also exits `ACCUMULATE`. That's why `CAPTURE` must be its own separate state and
cannot be folded into `ACCUMULATE`'s last cycle.

**BRAM note:** a single registered-address read (one `always_ff`, one flop) = 1 cycle
and is what infers a real BRAM primitive. This is NOT the same as the optional second
"output register" BRAM mode (2-cycle), which is an unrelated high-Fmax optimization not
needed at 100MHz.

---

## 6. Current state of every file

| File | State |
|---|---|
| `rtl/mac_lane.sv` | **Implemented**, but reflects the OLD design — still has `bias_in` port, no buffer FF, no `bias_selector`, stale `LOCAL_MEM_DEPTH=1088` |
| `rtl/requant_relu.sv` | **Implemented.** See the gotcha below. |
| `rtl/fc_layer.sv` | **Skeleton / TODOs only** |
| `rtl/argmax10.sv` | **Skeleton / TODOs only** |
| `rtl/digit_top.sv` | **Skeleton / TODOs only** |
| `sw/train_quantize.py` | Done **except** per-lane weight/bias file splitting — currently exports one flat `w1.mem`/`w2.mem`/`b1.mem`/`b2.mem` |
| `sw/golden_model.py` | Done |
| `tb/tb_digit_top.sv` | Lab 4, not started |

**`requant_relu.sv` gotcha (already caught, affects `digit_top`):** its `always_ff`
drives `a1_out <= 0` in the `else` branch. So `a1_out` is valid for **exactly one cycle**
after `en` and then self-clears. Any capture register downstream must be gated on
`valid_out`, or it will latch zeros the very next cycle.

**`argmax10.sv` is a clocked module** — ports are `clk, rst, en, scores_in[0:9]`,
`pred_digit[3:0]`, `valid_out`. Not combinational.

### Module port lists (ground truth)

```systemverilog
// fc_layer — params: N_IN=196, N_OUT=64, LANES=16, BIAS_INIT_FILE (stale)
clk, rst, start,
output act_addr[$clog2(N_IN)-1:0], input act_data[8:0],
output busy, done,
output signed [31:0] acc_out [0:N_OUT-1]
// NOTE: valid_out is an INTERNAL signal, not a port.
// localparam NUM_BATCHES = ceil(N_OUT/LANES)  -> FC1: 4, FC2: 1

// mac_lane — params: LOCAL_MEM_DEPTH (stale 1088), WEIGHT_INIT_FILE
clk, rst, weight_addr[$clog2(LOCAL_MEM_DEPTH)-1:0], act_data[8:0],
acc_clear, bias_in (signed 31:0)  <-- being REPLACED by bias_addr,
acc_en, output signed [31:0] acc_out
```

---

## 7. Design state: the block diagrams

All diagrams are drawn and reviewed except **the `requant_relu` internal diagram, which
is the last one still to draw.** Diagrams live in `Block diagrams/` (untracked).

### `mac_lane` internal (complete)

`weight_addr` → `buffer FF` → `Weights ROM` (clk, registered read) → `weight_data` (8,
signed) → comb block.
`act_data` → `act_data_FF (9)` → `act_data_r` (9) → comb block.
`bias_addr` → `bias_selector` (async LUTROM, **no clk**) → `bias_in` (32) → comb block.
`acc_clear`, `acc_en` → comb block.
Comb block:
```
if acc_clear      next_acc_out = bias_in
else if acc_en    next_acc_out = acc_out + (weight_data * act_data_r)
else              next_acc_out = acc_out
```
→ `next_acc_out` (32) → `acc_out_FF (32)` → `acc_out`, with a **feedback wire** from
`acc_out` back into the comb block (required by rule #3).

### `fc_layer` FSM (5 states, 3-bit)

| State | Outputs | Exit |
|---|---|---|
| `IDLE` | all 0 | `start` → `CLEAR` |
| `CLEAR` | `acc_clear=1`, `busy=1` | unconditional → `ACCUMULATE` |
| `ACCUMULATE` | `acc_en=1`, `busy=1` | `weights_done` → `CAPTURE` |
| `CAPTURE` | `valid_out=1`, `busy=1` | `batches_done` → `DONE`; `!batches_done` → `CLEAR` |
| `DONE` | `done=1` | unconditional → `IDLE` |

`fc_layer`'s `done` is a deliberate **one-cycle pulse** — valid because only
`digit_top`'s FSM consumes it, sampled combinationally every cycle. This is a different
contract from `digit_top`'s own `done`, which spec §5 requires be **held** until the
next `start` (it drives an LED). Both are intentional; don't "fix" either.

### `fc_layer` datapath (complete except open bugs)

- `weights counter` — counts to `N_IN+2`; outputs `weights_done` and `weight_count`
  (`$clog2(N_IN+2)`). Enabled by `acc_en`, reset by `acc_clear`. **← this is bug #1's trap**
- `batch counter` — outputs `batches_done` and `batch` (`$clog2(NUM_BATCHES)`).
  Incremented by `valid_out` (one pulse per batch, since `CAPTURE` is one cycle).
- clamp comb box: `act_addr = (weight_count > N_IN-1) ? N_IN-1 : weight_count`
- `weight_addr = batch*N_IN + <clamped count>` comb box → `mac_lane`
- `bias_addr = batch` comb box → `mac_lane`
- `mac_lane (x16)` → `mac_lane_out[0:15]` (32) → capture comb block:
  ```
  if (valid_out && lane < N_OUT - batch*LANES)
       next_acc_out[batch*LANES + lane] = mac_lane_out[lane]
  else next_acc_out = acc_out
  ```
  → `acc_out_r (N_OUT)` → `acc_out` port, with feedback.
  (`lane` is a **generate/elaboration-time index**, not a runtime signal — it describes
  16 parallel copies of the write logic, all firing in the same `CAPTURE` cycle.)
  The `lane < N_OUT - batch*LANES` term handles FC2's partial final batch.

### `digit_top` FSM (10 states, 4-bit)

`IDLE → FC1_start → FC1_wait → REQUANT_START → REQUANT_WAIT → FC2_START → FC2_WAIT →
ARGMAX_START → ARGMAX_WAIT → DONE → (start) → IDLE`

Pattern is pulse-then-wait: each `*_START` asserts an enable for one cycle, each
`*_WAIT` waits on the corresponding `done`/`valid_out`.
Unlike `fc_layer`'s FSM (which advances by internal cycle-counting), this one advances
on external completion signals from submodules.

### `digit_top` datapath

```
img_idx -> img_idx_r(5) -> rom arithmetic address logic -> image ROM(clk) -> act_data(9) -> fc_layer1
fc_layer1 -> acc_out[0:63](32) -> requant_relu_bank(x64) -> requant_out[0:63](8)
          -> requant FF(512) -> requant_r(512) -> mux logic -> act_data(9)
          -> delay FF(9) -> act_data_delay(9) -> fc_layer2
fc_layer2 -> acc_out2[0:9](32) -> argmax -> pred_digit(4) -> digit FF(4) -> pred_digit
          -> convert array into scalar -> all_scores(320) -> score_debug
img_idx_r -> Label ROM(clk) -> comparator logic <- pred_digit -> correct FF -> correct
enable FF(1) -> delayed_en -> correct FF enable
Counter(en, clear) -> cycle_count
```

---

## 8. OPEN BUGS — found in review, NONE fixed yet

These are the live TODO list. Ordered by severity.

**Blocking (silent wrong answers):**

1. **`acc_en` is high for all of `ACCUMULATE`** → two garbage MACs per batch.
   Valid accumulates are only cycles `2..N_IN+1`, but the Moore output asserts `acc_en`
   from cycle 0. Cycles 0 and 1 add stale pipeline contents (`W[b-1][N_IN-1]*x[N_IN-1]`
   and a duplicate `W[b][N_IN-1]*x[N_IN-1]`) on top of the freshly loaded bias.
   **The trap:** `acc_en` currently doubles as the weights counter's enable. Naively
   gating it on `weight_count >= 2` deadlocks the counter at 0 forever. Fix requires
   decoupling: a separate `count_en` FSM output for the counter, plus a comb gate
   `acc_en = count_en && (weight_count >= 2)` for `mac_lane`. FSM stays Moore.

2. **`requant FF (512)` has no enable.** Combined with `requant_relu`'s self-zeroing
   output (§6), FC2 reads all zeros → same predicted digit for every image. Needs a
   comb block gated on `requant_valid_out` with a hold/feedback branch.

3. **`DONE → IDLE` on `start` in `digit_top`'s FSM** consumes the one-cycle start pulse,
   so every inference after the first needs two button presses. Fix: `DONE --start-->
   FC1_start`, making `IDLE` reset-entry only.

4. **`argmax` drawn without a clock** — it's a sequential module. Also naming: FSM says
   `valid_out`, datapath says `argmax_valid_out`.

**Parameter / elaboration:**

5. **`$clog2(NUM_BATCHES) == 0` for FC2** (`NUM_BATCHES=1`) — zero-width signal, illegal.
   Affects batch counter, `batch`, `bias_addr`, `bias_selector`. Use `$clog2(N+1)` or a guard.
6. **FC2 lanes 10–15 need a zeros `.mem`** — `$readmemh("")` leaves X's that propagate
   into their accumulators. Harmless functionally (output is gated) but floods Lab 4
   waveforms. Also guard the `$readmemh` against an empty filename.
7. **`img_idx` clamp to 19 is missing entirely** (spec §5 requires it). Belongs in the
   `next_img_idx` comb block.
8. **Batch counter has no explicit clear** — currently relies on natural 2-bit wraparound,
   which only works because FC1's `NUM_BATCHES` happens to be a power of two.

**Rule violations (my own rule #2):**

9. Four registers still bundle control into the FF and need `next_X` comb blocks +
   feedback arrows: `img_idx_r` (en), `digit FF` (en), `correct FF` (en),
   `Counter` (en + clear). Plus `requant FF` once #2 is fixed.
   (`requant_relu.sv` itself is also a three-branch `always_ff` — same violation.)
10. **`digit_top` FSM states don't list all outputs** (`fc_layer`'s does — that's the
    standard). Every state should list all six: `busy, done, fc1_start, requant_en,
    fc2_start, argmax_en`. Also `IDLE` currently lists `start=0` as an *output* — but
    `start` is an input to `digit_top`.

**Labeling:** `rom arithmetic address logic` and `mux logic` boxes need their pseudocode
written in (`img_idx_r*N_IN + act_addr`, and the zero-extending
`{1'b0, requant_r[act_addr]}`); several wires need name/width labels; `mac_lane` should
annotate **signed** on `weight_data`/`act_data_r`/`bias_in`/`acc_out` (signedness is
infectious in SV — every operand must be declared signed or the whole expression
evaluates unsigned).

---

## 9. What's verified correct (don't re-litigate)

- Pipeline depth matching, both layers — see §5. The `delay FF` in `digit_top` is
  correct and necessary.
- `weight_addr` sources the **clamped** count, so it can't run off the per-lane ROM.
- Width narrowing from `weight_count` (`$clog2(N_IN+2)`) to `act_addr` (`$clog2(N_IN)`)
  is safe *because* the clamp precedes it.
- `batch` is stable during both `CLEAR` (bias load) and `CAPTURE` (write index), since
  it increments at the end of `CAPTURE`.
- `valid_out` as the batch counter's increment — one pulse per batch, no new signal.
- `fc_layer`'s 1-cycle `done` vs `digit_top`'s held `done` — both intentional.
- `score_debug` 320 = `N_CLASS*32`; `img_idx_r` 5 = `$clog2(NTEST)`; `requant FF` 512 =
  64×8. All match the port declarations.

---

## 10. Remaining work, in order

1. Draw the `requant_relu` internal block diagram (the only one left).
2. Fix the §8 findings in the diagrams.
3. Update `mac_lane.sv` to match its redesigned diagram: remove `bias_in`, add
   `bias_addr` + `bias_selector` + `BIAS_INIT_FILE`, add the buffer FF, right-size
   `LOCAL_MEM_DEPTH` per instantiation (784 / 64).
4. Write `fc_layer.sv`, `argmax10.sv`, `digit_top.sv` from the diagrams.
5. Extend `train_quantize.py`'s `export_mem_files()` for per-lane weight splitting
   (contiguous: lane `L` → `W1_q[L*4:L*4+4]`, batch-major concatenation) and the
   matching per-lane bias split.
6. Update `docs/architecture_spec.md` for the §4 divergences (32 DSPs, dead bit-10
   addressing, revised cycle estimate).
7. Lab 4: bit-exact testbench vs `golden_model.py`.
8. Lab 5: Vivado project, XDC, timing closure, **verify DSP48E1 inference in the
   utilization report** (if Vivado builds LUT multiplier trees instead, the MACC coding
   pattern broke).
9. Lab 6: bitstream, board bring-up, debounce wrapper.
10. Lab 7: README, final block diagram, interview prep.
