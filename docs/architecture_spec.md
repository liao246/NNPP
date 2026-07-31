# Architecture Spec — MNIST Digit Classifier Accelerator

This is the locked reference spec. Every lab file points back here. If your
implementation and this document ever disagree, that's a bug in your implementation,
not a hint that the spec is negotiable — the entire point of this project is that the
Python golden model and the RTL implement *exactly* the same integer algorithm, so
they can be compared bit-for-bit.

Target board: Digilent Basys3, Xilinx Artix-7 XC7A35T-1CPG236C (33,280 logic cells,
225KB total BRAM, 90 DSP48E1 slices), Vivado toolchain.

---

## 1. Network Topology

**196 → 64 (ReLU) → 10**, a single-hidden-layer MLP (multi-layer perceptron).

- Input: real MNIST (28×28, 8-bit grayscale), 2×2-average-pooled down to 14×14 = 196
  values in Python preprocessing only. The RTL never sees a 28×28 image — pooling
  happens entirely offline, before any `.mem` file is generated. This is a deliberate,
  disclosed trade-off (see Lab 1) — it's still real MNIST data, just resampled.
- Hidden layer: 64 neurons, ReLU activation.
- Output layer: 10 neurons (one per digit class), no activation function — the raw
  scores ("logits") go straight into an argmax.
- No convolution. A CNN is explicitly out of scope for this build — it's listed as
  future work in Lab 7.

Weight/bias counts you'll be generating and storing:

| Layer | Weights | Biases |
|---|---|---|
| FC1 (196→64) | 196 × 64 = 12,544 | 64 |
| FC2 (64→10) | 64 × 10 = 640 | 10 |

Target accuracy: **≥95%** on the MNIST test set after quantization (a float32 version
of this exact topology should score roughly 96–97.5%; if your quantized accuracy is
far below that, it's a signal to go debug your quantization math, not your training).

---

## 2. Quantization Scheme

Hardware doesn't do floating point here — everything is fixed-point integer math.
Lab 1 builds up *why* piece by piece; this section is the exact spec to implement.

**Weights** — symmetric signed int8, range `[-127, 127]` (127 not 128, so the range is
symmetric around zero with no offset/"zero-point" to track — this is what "symmetric
quantization" means):

```
w_scale = 127 / max(abs(W_fp32))
w_q     = clamp(round(W_fp32 * w_scale), -127, 127)
```
Each layer gets its own `w_scale` (`w_scale_fc1`, `w_scale_fc2`) computed independently.

**Input pixels** — unsigned uint8 `[0, 255]` (grayscale, post-pooling), no scale
applied (`x_scale_input = 1`).

> **RTL gotcha, called out explicitly because it's an easy bug to introduce:** pixels
> are *unsigned* uint8, not signed int8. When you feed a pixel value into a signed
> multiplier (which is what you'll be building for the MAC lanes), you must
> **zero-extend** it to 9 bits, not sign-extend it from 8 bits. Sign-extending a pixel
> value of, say, 200 (`8'b1100_1000`) would treat it as -56. Zero-extending keeps it as
> +200, which is correct.

**Accumulators** — int32 for both layers. Quick overflow sanity check (worst case, not
typical case): FC1's worst possible sum is `196 × 255 × 127 ≈ 6.35M`, which needs about
23 bits — comfortably inside int32. FC2's worst case is `64 × 127 × 127 ≈ 1.03M`, about
21 bits. int32 has enormous headroom in both cases; this is intentional — accumulate
wide, then requantize back down, rather than trying to keep the running sum inside int8
the whole time.

**Bias quantization** — biases must be scaled into the same fixed-point domain as the
accumulator they're added to:

```
bias_q_fc1 = round(bias_fp32_fc1 * w_scale_fc1)                       # x_scale_input = 1
bias_q_fc2 = round(bias_fp32_fc2 * w_scale_fc2 * x_scale_hidden)
x_scale_hidden = w_scale_fc1 / 2^SHIFT1
```

### FC1 requantization — the exact bug this project fixes

The old version of this project computed a requantization shift in Python and then
never applied it in hardware, silently turning an "int8 quantized accelerator" into an
unquantized int32 accelerator with int8 weights bolted onto the input side. Get this
step right and get it bit-identical between Python and RTL. Five steps, in this exact
order:

```
1. acc1        = sum(w1[i] * x[i] for i in 0..195) + bias_q_fc1   # int32
2. acc1_round  = acc1 + (1 <<< (SHIFT1 - 1))                       # round-half-up, int32
3. acc1_shift  = acc1_round >>> SHIFT1                              # arithmetic right shift
4. acc1_clamp  = clamp(acc1_shift, -128, 127)                       # saturate to int8 range
5. a1          = (acc1_clamp < 0) ? 0 : acc1_clamp                  # ReLU; output is int8 (effectively [0,127])
```

**`SHIFT1` is a single calibrated constant**, computed once, offline, in Python (Lab 1
walks through this), and baked into the RTL as a `localparam` — it is *not*
recalculated or configurable at runtime. Expect a value somewhere around 10–12, but
don't hardcode that guess — calibrate it from your actual trained weights:

```
SHIFT1 = ceil(log2(P99.9(abs(acc1)) / 127))
```
(`P99.9` = the 99.9th percentile of `|acc1|` measured over a calibration set of
real training images — using a percentile instead of the true max avoids one outlier
image forcing an overly conservative shift for everything else.)

### FC2 — deliberately *not* requantized

Unlike FC1, FC2's output doesn't feed another quantized multiply — it feeds an argmax.
Requantizing it would add risk (two close logits could saturate to the same clamped
value, destroying the ranking argmax depends on) for zero benefit (a uniform shift
can't change which of 10 numbers is the largest). So:

```
acc2[c] = sum(w2[c][i] * a1[i] for i in 0..63) + bias_q_fc2[c]     # int32, c = 0..9
pred    = argmax(acc2[0..9])                                        # straight on raw int32
```

---

## 3. Memory Architecture

Total data footprint (weights + biases + 20 stored test images + their labels) is
roughly **17.4KB**, against a 225KB BRAM budget on the XC7A35T. Capacity is not the
constraint here — DSP48 count and timing closure are (Lab 5 covers this). Don't spend
effort trying to shrink this further.

**Layout: weight-stationary, output-neuron-parallel, input-broadcast.** This phrase
gets unpacked fully in Lab 2 and Lab 3; the short version: each of the 16 parallel MAC
lanes (Section 4) owns a fixed slice of the weight matrix in its own local memory, and
one activation value gets broadcast to all 16 lanes each cycle. This avoids needing a
wide, multi-port shared memory — the one shared resource per cycle is a single
broadcast value, which is cheap.

| Memory | Contents | Size | Expected primitive |
|---|---|---|---|
| 16× per-lane weight memory | Each lane: 4 FC1 neurons × 196 weights (784B) + up to 1 FC2 neuron × 64 weights (64B), addressed with bit 10 selecting FC1 vs FC2 region | ~848B × 16 lanes | RAMB18E1 (one per lane) |
| 1× image ROM | 20 test images × 196B | 3,920B | RAMB36E1 |
| Bias ROMs (FC1, FC2) | 64×int32 + 10×int32 | 256B + 40B | expect LUTROM (distributed RAM), not a BRAM primitive — verify this in your Lab 5 synthesis report |
| Label ROM | 20 × 4-bit true labels, for the on-board self-check LED | 20B (packed) | expect LUTROM |

---

## 4. Pipeline / Parallelism

**16 parallel MAC lanes → 16 DSP48E1 slices** (of 90 available on the XC7A35T — 17.8%
utilization, leaving generous margin). Each lane performs one signed multiply-
accumulate per cycle, coded so Vivado maps it onto a single DSP48E1 in its internal
MACC mode (multiplier + adder + accumulate register, all inside one hardware primitive
— Lab 2 explains the specific RTL coding pattern this requires). **Verify this in your
utilization report** — if Vivado instead builds 16 separate multiplier trees out of
LUTs, something in your coding style broke DSP inference, and that's worth debugging
before you move on to timing closure.

**Topology:** each lane owns a disjoint set of output neurons and its own local weight
memory (Section 3). On every clock cycle, one activation value is broadcast to all 16
lanes simultaneously; each lane multiplies it by its own locally-stored weight for that
input index and accumulates.

**Cycle estimate** (at 100MHz, driven directly off the Basys3's onboard oscillator —
no clock-management IP needed for a design this size):

| Phase | Work | Cycles (approx) |
|---|---|---|
| FC1 | 64 neurons ÷ 16 lanes = 4 batches × 196 inputs/batch | ~800 |
| FC1 requantize | pipelined, 16-wide | ~10 |
| FC2 | 10 of 16 lanes active × 64 inputs | ~68 |
| Argmax | 10-way comparator tree, ~4 levels | ~5 |
| FSM/handshake overhead | | ~15 |
| **Total** | | **~900 cycles ≈ 9µs/inference** |

That's about a **14.6× speedup** over a fully-serial (1 MAC/cycle) baseline — close to,
but short of, the theoretical 16× ceiling. The shortfall comes from FC2 only using
10 of 16 lanes and fixed FSM overhead; both are worth being able to explain precisely
in an interview, not just cite as "some overhead."

9µs is far too fast to see anything happen on a button press — latch a free-running
cycle counter (captured at `done`) onto the spare LEDs. This doubles as both a bring-up
sanity check and your actual demo number.

---

## 5. Top-Level Interface

| Basys3 I/O | Function |
|---|---|
| Clock, pin W5 (100MHz) | System clock, used directly |
| `SW[4:0]` | Selects stored test image, 0–19 (values above 19 clamp to 19, no wraparound) |
| `BTNC` | `start` — debounce it, then detect the rising edge to produce a single clean one-cycle pulse |
| `BTNU` | `rst` |
| 7-segment display, 1 digit only | Predicted digit 0–9. Only one digit is ever driven — since you're never displaying more than one decimal digit, you don't need digit-multiplexing/refresh logic at all. |
| `LD[0]` | `busy` |
| `LD[1]` | `done` (held until the next `start`) |
| `LD[2]` | `correct` — `pred_digit == stored true label` for the selected image (an on-board self-check, using the label ROM from Section 3) |
| `LD[15:4]` | last-inference cycle count, for the demo/debug readout described above |

**`digit_top` port list:**

| Port | Direction | Width | Purpose |
|---|---|---|---|
| `clk` | in | 1 | 100MHz board clock |
| `rst` | in | 1 | synchronous reset (pick one polarity and be consistent everywhere — don't mix an active-low port name with active-high board buttons) |
| `start` | in | 1 | debounced, single-cycle start pulse |
| `img_idx` | in | 5 | selects stored test image, 0–19 |
| `busy` | out | 1 | high while an inference is in flight |
| `done` | out | 1 | high once inference completes |
| `pred_digit` | out | 4 | argmax result, 0–9 |
| `correct` | out | 1 | `pred_digit == stored label` for `img_idx` |
| `cycle_count` | out | 16 | cycles elapsed during the last inference |
| `score_debug` | out | 10×32 | raw FC2 accumulators — simulation/ILA debug only, doesn't need to reach a real board I/O pin |

---

## Quick Reference: What Gets Built Where

| Piece | File | Lab |
|---|---|---|
| Train float MLP, quantize, calibrate `SHIFT1`, export `.mem` files | `sw/train_quantize.py` | Lab 1 |
| Bit-exact integer-only reference model | `sw/golden_model.py` | Lab 1 |
| Single MAC lane (DSP48E1 + local weight BRAM) | `rtl/mac_lane.sv` | Lab 3 |
| FC1 requantize/ReLU pipeline | `rtl/requant_relu.sv` | Lab 3 |
| 16-lane layer wrapper + batch FSM (reused for FC1 and FC2) | `rtl/fc_layer.sv` | Lab 3 |
| Argmax over 10 scores | `rtl/argmax10.sv` | Lab 3 |
| Top-level integration + board I/O | `rtl/digit_top.sv` | Lab 3 |
| Bit-exact + coverage-driven testbench | `tb/tb_digit_top.sv` | Lab 4 |
| Vivado project, XDC constraints | (new, no starter given) | Lab 5 |
| Bitstream + board bring-up | — | Lab 6 |
| README, block diagram, interview prep | `README.md` | Lab 7 |
