# Lab 1 — Real MNIST, Quantization, and the Golden Model

This is the biggest software lab and arguably the most important one for your
interview story — the quantization math here is exactly what the old version of this
project got wrong, and being able to explain precisely *why* each step exists is what
separates "I quantized a network" from "I can reason about fixed-point hardware
design."

## Learning objectives

- Explain why hardware inference uses fixed-point integers instead of floating point.
- Explain symmetric int8 quantization and how a scale factor is derived from a weight
  matrix.
- Explain why the accumulator needs to be wider than the multiply inputs, and why a
  requantization (shift + clamp) step is needed between layers.
- Produce: a trained float model on real MNIST, a quantized int8 version of it, a
  calibrated `SHIFT1` constant, and a bit-exact Python "golden model" that will later
  be the ground truth your RTL gets checked against.

## Background: why not just use floats in hardware?

Floating-point arithmetic (IEEE 754) represents numbers with a sign, exponent, and
mantissa, and doing add/multiply on that format correctly requires normalizing,
aligning exponents, rounding — real logic, real area, real latency. A hardware
floating-point multiply-add unit is dramatically more expensive (in LUTs, DSP usage,
critical path) than a fixed-point one. For a small student accelerator project,
floating point is both unnecessary (this network doesn't need huge dynamic range) and
a distraction from the actual point of the project (quantized fixed-point datapath
design, which is what real edge-inference accelerators do).

**Fixed-point** means: pick a scale factor once, represent every number as a plain
integer, and remember that the "real" value is `integer / scale`. As long as you're
consistent about what scale each quantity is at, ordinary integer add/multiply
hardware does the job — the DSP48E1 slices on your FPGA are integer multiply-
accumulate units already; this is exactly what they're built for.

## Background: symmetric int8 weight quantization

Each layer's weight matrix `W` (floating point, from training) gets converted to int8
by finding the largest magnitude weight in that layer and scaling so that value maps
to 127:

```
w_scale = 127 / max(abs(W))
w_q     = clamp(round(W * w_scale), -127, 127)
```

This is called **symmetric** quantization because the representable range is
`[-127, 127]` — symmetric around zero, no separate "zero point" offset to track (some
quantization schemes use an asymmetric range with a zero-point, which is more
accurate for skewed distributions but adds an extra offset term to every multiply;
not worth the complexity here). Each layer gets its own independently-computed
`w_scale` because different layers' weight distributions can have very different
magnitudes.

**Why quantize weights but not just clip them?** Rounding to the nearest representable
int8 value after scaling (rather than just clipping floats to `[-127,127]` without
scaling) is what makes the full int8 range actually get used — a weight matrix whose
values are all between -0.05 and 0.05 would round to all-zero if you just clipped it,
but scales up to a well-spread int8 distribution correctly.

## Background: why the accumulator is wider than the inputs, and what a requantization shift does

Multiplying two int8 values gives a result that needs up to 16 bits to represent
exactly (`127 * 127 = 16,129`), and summing 196 of those (FC1) needs even more
headroom — this project uses int32 accumulators, with a comfortable margin (see the
overflow arithmetic in `architecture_spec.md` Section 2). This is completely standard:
**accumulate wide, then requantize back down**, rather than trying to force
intermediate sums to stay inside int8 the entire time (which would require clamping
after every single multiply-add, losing precision constantly and being slower).

But there's a problem: FC1's output (a 32-bit accumulator value) needs to become FC2's
*input*, and FC2 is expecting int8 activations (so its own weights × activations
multiply stays in a sane range). That conversion — wide accumulator back down to int8
— is **requantization**, and it's not just clamping; it's a right-shift (division by a
power of 2) chosen so that the *typical* accumulator value lands inside the int8
range, followed by a clamp for the rare cases that don't. Read the exact 5-step
pipeline in `architecture_spec.md` Section 2 now if you haven't already — this lab's
task is to make that pipeline real, both in Python and (later, Lab 3) in hardware.

**This is precisely the step the old version of this project computed
(`SHIFT1` was in its `meta.json`) but never actually implemented in RTL.** Its FC1
module just ReLU'd the raw 32-bit sum and fed that directly into FC2, which is a
functionally different (and unintended) algorithm than what the Python-side
quantization math assumed. Getting this step right — and identical between your
Python golden model and your RTL — is the main technical fix this rebuild makes.

## Why FC2 has no requantization step

Covered in `architecture_spec.md` Section 2 and Lab 0's reflection questions: FC2 feeds
an argmax, not another quantized multiply, and a monotonic rescale can't change which
of 10 values is biggest. Requantizing it would only add risk (two logits saturating to
the same clamped value) for no benefit.

## Your task

1. **Get real MNIST.** Use `torchvision.datasets.MNIST` (or `tensorflow.keras.datasets.mnist`
   if you'd rather use Keras/TF) to download the actual 60,000-train/10,000-test,
   28×28, 8-bit grayscale dataset. Do not reuse `sklearn.datasets.load_digits()` —
   that's the 8×8 toy dataset the old version of this project mistakenly used.
2. **Pool to 14×14.** Apply 2×2 average pooling to every image, producing 196 values
   per image (still 0–255 range grayscale, just at 14×14 resolution). This is a
   disclosed, deliberate resolution trade-off — see `architecture_spec.md` Section 1
   for why.
3. **Train the float model.** A plain 196→64(ReLU)→10 MLP, trained with any reasonable
   framework/optimizer (PyTorch is a fine default) until it converges. Sanity-check:
   you should see something in the 96–97.5% test-accuracy range. If you're far below
   that, something is wrong with training/data before you even get to quantization —
   don't proceed to quantizing a broken model.
4. **Quantize.** Implement symmetric int8 weight quantization per-layer as above.
   Quantize the biases into the correct scale domain (formulas in
   `architecture_spec.md` Section 2 — note FC1 and FC2 biases use different formulas,
   because FC2's formula depends on `SHIFT1`, which you calibrate next).
5. **Calibrate `SHIFT1`.** Run a calibration set (e.g., 1,000 training images) through
   FC1's int32 accumulation (weights × pixels + bias, no shift yet), record the
   distribution of `|acc1|` across all neurons and all calibration images, and compute
   `SHIFT1` from the 99.9th percentile per the formula in `architecture_spec.md`.
6. **Validate.** Run the *quantized* model (the exact 5-step FC1 pipeline + raw-argmax
   FC2, as an integer-only computation, no floats anywhere) over the full 10,000-image
   test set. Confirm you hit the ≥95% target. If quantized accuracy is far below the
   float baseline, that's a signal of a bug in your quantization/requantization math
   (wrong shift direction, wrong clamp bounds, sign-extension mistake on the pixel
   values — see the "RTL gotcha" callout in the architecture spec, which applies to
   your Python model too if you're not careful about numpy dtypes) — go find it before
   moving on; don't paper over it by loosening the accuracy target.
7. **Export.** Write out `.mem` files for weights/biases/20 test images (2 per digit
   class, chosen so you have an easy accuracy sanity set) in whatever hex/text format
   your future RTL testbench will read with `$readmemh`, plus the true labels for
   those 20 images, plus `data/meta.json` recording the topology and calibrated
   `SHIFT1`.
8. **Write the golden model.** `sw/golden_model.py` (skeleton provided) is the
   integer-only, bit-exact reference your RTL will be checked against in Lab 4 — it
   must implement precisely the same 5-step FC1 pipeline and raw-accumulator FC2
   argmax as the spec, with no floating point anywhere in the inference path. For each
   of your 20 stored test images, dump the intermediate values (`acc1[64]`, `a1[64]`,
   `acc2[10]`, `pred`) to files — Lab 4's testbench needs these to do a bit-exact
   comparison against RTL, not just a final-answer comparison.

## Starter skeletons

Two files with function signatures and docstrings filled in, bodies left as `TODO`:

- [`sw/train_quantize.py`](../sw/train_quantize.py) — steps 1–7 above.
- [`sw/golden_model.py`](../sw/golden_model.py) — step 8, the integer-only reference.

Read every docstring before writing code — they point back to the specific
architecture spec section each function must match.

## Self-check

- Float model test accuracy is roughly 96–97.5%.
- Quantized (integer-only) model test accuracy is ≥95% on the full 10,000-image test
  set.
- `data/meta.json` exists and records your calibrated `SHIFT1` (not a guessed value).
- For at least one test image, manually trace through `golden_model.py`'s FC1
  computation for one neuron by hand/calculator and confirm it matches the script's
  own output — this is the same "trace it by hand" discipline as Lab 0, now applied to
  the real quantized pipeline.
- `data/` contains `.mem` files for weights, biases, 20 images, 20 labels, plus
  per-image intermediate reference vectors.

## Reflection questions

1. Why does FC1 need a requantization step but FC2 doesn't?
2. Walk through the 5-step FC1 requantization pipeline from memory. What does each
   step do, and what would break (functionally, not just "it'd be less accurate") if
   you skipped the rounding step (step 2) and just truncated via the shift?
3. Why is `SHIFT1` calibrated from the 99.9th percentile of `|acc1|` rather than the
   true maximum? What failure mode does the maximum-based approach risk?
4. Why does quantizing FC2's bias require knowing `SHIFT1`, while FC1's bias
   quantization doesn't depend on any shift?
5. If someone asked "why int8 and not int16 or int4," what would you say, in terms of
   DSP48E1 multiplier width, weight storage size, and accuracy?
