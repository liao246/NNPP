# Lab 4 — Verification

This lab directly fixes the old project's biggest gap: its top-level testbench only
checked the final predicted digit against the *true dataset label*, never against a
Python-computed RTL golden reference. A design can get the right final answer while
being internally wrong (or right by luck on your specific 20 test images) — bit-exact
intermediate-value comparison is what actually proves the datapath is correct.

## Learning objectives

- Explain the difference between "soft" accuracy checking and bit-exact verification,
  and why this project needs the latter.
- Build a self-checking testbench scoreboard pattern without full UVM.
- Write and interpret basic SystemVerilog functional coverage (covergroups/
  coverpoints).

## Background: why bit-exact, not just final-answer

Comparing only `pred_digit` against the true label conflates two very different
things: "is the network accurate" (a property of your *training*, checked in Lab 1)
and "does the hardware correctly implement the quantized algorithm" (a property of
your *RTL*, which is what a hardware testbench should actually be checking). A bug
that corrupts internal accumulator values could still happen to produce the right
final digit on some fraction of your 20 test images by coincidence — especially with
only 20 images, that's a real risk, not a theoretical one. Comparing `acc1[64]`,
`a1[64]`, and `acc2[10]` — not just `pred` — against `golden_model.py`'s dumped
reference vectors (Lab 1, step 8) is what actually isolates "the hardware is right" as
its own claim, checked at the point where the two implementations (Python integer
model, SystemVerilog RTL) should produce identical intermediate values if both
correctly implement the same spec.

## Background: self-checking testbenches without UVM

Full UVM (class-based transactors, sequences, virtual interfaces, factory overrides)
is real overhead that doesn't pay for itself on a project this size, and it's
explicitly out of scope here. What you still want, without UVM's machinery:

- **Directed tests**: your 20 stored images, run one at a time, each checked against
  its dumped golden-model reference vectors.
- **A scoreboard pattern**: not a UVM `uvm_scoreboard` class, just a task/function
  that reads the golden-model reference file for the current image, runs it through
  the DUT, and compares every intermediate value, reporting pass/fail per image with
  enough detail (which neuron, expected vs. actual) to debug a mismatch without
  re-running in a waveform viewer first.
- **A little constrained-random stimulus**, layered on top of the directed images —
  not as a replacement for them. Useful targets here: random weight/pixel values at
  the extremes (0, 255, ±127) to specifically exercise the requantization clamp path
  (Section 2's step 4), since your 20 real MNIST images may never actually hit the
  clamp boundary, and you want that path tested regardless.

## Background: functional coverage basics

A **covergroup** is SystemVerilog's built-in way to answer "did my testing actually
exercise the scenarios I care about," separately from "did the values that got tested
come out correct" (which is the scoreboard's job). A **coverpoint** inside a
covergroup tracks how often a signal fell into each of a set of value ranges (called
"bins") across the whole simulation run.

Minimal example shape (not the actual project's bins — just the mechanics):

```systemverilog
covergroup cg_example @(posedge clk);
    cp_relu_clip: coverpoint relu_clipped_this_cycle {
        bins clipped     = {1};
        bins not_clipped = {0};
    }
endgroup
```

At the end of simulation, a coverage report tells you what percentage of each
coverpoint's bins were actually hit — 100% means every scenario you defined as
interesting happened at least once during the run. This project's target coverpoints
(five, matching `architecture_spec.md`'s risk list and the constraint that full UVM is
out of scope — keep this small and specific, not exhaustive):

1. **Weight/activation extremes** — did a pixel value of exactly `0` and exactly `255`
   both occur; did a weight of exactly `-127` and `127` both occur.
2. **Shift-saturation hit vs. not-hit** — did `requant_relu`'s clamp step (step 4)
   actually clamp at least once, and also pass through unclamped at least once.
3. **ReLU clip vs. pass** — did `requant_relu`'s ReLU step (step 5) zero out a negative
   value at least once, and pass a positive value through unchanged at least once.
4. **Argmax tie-break** — did a tie between two FC2 scores actually get exercised
   (this one may need a targeted directed/random stimulus case rather than relying on
   your 20 real images, since real trained-model ties are rare).
5. **All 20 stored images exercised, and a start-while-busy case** — did every image
   index get run at least once, and did asserting `start` while the DUT is `busy` get
   tested (and correctly ignored, not corrupting an in-flight inference).

## Your task

1. Write unit-level self-checking testbenches for the pieces from Lab 3 that benefit
   from isolated testing before integration — at minimum `requant_relu` (check all 5
   steps against hand-computed or golden-model-derived expected values, including a
   clamp-triggering case and a ReLU-triggering case) and `argmax10` (including a tie
   case). No starter skeleton is provided for these — apply the same pattern Lab 4's
   `tb_digit_top.sv` skeleton demonstrates, at a smaller scale.
2. Complete `tb/tb_digit_top.sv` (skeleton provided): for all 20 stored images, drive
   the DUT, wait for `done`, and compare `acc1[64]`, `a1[64]`, `acc2[10]`, and
   `pred_digit` **bit-exactly** against the reference vectors `golden_model.py` dumped
   in Lab 1. Report pass/fail per image with enough detail to debug a mismatch
   directly from the log.
3. Add the 5 covergroups/coverpoints above (or an equivalent set you can justify) and
   confirm the coverage report shows them all hit by the end of your test run —
   supplementing your 20 directed images with light constrained-random stimulus
   wherever the real data doesn't naturally exercise a bin (the clamp and tie-break
   cases are the most likely to need this).

## Starter skeleton

[`tb/tb_digit_top.sv`](../tb/tb_digit_top.sv) — clock/reset generation and file-reading
boilerplate are filled in; the comparison logic and covergroups are `TODO`.

## Self-check

- All 20 stored images pass bit-exact comparison on `acc1`, `a1`, `acc2`, and
  `pred_digit` — not just `pred_digit`.
- Coverage report shows all 5 target coverpoints/bins hit.
- At least one deliberately-introduced bug (e.g., temporarily flip a `>` to `>=` in
  your argmax tie-break, or skip the rounding step in `requant_relu`) causes the
  testbench to fail loudly and specifically — if you can't get your own testbench to
  catch a bug you know you introduced, it's not actually checking what you think it's
  checking. Revert the deliberate bug once you've confirmed this.

## Reflection questions

1. Why is comparing only `pred_digit` against the true label a weaker test than
   comparing `acc1`/`a1`/`acc2` against a golden model — what specific class of bug
   would the weaker test miss?
2. What's the difference between what a scoreboard checks and what a covergroup
   checks? Could you have 100% functional coverage and a completely broken design at
   the same time? Could you have a perfectly correct design and 0% coverage?
3. Why is constrained-random stimulus used here to supplement the 20 real images
   rather than replace them?
4. Describe, from your own testbench, one specific bug your bit-exact comparison would
   catch that a final-digit-only comparison would not.
