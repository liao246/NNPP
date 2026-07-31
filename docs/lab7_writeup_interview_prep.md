# Lab 7 — Writeup and Interview Prep

The project isn't done until someone else (a README reader, or an interviewer) can
understand what you built and why, without you standing next to them. This lab is
about writing that down — and stress-testing that you can actually explain it out
loud, not just on paper.

## Learning objectives

- Write a README that documents the architecture, results, and trade-offs clearly
  enough to serve as your own interview prep material later.
- Produce a block diagram that matches what you actually built.
- Be able to answer, unscripted, why each major decision in this project was made.

## Your task

### 1. README

Should cover, using your **actual** results from Labs 1–6 (not the architecture spec's
estimates — those were predictions; the README reports what really happened):

- **What this is**: one paragraph, plain language — MNIST digit classification on a
  Basys3 FPGA, quantized int8 MLP, custom SystemVerilog datapath.
- **Architecture summary**: topology, quantization scheme, memory layout, pipeline —
  can lean on `architecture_spec.md` but should be your own words, written as if
  explaining it to someone who hasn't read that internal doc.
- **Results**: your actual float accuracy, actual quantized accuracy (Lab 1), your
  actual Vivado utilization and timing numbers (Lab 5, not the estimates), and
  confirmation of real hardware bring-up (Lab 6) with evidence (photo/video).
- **Design decisions and trade-offs**: the specific, named decisions from this
  project — why 14×14 input, why int8, why 16 lanes, why FC2 skips requantization, why
  weight-stationary — each with the reasoning, not just the choice. This section is
  effectively your interview prep written down.
- **What's next / explicitly cut**: full 784-resolution input, a CNN — name these as
  deliberate scope decisions with reasons, not omissions.
- **Verification summary**: what your testbench checks (bit-exact intermediate
  comparison, not just final digit — worth stating explicitly, since it's a real
  differentiator from a weaker "it predicts the right digit most of the time" test),
  and your functional coverage results.

### 2. Block diagram

One diagram showing: image ROM → `fc_layer` (FC1, 16 lanes) → `requant_relu` bank →
`fc_layer` (FC2, 16 lanes, 10 active) → `argmax10` → board outputs, with the DSP48E1
count, BRAM count, and cycle-count numbers labeled on it. This is the diagram you'd
sketch on a whiteboard in an interview — make it match that use case, not a dense
schematic.

### 3. Interview question bank

Practice answering these **using your own actual results and decisions**, not generic
answers — an interviewer will follow up, and generic answers don't survive follow-ups.

- Why int8 for weights/activations instead of int16 or int4? (Tie to DSP48E1
  multiplier width, weight storage size, and your actual measured accuracy at int8.)
- Walk me through what happens between FC1 and FC2 — why is there a shift, and how did
  you pick the shift amount? (This is the exact thing the old version of this project
  got wrong — you should be able to explain both the correct approach and specifically
  what a design that skips this step gets wrong.)
- Why does the output layer skip the requantization the hidden layer has?
- How many DSP48E1 slices does your design use, and why that number specifically (not
  more, not fewer)?
- What's your memory architecture, and why did you choose weight-stationary instead of
  a shared/multi-port memory?
- What was your critical path, and what did you do (if anything) to close timing?
- How did you verify this design was correct? What would a *weaker* verification
  approach have missed?
- What would you do differently, or extend, with more time? (Full 784 input, a CNN,
  per-channel instead of per-layer quantization, pipelining across more stages — pick
  ones you can actually reason about, not just name.)
- What went wrong in your first attempt at this project, and what did you change?
  (This one is specific to your situation — you have a real, honest answer available:
  the quantization pipeline was computed but never implemented in RTL, verification
  only checked final accuracy, and it was never tested on real hardware. That's a
  genuinely good answer — it shows you can identify what's wrong with your own past
  work, which is exactly the kind of self-assessment interviewers are trying to
  probe for.)

## Self-check

- README exists, uses real (not estimated) results throughout, and covers all six
  bullets in the "README" section above.
- Block diagram exists and matches what you actually built (re-check it against your
  final `digit_top.sv`, not an early plan).
- You can answer every question in the interview question bank out loud, without
  notes, in under two minutes each.

## Reflection questions

1. Which of your design decisions would you defend most confidently if pushed back on,
   and which are you least sure about? Why?
2. If an interviewer asked "how do you know your hardware implementation is correct,"
   what's the single strongest piece of evidence from this project you'd point to?
3. Looking back at the old version of this project, what's the most important lesson
   that generalizes beyond just "implement the shift" — something about how you'll
   approach verification or scoping on a future project?
