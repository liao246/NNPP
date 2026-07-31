# Lab 6 — FPGA Bring-Up

This is the lab that makes the project actually true to its goal: tested on real
hardware, not just simulated. The old version of this project never got here at all.
Per the risk flags in `architecture_spec.md`, do the "hello world" exercise below
*early* (ideally in parallel with Lab 3, not after Lab 5) — board/toolchain issues are
much cheaper to find before your real design is ready to test.

## Learning objectives

- Explain why mechanical button presses need debouncing, and implement it.
- Drive a single seven-segment display digit correctly (including figuring out your
  board's segment polarity).
- Program a bitstream onto the Basys3 and interpret its behavior against simulation.
- Have a debugging plan ready for when hardware doesn't match simulation (it's common,
  not a sign you did something wrong).

## Background: why buttons need debouncing

A mechanical pushbutton doesn't transition cleanly from 0 to 1 — the physical contacts
bounce for a few milliseconds, producing a burst of rapid 0/1/0/1 transitions before
settling. At a 100MHz clock, a few milliseconds is *hundreds of thousands* of clock
cycles, so an undebounced button read directly by your FSM will very likely be
interpreted as many separate button presses instead of one, triggering `start`
repeatedly or corrupting an in-flight inference. **Debouncing** means holding a
button's reported value stable until the raw input has stayed constant for some
minimum time (a counter that resets on every raw transition and only updates the
debounced output once it saturates), then **edge-detecting** the debounced signal to
produce the single clean one-cycle `start` pulse that `digit_top` expects (recall from
Lab 3: `digit_top` itself assumes clean inputs — this logic lives in a board-level
wrapper around it).

## Background: seven-segment display polarity

The Basys3's seven-segment display is common-anode, which typically means each segment
turns **on** when its control signal is driven **low**, not high — the opposite of
what you'd naively guess. Check this against the board's reference manual before
wiring up your digit-to-segment decode logic, or you'll get an inverted (or entirely
dark, or entirely lit) display and waste time debugging logic that was never wrong.
Since this project only ever displays one digit (0–9) on one 7-segment position (per
`architecture_spec.md` Section 5 — no digit-multiplexing needed), the anode-enable
signal for that one position is simply held constantly active, and the other three
digit positions' anodes held inactive.

## Your task

1. **Do this first, before your real design is ready** — a throwaway "hello world":
   wire all 16 switches directly to all 16 LEDs, and one button to one segment of the
   7-segment display, program it, and confirm it works. This is purely to catch XDC
   pin mistakes, board driver/cable issues, or Vivado hardware-manager setup problems
   in isolation, before they're tangled up with debugging your actual accelerator.
2. **Write a debounce module** and instantiate it (along with rising-edge detection)
   in a new top-level board wrapper (e.g. `rtl/board_top.sv` — no skeleton is provided;
   this is straightforward enough, and specific enough to your button-debounce design
   choices, to write from scratch) that instantiates `digit_top` and connects its
   `start`/`rst` inputs to debounced/edge-detected `BTNC`/`BTNU`.
3. **Wire up the 7-segment decode** (digit 0–9 → 7 segment signals, with correct
   polarity) and the LED assignments from `architecture_spec.md` Section 5
   (`busy`/`done`/`correct`/cycle count).
4. **Program the board** with your real design and sweep through all 20 stored images
   via the switches, confirming the predicted digit, `correct` LED, and cycle count
   match what Lab 4's simulation predicted for each one.
5. **Capture evidence** — photos or video of the board correctly classifying at least
   a few different digits, switching between them. You'll want this for Lab 7's
   README.

## Debug checklist, if hardware doesn't match simulation

This is common enough to plan for, not a sign of failure:
- Re-check 7-segment polarity and unused-digit blanking first — a huge fraction of
  "nothing displays correctly" bugs are exactly this.
- Confirm the bitstream you programmed is actually the one you most recently built
  (a surprisingly common mistake — check the build timestamp).
- If a *specific* image gives a wrong answer on hardware but was correct in
  simulation, suspect a timing violation that Lab 5's static timing analysis missed
  covering, or an XDC constraint that's wrong for a signal not in your critical-path
  analysis (e.g., an unconstrained input causing metastability) — an Integrated Logic
  Analyzer (ILA) core attached to `score_debug` or key internal signals can let you
  observe real hardware behavior directly, similar to a simulation waveform.
- If *every* image gives the same wrong answer, or the design never asserts `done`,
  suspect the board wrapper (debounce/edge-detect) rather than `digit_top` itself,
  since `digit_top` was already verified in Lab 4's simulation.

## Self-check

- The "hello world" exercise worked before you moved on to the real design.
- All 20 stored images, swept via the switches, produce the same predicted digit and
  `correct` result on real hardware as they did in Lab 4's simulation.
- You have photo/video evidence of this for Lab 7.

## Reflection questions

1. Why does a mechanical button need debouncing but a signal coming from another
   synchronous digital circuit doesn't?
2. What's the actual difference between what an ILA lets you observe and what a
   simulation waveform lets you observe?
3. If your design's `done` signal never asserted on hardware despite working
   perfectly in simulation, what's your first hypothesis and why?
