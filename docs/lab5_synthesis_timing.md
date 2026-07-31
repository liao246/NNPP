# Lab 5 — Synthesis and Timing Closure

Simulation proves your RTL is *functionally* correct. It says nothing about whether
the physical circuit Vivado builds from it can actually run at 100MHz on real silicon.
This lab is where that gets checked — and where the estimates in
`architecture_spec.md` get confronted with a real report instead of a prediction.

## Learning objectives

- Set up a Vivado project targeting the XC7A35T-1CPG236C from RTL sources.
- Write/adapt XDC constraints for the Basys3's clock, switches, buttons, LEDs, and
  7-segment display.
- Read a utilization report and confirm it matches the DSP48E1/BRAM estimates from
  `architecture_spec.md`.
- Read a timing report, find WNS and the critical path, and fix a negative WNS if you
  hit one.

## Background: recap from Lab 2

Lab 2 covered *why* BRAM/DSP48E1 inference and XDC constraints matter. This lab is
the *how*: actually running synthesis and implementation and reading what comes out.
If any of "BRAM inference," "DSP48E1 MACC mode," "WNS," or "critical path" don't ring
a bell, go back to Lab 2 before continuing here.

## Your task

1. **Create the Vivado project.** New RTL project, part `xc7a35tcpg236-1`, add all
   files under `rtl/` as design sources (not simulation sources), set `digit_top` as
   the top module.
2. **Add constraints.** Start from Digilent's Basys3 Master XDC file (search
   "Digilent Basys3 Master XDC" — it's a standard, widely-distributed reference file
   listing every pin on the board, all commented out by default). Uncomment only the
   lines this project actually uses, matching `architecture_spec.md` Section 5's I/O
   table: the 100MHz clock pin, `SW[4:0]`, `BTNC`, `BTNU`, one 7-segment digit's
   segments + its anode-enable pin, and `LD[15:0]`. Add a clock constraint
   (`create_clock -period 10.000 [get_ports clk]` for 100MHz — period in nanoseconds =
   1000/frequency in MHz).
3. **Run synthesis, then implementation.** Fix any elaboration/synthesis errors first
   (missing port connections, width mismatches — normal at this stage) before worrying
   about timing.
4. **Read the utilization report.** Confirm (or explain any discrepancy from):
   - **DSP48E1 count = 16** (`architecture_spec.md` Section 4). If it's 0 or doesn't
     match, go back to `mac_lane.sv`'s multiply-accumulate coding and compare it
     against Lab 2's DSP48E1 inference pattern.
   - **BRAM primitive count ≈ 16× RAMB18E1 (weight lanes) + 1× RAMB36E1 (image ROM)**
     (`architecture_spec.md` Section 3). Also check whether the bias/label ROMs
     surprised you by consuming a BRAM primitive instead of inferring as LUTROM — if
     they did, that's worth understanding, not necessarily fixing (small ROMs
     sometimes get mapped to BRAM anyway depending on tool heuristics; the point is to
     notice and be able to explain it, not to hit an exact number).
5. **Read the timing report.** Find **WNS (Worst Negative Slack)**.
   - If **WNS ≥ 0**: timing is met at 100MHz. Note the critical path's start/end
     points (the report names them) — compare against the prediction in
     `architecture_spec.md` Section 4 (the requant shifter/clamp or the argmax
     comparator tree, not the DSP48E1 chain).
   - If **WNS < 0**: timing failed. Find the critical path in the report, understand
     *why* it's slow (how many logic levels between registers), and add a pipeline
     register somewhere along that path to break it into two shorter stages. This adds
     one cycle of latency (negligible — see the "9µs is imperceptible" note in
     `architecture_spec.md` Section 4) in exchange for meeting timing. Re-run
     implementation and confirm WNS is now ≥ 0.
6. **Record the actual numbers.** You'll need the real utilization and timing numbers
   (not the architecture spec's estimates) for your README in Lab 7 — take screenshots
   or copy the report tables now.

## Self-check

- Vivado implementation completes with no critical warnings you can't explain.
- Utilization report reviewed and DSP48E1/BRAM counts understood (matching or
  explained if not).
- Timing report shows WNS ≥ 0 at the 100MHz constraint.
- You can point to the specific critical path in the report and explain, in your own
  words, why that path (and not some other one) is the slowest.

## Reflection questions

1. If your DSP48E1 count came back as 0, what's the first thing you'd check, and why?
2. What does "critical path" mean, concretely — is it a single wire, a single gate, or
   something else?
3. Why does adding a pipeline register fix a timing violation, and what does it cost
   you in exchange?
4. Why can't simulation alone ever have told you about a timing violation?
