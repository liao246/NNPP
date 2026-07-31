# Lab 2 — FPGA Concepts Primer

You already know how to write SystemVerilog and simulate/test it — that's not what
this lab teaches. This lab covers the specific things that matter once you're
targeting real synthesis on a real FPGA for a datapath like this one: how memory and
multiply-accumulate hardware actually get built from your RTL, and how to read the
reports that tell you whether it worked. No task/skeleton yet — this is pure grounding
for Lab 3, where you'll write the RTL these concepts apply to.

## Learning objectives

- Explain what BRAM inference is and what RTL pattern causes Vivado to map a memory
  array onto a RAMB18E1/RAMB36E1 primitive instead of registers or LUTs.
- Explain what DSP48E1 MACC-mode inference is and the specific coding pattern
  (registered inputs, registered accumulator) that gets a multiply-accumulate mapped
  onto one DSP48E1 instead of built from fabric logic.
- Explain what a constraints (XDC) file does and why synthesis needs one.
- Read a utilization report and a timing report well enough to know whether your
  design matched its own resource/timing estimates.

## BRAM inference

An FPGA doesn't have "arrays" as a primitive — it has two ways to store data: registers
(flip-flops, one bit each, expensive per bit, fast) and dedicated Block RAM primitives
(RAMB18E1 = 18Kb, RAMB36E1 = 36Kb on this device family, cheap per bit, one or two
read/write ports, needs an address+clock like a small synchronous memory chip).

When you write SystemVerilog like this:

```systemverilog
logic [7:0] weight_mem [0:1087];
initial $readmemh("data/w1.mem", weight_mem);
```

...simulators just allocate this like any array. But **synthesis tools recognize this
pattern** (an array, accessed through registered address inputs, with a clocked read)
and map it onto a BRAM primitive automatically — this is called **BRAM inference**, and
it's why you don't need to manually instantiate a `RAMB18E1` primitive by name; you
write behavioral code that looks like a memory, and the synthesis tool recognizes the
pattern and infers the right primitive. `$readmemh` in an `initial` block is exactly
how you preload a BRAM's contents at configuration time (bitstream load), which is how
this project gets its weights onto the chip with no runtime loading step needed — the
weights are baked into the bitstream.

**What breaks inference:** if your read path isn't registered (i.e., you read
combinationally off the address in the same cycle it changes, rather than registering
either the address or the output), Vivado may build the "memory" out of distributed
LUT-RAM or plain registers instead of a dedicated BRAM primitive — functionally
identical in simulation, but a much worse use of chip resources. This is exactly why
Lab 5 asks you to check the utilization report for the expected BRAM primitive
counts, not just assume they showed up.

## DSP48E1 MACC-mode inference

A DSP48E1 slice is a dedicated hardware block that can do a signed multiply, add, and
accumulate — all inside one primitive, in one cycle, once pipelined — instead of
synthesis building an equivalent circuit out of general-purpose LUTs (which would be
far slower and use far more area for the same multiply-accumulate operation). Getting
Vivado to actually use a DSP48E1 for your MAC lanes (rather than building a multiplier
out of LUT fabric) depends on writing the multiply-accumulate in a specific registered
style — from Xilinx's UG901 synthesis guide, the pattern is roughly:

```systemverilog
always_ff @(posedge clk) begin
    a_reg   <= a_in;              // register the multiplier's A input
    b_reg   <= b_in;              // register the multiplier's B input
    acc_reg <= acc_reg + (a_reg * b_reg);   // registered accumulate
end
```

The key idea: the DSP48E1 has its own internal registers on the A input, B input, and
accumulator output, and Vivado's inference logic looks for RTL that matches that
internal register structure. If your multiply-accumulate is purely combinational (no
registers at all), or structured in a way that doesn't line up with the DSP48E1's
internal pipeline stages, synthesis will still produce a *correct* circuit — just one
built from LUTs instead of the dedicated multiplier hardware, using dramatically more
fabric resources for the same 90-slices-available budget this project is trying to use
efficiently (16 of 90, see `architecture_spec.md` Section 4). This is exactly why Lab 5
tells you to explicitly check for "DSP48E1: 16" in your utilization report rather than
just trusting that it worked because the design simulated correctly — simulation
can't tell you which physical primitive your multiply-accumulate landed on.

## Why weight-stationary/broadcast avoids a memory bottleneck

If you tried to have all 16 MAC lanes read from one shared weight memory, you'd need a
16-read-port memory, which BRAM primitives don't have (they typically offer 1 or 2
ports). The **weight-stationary** design in this project's spec sidesteps that: each
lane gets its *own* small local BRAM holding just its slice of the weight matrix, so
there are 16 independent single-port memories instead of one impossible 16-port one.
The only thing shared across lanes per cycle is a single broadcast activation value —
routing one value to 16 destinations is cheap; routing 16 independent memory read
requests to one memory is not. This is the concrete hardware reason behind the memory
layout in `architecture_spec.md` Section 3.

## Constraints (XDC) files

RTL describes *behavior*; it says nothing about which physical pin the 100MHz clock
arrives on, which pin `SW[0]` is wired to, or what clock frequency you're promising the
tool you'll actually hit. A **Xilinx Design Constraints (XDC)** file supplies exactly
that information: pin assignments (`set_property PACKAGE_PIN ... [get_ports ...]`) and
timing constraints (`create_clock -period 10.000 [get_ports clk]` for a 100MHz clock,
since period = 1/frequency = 10ns). Without pin constraints, Vivado has no way to know
your `img_idx` port should connect to the physical switches on the board. Without a
clock constraint, Vivado's timing analysis has nothing to check your design against —
it wouldn't even attempt to tell you whether you'll meet timing. Lab 5 walks through
getting Digilent's Basys3 Master XDC file and uncommenting the specific lines this
project's top-level ports need.

## Reading a utilization report

After synthesis (or implementation), Vivado produces a utilization report listing how
many of each resource type your design consumed: LUTs, flip-flops, block RAM
(RAMB18E1/RAMB36E1 counts), and DSP48E1 slices. For this project, the numbers to
specifically check against the architecture spec's estimates (Lab 5 goes through this
in detail) are: **DSP48E1 count should read 16** (Section 4), and **BRAM primitive
count should be roughly 16× RAMB18E1 + 1× RAMB36E1** (Section 3) — if either number is
off, that's a concrete signal something about your coding style isn't inferring the
primitive you intended, worth chasing down before moving on to timing closure.

## Reading a timing report

A timing report tells you whether your design can actually run at the clock frequency
you constrained it to (100MHz here). The single most important number is **WNS —
Worst Negative Slack**. Positive WNS means your slowest path finishes with time to
spare before the next clock edge (timing met). Negative WNS means the slowest path
("critical path") takes longer than one clock period allows — your design is not
guaranteed to work correctly at that frequency on real silicon, even though it will
simulate perfectly (simulation doesn't model real gate/wire delay unless you're doing
gate-level timing simulation, which this project doesn't need). The report will also
name the specific start/end points of the critical path — for this design, a
reasonable prediction (per `architecture_spec.md` Section 4) is that it runs through
the requantization shifter/clamp logic or the argmax comparator tree, not through the
DSP48E1 MAC chain itself (which is a well-pipelined, tool-optimized primitive). If WNS
comes back negative, the standard fix is adding a pipeline register somewhere along
that critical path — Lab 5 covers this concretely once you have a real report to look
at.

## Self-check

Before moving to Lab 3, you should be able to answer, without looking back at this
document:

- What's the difference between how registers and BRAM primitives store data, and why
  does that difference matter for a memory holding 12,544 weight values?
- What specific RTL coding pattern makes a multiply-accumulate map onto a DSP48E1
  instead of fabric LUTs?
- Why does this design give each MAC lane its own weight memory instead of one shared
  memory all 16 lanes read from?
- What does WNS mean, and what does a negative WNS actually tell you (and not tell
  you) about whether your design "works"?

## Reflection questions

1. If Vivado's utilization report showed 0 DSP48E1 slices used and a huge LUT count
   instead, what would that tell you about your RTL, and where would you start
   looking?
2. Why doesn't a negative WNS show up as a bug during RTL simulation?
3. Explain, in your own words, why "weight-stationary" is a memory-architecture
   decision, not just a naming choice — what alternative would run into a real
   hardware limitation?
