# MNIST Hardware Accelerator — Lab Manual

This is a from-scratch rebuild of a small neural-network hardware accelerator for
MNIST digit classification, targeting a Digilent Basys3 (Artix-7 XC7A35T) board. It
replaces an earlier version of this project (archived in `old_v1/`) that used a toy
8×8 dataset instead of real MNIST, never implemented the quantization math it claimed
to, and was never tested on real hardware.

**How this manual works:** it's a sequence of labs, each teaching what you need to
know right before you need it, then handing you a starter skeleton (module ports /
function signatures filled in, all actual logic left as `TODO`) to implement yourself.
Claude wrote the teaching content, the architecture spec, and the skeletons — **you
write every line of working logic.** That's the point: you need to be able to defend
every decision in this design in an interview, and the only way that sticks is if you
made the decisions yourself, with guidance.

Each lab ends with a short self-check and a handful of reflection questions written
the way an interviewer might ask them. Don't skip those — if you can't answer them
after finishing a lab, that's worth stopping and reviewing before moving on, not
pushing through.

## Before you start

Read [`architecture_spec.md`](architecture_spec.md) once, fully, before Lab 0. You
don't need to understand every line yet — Lab 0 and Lab 1 will teach the reasoning
behind it — but you should know it's there and what shape the final answer takes. Every
lab refers back to it instead of repeating itself.

## Roadmap

| # | Lab | What you'll be able to do afterward |
|---|---|---|
| 0 | [lab0_nn_fundamentals.md](lab0_nn_fundamentals.md) | Trace a small neural network's forward pass by hand and explain what training actually produces |
| 1 | [lab1_quantization.md](lab1_quantization.md) | Train the float model on real MNIST, quantize it to int8, calibrate the requantization shift, and write a bit-exact integer golden model |
| 2 | [lab2_fpga_concepts.md](lab2_fpga_concepts.md) | Explain BRAM inference, DSP48E1 MACC inference, and what a synthesis utilization/timing report is telling you |
| 3 | [lab3_rtl_datapath.md](lab3_rtl_datapath.md) | Implement the parallel MAC-lane datapath, requant/ReLU pipeline, and top-level FSM in SystemVerilog |
| 4 | [lab4_verification.md](lab4_verification.md) | Build a self-checking, bit-exact testbench with functional coverage |
| 5 | [lab5_synthesis_timing.md](lab5_synthesis_timing.md) | Set up the Vivado project, add constraints, and close timing |
| 6 | [lab6_fpga_bringup.md](lab6_fpga_bringup.md) | Get the design running on real Basys3 hardware |
| 7 | [lab7_writeup_interview_prep.md](lab7_writeup_interview_prep.md) | Write the README/block diagram and prep for interview questions about this project |

## Directory layout

```
NNPP/
  docs/          this manual
  rtl/           SystemVerilog RTL (skeletons live here now; you fill them in)
  tb/            SystemVerilog testbenches
  sw/            Python: training, quantization, golden model
  data/          generated .mem files, reference vectors (empty until Lab 1)
  old_v1/        archived previous attempt — for reference only, not part of the rebuild
```

## Ground rules

- Don't copy logic between `old_v1/` and the new project. It's there so you can see
  what went wrong (Lab 1 and Lab 3 both point back to specific bugs in it), not as a
  source of working code — the whole reason for the rebuild is that its datapath is
  wrong.
- When a lab's skeleton and `architecture_spec.md` seem to disagree, the spec wins —
  say so and ask, don't guess.
- Ask Claude for help debugging, reviewing what you've written, or re-explaining a
  concept at any point — that's the intended use. Asking for the working
  implementation isn't; if that happens, expect to be redirected back to the lab.
