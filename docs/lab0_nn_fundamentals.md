# Lab 0 — Neural Network Fundamentals

No RTL, no Python yet. Before building hardware that runs a neural network, you need
to know what a neural network actually computes — it's less mysterious than it sounds,
and the whole rest of this project is just "implement this specific arithmetic
efficiently in hardware."

## Learning objectives

- Explain what a neuron computes and why stacking them in layers is useful.
- Explain what ReLU does and why it's needed (a network without it would collapse to
  something much less powerful).
- Explain what "training" produces and why hardware only ever needs to do *inference*.
- Trace a tiny forward pass by hand, on paper, correctly.

## Background

### A single neuron

A neuron takes some number of inputs, multiplies each by its own weight, adds them
up, adds one more number called the bias, and that's the output:

```
z = w1*x1 + w2*x2 + ... + wn*xn + b
```

That's it — a weighted sum plus an offset. If that's all a network was, stacking
layers of neurons wouldn't help: a weighted sum of weighted sums is still just a
weighted sum (a linear function), so a 10-layer network of neurons like this would be
mathematically equivalent to a single layer. This is why every layer (except usually
the very last one) applies a nonlinear **activation function** to `z` before passing it
on.

### ReLU

The activation function used in this project is ReLU (Rectified Linear Unit), because
it's the cheapest one to compute:

```
relu(z) = max(0, z)
```

That's the entire function — if `z` is negative, output 0; otherwise, pass `z`
through unchanged. In software this is a comparison and a select. In hardware it's a
comparator and a mux — trivial compared to, say, a sigmoid or tanh, which would need a
lookup table or a polynomial approximation. This is part of why ReLU is the default
choice for hardware-targeted networks, not just software ones.

### Layers, and why "fully connected"

A **fully connected layer** (also called "dense" or, in the weight-matrix view below,
just "FC") means every input connects to every neuron in the layer, each with its own
independent weight. If a layer has 196 inputs and 64 neurons, it has `196 × 64 =
12,544` independent weights — one per (input, neuron) pair — plus 64 biases, one per
neuron.

Written as a matrix equation for a whole layer at once (this is how you should think
about it, and how you'll eventually structure the hardware):

```
z = W @ x + b        # W is (num_neurons x num_inputs), x is (num_inputs,), b is (num_neurons,)
a = relu(z)           # applied elementwise
```

This project's network is two of these layers stacked:

```
input (196 pixels) -> FC1 (64 neurons, ReLU) -> FC2 (10 neurons, no activation) -> argmax -> predicted digit
```

### Why the last layer has no activation, and what argmax does

FC2 produces 10 numbers, one per digit class (0 through 9). These are called
**logits** — raw, unconstrained scores where a bigger number means "the network thinks
this class is more likely." There's no ReLU here because you don't want to clip
negative scores to zero and lose the ability to tell "moderately unlikely" (a small
negative number) from "very unlikely" (a large negative number) — you want the full
ranking preserved.

**Argmax** just means "the index of the largest value." `argmax([1.2, 5.8, -0.3, ...])`
picks out whichever class got the highest score. That's the network's prediction.
Note what this means for hardware: you never need to interpret the logits as
probabilities (no softmax, no division, no exponentials) — you only ever need to know
*which one is biggest*. That's a comparator tree, which is cheap. This is exactly why
Section 2 of the architecture spec skips requantizing FC2's output: a monotonic
rescaling can't change which of 10 numbers is the largest, so there's no reason to pay
for it.

### What training produces, and why hardware doesn't need to do it

**Training** is the process of searching for the values of every weight and bias (all
13,184 weights + 74 biases in this network) that make the network's predictions match
the correct labels as often as possible, usually via gradient descent on a large
labeled dataset (MNIST, in this case: 60,000 labeled training images). Training is
computationally expensive and needs floating-point math with wide dynamic range to
converge reliably — this is why it happens once, offline, in Python (Lab 1), not in
hardware.

**Inference** is just running the forward pass (the matrix equations above) on one new
input, using the weights training already found. This is what the FPGA does — and
critically, inference is nothing but multiply, add, compare, and select. No learning,
no gradients, no floating point required if you're willing to quantize (Lab 1). That's
the entire reason a neural network is something you can reasonably put on an FPGA: the
expensive, iterative, floating-point-hungry part (training) is a one-time offline
cost, and what's left over (inference) is cheap, fixed, repetitive arithmetic — exactly
what hardware is good at.

## Your task: trace a tiny forward pass by hand

No code. On paper (or in a text file, whatever), work through this toy network:

- 3 inputs: `x = [2, -1, 3]`
- 1 hidden layer, 2 neurons, ReLU:
  - Neuron A: weights `[1, 2, -1]`, bias `0`
  - Neuron B: weights `[-1, 0, 1]`, bias `1`
- 1 output layer, 2 neurons (think of these as 2 "classes"), no activation:
  - Neuron C: weights `[1, -1]` (from hidden A, hidden B), bias `0`
  - Neuron D: weights `[0, 2]`, bias `-1`

Compute:
1. The hidden layer's pre-activation values (`z` for A and B).
2. The hidden layer's post-ReLU output (`a` for A and B).
3. The output layer's logits (`z` for C and D) — note these use the *ReLU'd* hidden
   values as input, not the pre-ReLU ones.
4. The argmax — which class does this tiny network predict, C or D?

## Self-check

Work it through, then verify:
- Neuron A: `z = 1*2 + 2*(-1) + (-1)*3 + 0 = 2 - 2 - 3 = -3`, so `a = relu(-3) = 0`.
- Neuron B: `z = -1*2 + 0*(-1) + 1*3 + 1 = -2 + 0 + 3 + 1 = 2`, so `a = relu(2) = 2`.
- Neuron C: `z = 1*0 + (-1)*2 + 0 = -2`.
- Neuron D: `z = 0*0 + 2*2 + (-1) = 3`.
- `argmax([-2, 3])` → index 1 → class D.

If you got a different answer, find where — this exact by-hand mechanics (matrix
multiply, ReLU, argmax) is precisely what 900 cycles of hardware in this project does,
just at a much bigger scale (196→64→10 instead of 3→2→2). If the toy version doesn't
click, the RTL in Lab 3 won't either.

## Reflection questions

Answer these out loud, as if an interviewer just asked them:

1. Why does a multi-layer network need a nonlinear activation function between layers?
   What would happen to a 5-layer network of neurons if none of them had one?
2. Why is ReLU specifically a good fit for a hardware implementation, compared to
   something like sigmoid?
3. Why does the output layer (FC2) skip the activation function?
4. What's the difference between training and inference, and why does that difference
   matter for what you're allowed to do in hardware (float vs. fixed-point, one-time
   vs. repeated cost)?
