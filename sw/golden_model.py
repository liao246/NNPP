"""
sw/golden_model.py

Lab 1 skeleton (final step) / Lab 4 dependency. See docs/lab1_quantization.md
and docs/architecture_spec.md Section 2 before writing any code here.

THIS FILE IS THE SPEC OF RECORD your RTL gets checked against in Lab 4. It must
be an integer-only, bit-exact implementation of the 5-step FC1 requantization
pipeline plus raw-accumulator FC2 argmax -- no floats anywhere in the inference
path. If this file and your RTL ever disagree, that disagreement is exactly
what Lab 4's testbench exists to catch.

Run standalone (python sw/golden_model.py) to regenerate reference vectors for
all 20 stored test images in data/, for the testbench built in Lab 4 to read.
"""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_quantized_params(data_dir=DATA_DIR):
    """
    Read w1.mem, b1.mem, w2.mem, b2.mem, meta.json (written by
    train_quantize.py's export_mem_files) back into numpy arrays.

    Must parse the exact hex format export_mem_files() wrote -- keep these two
    functions in sync as you write them.

    Returns
    -------
    W1_q : np.ndarray, shape (64, 196), dtype int8
    bias_q_fc1 : np.ndarray, shape (64,), dtype int32
    W2_q : np.ndarray, shape (10, 64), dtype int8
    bias_q_fc2 : np.ndarray, shape (10,), dtype int32
    shift1 : int
    """
    raise NotImplementedError("TODO: parse .mem files + meta.json")


def fc1_forward_int(x_pixels_u8, W1_q, bias_q_fc1, shift1):
    """
    The 5-step FC1 pipeline from architecture_spec.md Section 2, EXACTLY, for
    one image. This is the function whose per-neuron intermediate value
    (acc1, before the shift) and final output (a1, after ReLU) both need to be
    dumped for Lab 4's bit-exact RTL comparison -- don't just return the final
    a1 and discard acc1.

    Parameters
    ----------
    x_pixels_u8 : np.ndarray, shape (196,), dtype uint8 (values 0..255 -- do
        NOT treat these as signed; zero-extend when combining with sign math,
        matching the RTL gotcha called out in architecture_spec.md Section 2)
    W1_q : np.ndarray, shape (64, 196), dtype int8
    bias_q_fc1 : np.ndarray, shape (64,), dtype int32
    shift1 : int

    Returns
    -------
    acc1 : np.ndarray, shape (64,), dtype int32   (step 1 output, pre-shift -- needed for RTL comparison)
    a1 : np.ndarray, shape (64,), dtype int8       (step 5 output, final FC1 activations)
    """
    raise NotImplementedError("TODO: implement the 5-step FC1 pipeline")


def fc2_forward_int(a1_i8, W2_q, bias_q_fc2):
    """
    FC2: raw int32 accumulation, NO requantization (see architecture_spec.md
    Section 2 for why FC2 skips the shift/clamp/ReLU that FC1 has).

    Parameters
    ----------
    a1_i8 : np.ndarray, shape (64,), dtype int8 (FC1's output)
    W2_q : np.ndarray, shape (10, 64), dtype int8
    bias_q_fc2 : np.ndarray, shape (10,), dtype int32

    Returns
    -------
    acc2 : np.ndarray, shape (10,), dtype int32
    """
    raise NotImplementedError("TODO: implement FC2 raw accumulation")


def predict(x_pixels_u8, W1_q, bias_q_fc1, W2_q, bias_q_fc2, shift1):
    """
    Full integer-only forward pass for one image: fc1_forward_int ->
    fc2_forward_int -> argmax. This is the function train_quantize.py's
    quantized_inference_one() should end up matching bit-for-bit -- consider
    having train_quantize.py import and call this function directly instead
    of maintaining a second copy of the same logic.

    Returns
    -------
    dict with keys: "acc1" (64,), "a1" (64,), "acc2" (10,), "pred" (int)
        All four values are what Lab 4's testbench compares against RTL --
        NOT just "pred". A design that gets pred right by accident while acc1
        or acc2 are wrong is exactly the kind of bug bit-exact comparison
        exists to catch.
    """
    raise NotImplementedError("TODO: full forward pass + argmax")


def dump_reference_vectors(data_dir=DATA_DIR):
    """
    For each of the 20 stored test images in data/, run predict() and write
    out acc1[64], a1[64], acc2[10], pred to a format Lab 4's SystemVerilog
    testbench can read with $fscanf (plain decimal integers, one per line or
    whatever layout you document clearly -- match it exactly on the SV side).

    This is what makes Lab 4's testbench comparison bit-exact rather than
    "final digit only" -- the exact gap in the old version of this project.
    """
    raise NotImplementedError("TODO: dump per-image reference vectors for Lab 4")


def main():
    W1_q, bias_q_fc1, W2_q, bias_q_fc2, shift1 = load_quantized_params()
    dump_reference_vectors()
    # TODO: also print a quick summary (accuracy over the 20 stored images vs
    # their true labels) as a fast sanity check that this file's independent
    # re-implementation agrees with train_quantize.py's own validation number
    # from Lab 1 step 6 -- two different code paths computing the same
    # quantized accuracy is itself a useful cross-check.


if __name__ == "__main__":
    main()
