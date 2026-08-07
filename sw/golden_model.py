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


def _read_hex_lines(path):
    with open(path) as f:
        return [int(line.strip(), 16) for line in f if line.strip()]


def _read_hex_i8(path):
    """Inverse of train_quantize.py's _to_hex_u8 -- two's-complement decode."""
    vals = [(v - 256) if v >= 128 else v for v in _read_hex_lines(path)]
    return np.array(vals, dtype=np.int8)


def _read_hex_i32(path):
    """Inverse of train_quantize.py's _to_hex_u32 -- two's-complement decode."""
    vals = [(v - (1 << 32)) if v >= (1 << 31) else v for v in _read_hex_lines(path)]
    return np.array(vals, dtype=np.int32)


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
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)

    n_input = meta["N_INPUT"]
    hidden = meta["HIDDEN"]
    n_class = meta["N_CLASS"]
    shift1 = meta["SHIFT1"]

    W1_q = _read_hex_i8(data_dir / "w1.mem").reshape(hidden, n_input)
    bias_q_fc1 = _read_hex_i32(data_dir / "b1.mem")
    W2_q = _read_hex_i8(data_dir / "w2.mem").reshape(n_class, hidden)
    bias_q_fc2 = _read_hex_i32(data_dir / "b2.mem")

    return W1_q, bias_q_fc1, W2_q, bias_q_fc2, shift1


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
    x_wide = x_pixels_u8.astype(np.int64)
    W1_wide = W1_q.astype(np.int64)
    bias1_wide = bias_q_fc1.astype(np.int64)

    acc1 = W1_wide @ x_wide + bias1_wide             # step 1

    acc1_round = acc1 + (1 << (shift1 - 1))          # step 2
    acc1_shift = acc1_round >> shift1                # step 3
    acc1_clamp = np.clip(acc1_shift, -128, 127)       # step 4
    a1 = np.where(acc1_clamp < 0, 0, acc1_clamp)      # step 5

    return acc1.astype(np.int32), a1.astype(np.int8)


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
    a1_wide = a1_i8.astype(np.int64)
    W2_wide = W2_q.astype(np.int64)
    bias2_wide = bias_q_fc2.astype(np.int64)

    acc2 = W2_wide @ a1_wide + bias2_wide

    return acc2.astype(np.int32)


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
    acc1, a1 = fc1_forward_int(x_pixels_u8, W1_q, bias_q_fc1, shift1)
    acc2 = fc2_forward_int(a1, W2_q, bias_q_fc2)
    pred = int(np.argmax(acc2))  # ties -> smallest index, same convention as train_quantize.py

    return {"acc1": acc1, "a1": a1, "acc2": acc2, "pred": pred}


def dump_reference_vectors(data_dir=DATA_DIR):
    """
    For each of the 20 stored test images in data/, run predict() and write
    out acc1[64], a1[64], acc2[10], pred to a format Lab 4's SystemVerilog
    testbench can read with $fscanf (plain decimal integers, one per line or
    whatever layout you document clearly -- match it exactly on the SV side).

    This is what makes Lab 4's testbench comparison bit-exact rather than
    "final digit only" -- the exact gap in the old version of this project.
    """
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    n_input = meta["N_INPUT"]
    ntest = meta["NTEST_STORED"]

    W1_q, bias_q_fc1, W2_q, bias_q_fc2, shift1 = load_quantized_params(data_dir)

    # Pixels are UNSIGNED (0..255) -- unlike weights, no two's-complement
    # correction here, or values 128..255 would wrongly come out negative.
    X = np.array(_read_hex_lines(data_dir / "x_test.mem"), dtype=np.uint8)
    X = X.reshape(ntest, n_input)

    with open(data_dir / "y_label.txt") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    correct = 0
    for img in range(ntest):
        result = predict(X[img], W1_q, bias_q_fc1, W2_q, bias_q_fc2, shift1)

        # Format must match tb_digit_top.sv's load_reference_vectors() task
        # EXACTLY: acc1[0..63], then a1[0..63], then acc2[0..9], then pred --
        # one plain decimal integer per line, no blank lines.
        with open(data_dir / f"ref_img{img}.txt", "w") as f:
            for v in result["acc1"]:
                f.write(f"{int(v)}\n")
            for v in result["a1"]:
                f.write(f"{int(v)}\n")
            for v in result["acc2"]:
                f.write(f"{int(v)}\n")
            f.write(f"{result['pred']}\n")

        if result["pred"] == labels[img]:
            correct += 1

    print(f"Wrote {ntest} reference files to {data_dir}/")
    print(f"Accuracy over the 20 stored images: {correct}/{ntest} ({100 * correct / ntest:.1f}%)")


def main():
    dump_reference_vectors()


if __name__ == "__main__":
    main()
