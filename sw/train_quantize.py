"""
sw/train_quantize.py

Lab 1 skeleton. See docs/lab1_quantization.md and docs/architecture_spec.md
(Sections 1-3) before writing any code here.

Pipeline this script must implement, in order:
  1. Load real MNIST (28x28), NOT sklearn.datasets.load_digits().
  2. 2x2-average-pool every image down to 14x14 (196 values), still 0..255 range.
  3. Train a float 196 -> 64 (ReLU) -> 10 MLP to convergence.
  4. Quantize weights/biases to the scheme in architecture_spec.md Section 2.
  5. Calibrate SHIFT1 from a calibration subset of training images.
  6. Validate quantized (integer-only) accuracy on the full test set.
  7. Export .mem files + meta.json to data/.

Every TODO below corresponds to one numbered step in docs/lab1_quantization.md's
"Your task" section. Fill in real logic -- these stubs intentionally do nothing.
"""

import json
import os

import numpy as np

OUTDIR = "data"

N_INPUT = 196     # 14x14 pooled MNIST
HIDDEN = 64
N_CLASS = 10
NTEST_STORED = 20  # test images exported for RTL/testbench use (2 per digit class)


def load_real_mnist():
    """
    Step 1. Load the real MNIST dataset (60,000 train / 10,000 test, 28x28,
    uint8 grayscale 0..255). Use torchvision.datasets.MNIST or
    tensorflow.keras.datasets.mnist -- NOT sklearn.datasets.load_digits(), which
    is the wrong (8x8 toy) dataset the old version of this project used.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        X_* shape (N, 28, 28) dtype uint8, y_* shape (N,) dtype int, values 0..9.
    """

    from torchvision.datasets import MNIST

    train_ds = MNIST(root="./mnist_data", train=True, download=True)
    test_ds = MNIST(root="./mnist_data", train=False, download=True)

    print(type(train_ds.data))
    print(type(train_ds.targets))

    X_train = train_ds.data.numpy()
    y_train = train_ds.targets.numpy()
    X_test = test_ds.data.numpy()
    y_test = test_ds.targets.numpy()

    print(X_train.shape, X_train.dtype, X_train.min(), X_train.max())
    print(y_train.shape, y_train.dtype, y_train.min(), y_train.max())


    return X_train, y_train, X_test, y_test


def pool_2x2(images_28x28):
    """
    Step 2. Average-pool a batch of 28x28 images down to 14x14 (196 values each).
    Output should still be in the 0..255 grayscale range (round to nearest int,
    do not renormalize to 0..1 -- pixels stay uint8-range integers all the way
    through this project, per architecture_spec.md Section 2).

    Parameters
    ----------
    images_28x28 : np.ndarray, shape (N, 28, 28)

    Returns
    -------
    np.ndarray, shape (N, 14, 14), same dtype/range as input
    """
    raise NotImplementedError("TODO: 2x2 average pool (Lab 1, step 2)")


def train_float_model(X_train_pooled, y_train, X_test_pooled, y_test):
    """
    Step 3. Train a float32 196 -> 64 (ReLU) -> 10 MLP on the pooled data.
    Expect ~96-97.5% test accuracy on convergence (see architecture_spec.md
    Section 1) -- if you're far below that, don't proceed to quantization until
    you understand why.

    Returns
    -------
    W1_fp32, b1_fp32, W2_fp32, b2_fp32 : np.ndarray
        W1: (64, 196), b1: (64,), W2: (10, 64), b2: (10,)
        (weight matrices oriented as (num_neurons, num_inputs), matching the
        z = W @ x + b convention from docs/lab0_nn_fundamentals.md)
    float_test_accuracy : float
    """
    raise NotImplementedError("TODO: train the float MLP (Lab 1, step 3)")


def quantize_weights(W_fp32):
    """
    Step 4. Symmetric int8 weight quantization for one layer's weight matrix.
    See architecture_spec.md Section 2:
        w_scale = 127 / max(abs(W_fp32))
        w_q     = clamp(round(W_fp32 * w_scale), -127, 127)

    Returns
    -------
    W_q : np.ndarray, dtype int8, same shape as W_fp32
    w_scale : float
    """
    raise NotImplementedError("TODO: symmetric int8 weight quantization (Lab 1, step 4)")


def quantize_bias_fc1(b1_fp32, w_scale_fc1):
    """
    Step 4 (bias, FC1). bias_q_fc1 = round(b1_fp32 * w_scale_fc1)
    (x_scale_input = 1, since raw 0..255 pixels are used directly with no scale.)

    Returns
    -------
    np.ndarray, dtype int32
    """
    raise NotImplementedError("TODO: quantize FC1 bias (Lab 1, step 4)")


def quantize_bias_fc2(b2_fp32, w_scale_fc1, w_scale_fc2, shift1):
    """
    Step 4 (bias, FC2). Depends on SHIFT1 -- unlike FC1's bias, so this must be
    called AFTER calibrate_shift1(). See architecture_spec.md Section 2:
        x_scale_hidden = w_scale_fc1 / 2^SHIFT1
        bias_q_fc2 = round(b2_fp32 * w_scale_fc2 * x_scale_hidden)

    Returns
    -------
    np.ndarray, dtype int32
    """
    raise NotImplementedError("TODO: quantize FC2 bias (Lab 1, step 4)")


def calibrate_shift1(W1_q, bias_q_fc1, X_calibration_images):
    """
    Step 5. Calibrate the FC1 requantization shift from real data.

    For every image in X_calibration_images (suggest ~1000 training images) and
    every one of the 64 FC1 neurons, compute the raw int32 accumulator:
        acc1 = sum(W1_q[neuron, i] * pixel[i] for i in 0..195) + bias_q_fc1[neuron]
    (no shift/clamp/ReLU yet -- this is the RAW accumulator, per architecture_spec.md
    Section 2's SHIFT1 formula.)

    Then:
        SHIFT1 = ceil(log2(P99.9(abs(acc1)) / 127))
    where P99.9 is the 99.9th percentile of |acc1| over ALL (image, neuron) pairs
    collected above.

    Returns
    -------
    int
    """
    raise NotImplementedError("TODO: calibrate SHIFT1 from real data (Lab 1, step 5)")


def quantized_inference_one(x_pixels_u8, W1_q, bias_q_fc1, W2_q, bias_q_fc2, shift1):
    """
    Step 6. Integer-only forward pass for ONE image, implementing the exact
    5-step FC1 pipeline from architecture_spec.md Section 2, followed by
    raw-accumulator FC2 + argmax (no requantization on FC2). This function's
    logic must later be duplicated bit-for-bit in golden_model.py -- consider
    implementing it once, carefully, and importing it from there instead of
    copy-pasting, once both files exist.

    No floating point anywhere in this function's body.

    Parameters
    ----------
    x_pixels_u8 : np.ndarray, shape (196,), dtype uint8

    Returns
    -------
    pred : int, the argmax digit prediction (0-9)
    """
    raise NotImplementedError("TODO: integer-only forward pass (Lab 1, step 6)")


def validate_quantized_accuracy(X_test_pooled, y_test, W1_q, bias_q_fc1, W2_q, bias_q_fc2, shift1):
    """
    Step 6. Run quantized_inference_one() over the full test set and report
    accuracy. Target: >= 95% (see architecture_spec.md Section 1). If far below
    that, debug the quantization math before moving on -- do not lower the target.

    Returns
    -------
    float
    """
    raise NotImplementedError("TODO: validate quantized accuracy (Lab 1, step 6)")


def export_mem_files(W1_q, bias_q_fc1, W2_q, bias_q_fc2, shift1,
                      X_test_pooled_u8, y_test, float_acc, quant_acc):
    """
    Step 7. Write to OUTDIR ("data/"):
      - w1.mem, b1.mem, w2.mem, b2.mem  (hex, one value per line, format your
        future $readmemh-based RTL testbench can read -- decide int8 vs int32
        hex width per array, matching architecture_spec.md Section 3's memory
        layout)
      - 20 stored test images (2 per digit class -- pick indices so this is a
        true, easy-to-defend sanity subset, not an arbitrary first-20)
      - their true labels
      - meta.json: topology (N_INPUT, HIDDEN, N_CLASS, NTEST_STORED), and the
        calibrated SHIFT1 (NOT a guessed value)

    No return value; writes files.
    """
    raise NotImplementedError("TODO: export .mem files + meta.json (Lab 1, step 7)")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # TODO: wire together load_real_mnist -> pool_2x2 -> train_float_model ->
    # quantize_weights (both layers) -> calibrate_shift1 -> quantize_bias_fc1 /
    # quantize_bias_fc2 -> validate_quantized_accuracy -> export_mem_files.
    # Print both float_acc and quant_acc clearly at the end -- you'll want these
    # numbers for your README (Lab 7) and for sanity-checking against Lab 4's
    # RTL simulation results.
    raise NotImplementedError("TODO: wire together the full pipeline (Lab 1)")


if __name__ == "__main__":
    main()
