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
import torch
import torch.nn as nn

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

    X_train = train_ds.data.numpy()
    y_train = train_ds.targets.numpy()
    X_test = test_ds.data.numpy()
    y_test = test_ds.targets.numpy()

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
    # Promote to a wider dtype before summing -- 4 uint8 pixels can sum to
    # up to 1020, which overflows uint8 (max 255) and would silently wrap.
    x = images_28x28.astype(np.float32)

    n = x.shape[0]
    # Reshape (N, 28, 28) -> (N, 14, 2, 14, 2): splits each 28-length axis
    # into (14 blocks, 2 pixels-per-block), in order, so block b's two
    # pixels are original indices 2b and 2b+1 -- exactly the non-overlapping
    # 2x2 grouping described in docs/lab3... err, lab1_quantization.md.
    x = x.reshape(n, 14, 2, 14, 2)

    # Average over the two "within-block" axes (2 and 4), collapsing each
    # 2x2 block to its single averaged value. Result shape: (N, 14, 14).
    pooled = x.mean(axis=(2, 4))

    # Round to nearest int (not truncate), then cast back down to uint8 now
    # that the value is safely back in the 0..255 range.
    pooled = np.round(pooled).astype(np.uint8)

    return pooled


class _MLP(nn.Module):
    """196 -> 64 (ReLU) -> 10, matching docs/architecture_spec.md Section 1.

    nn.Linear(in_features, out_features) stores its weight as shape
    (out_features, in_features) -- which already matches the project's
    z = W @ x + b convention (W shaped (num_neurons, num_inputs)) with no
    transpose needed when we pull the weights out later.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_INPUT, HIDDEN)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN, N_CLASS)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


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
    torch.manual_seed(0)

    # Flatten (N, 14, 14) -> (N, 196) and use RAW 0..255 float values --
    # deliberately NOT normalized to 0..1. See the explanation above this
    # function's call site: the quantization scheme fixes x_scale_input = 1,
    # so the float model must already be trained at the same input scale
    # the quantized/RTL model will see at inference, or the bias terms end
    # up calibrated to the wrong scale.
    X_train_t = torch.tensor(X_train_pooled.reshape(-1, N_INPUT), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test_pooled.reshape(-1, N_INPUT), dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = _MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()  # expects raw logits + integer class labels

    n_epochs = 15
    batch_size = 128
    n_samples = X_train_t.shape[0]

    test_acc = 0.0
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]

        train_loss = total_loss / n_samples
        with torch.no_grad():
            test_pred = model(X_test_t).argmax(dim=1)
            test_acc = (test_pred == y_test_t).float().mean().item()
        print(f"epoch {epoch + 1}/{n_epochs}  train_loss={train_loss:.4f}  test_acc={test_acc * 100:.2f}%")

    W1_fp32 = model.fc1.weight.detach().numpy()  # (64, 196)
    b1_fp32 = model.fc1.bias.detach().numpy()    # (64,)
    W2_fp32 = model.fc2.weight.detach().numpy()  # (10, 64)
    b2_fp32 = model.fc2.bias.detach().numpy()    # (10,)

    return W1_fp32, b1_fp32, W2_fp32, b2_fp32, test_acc


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
    w_scale = 127.0 / np.max(np.abs(W_fp32))
    w_q = np.clip(np.round(W_fp32 * w_scale), -127, 127).astype(np.int8)
    return w_q, w_scale


def quantize_bias_fc1(b1_fp32, w_scale_fc1):
    """
    Step 4 (bias, FC1). bias_q_fc1 = round(b1_fp32 * w_scale_fc1)
    (x_scale_input = 1, since raw 0..255 pixels are used directly with no scale.)

    Returns
    -------
    np.ndarray, dtype int32
    """
    return np.round(b1_fp32 * w_scale_fc1).astype(np.int32)


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
    x_scale_hidden = w_scale_fc1 / (2 ** shift1)
    return np.round(b2_fp32 * w_scale_fc2 * x_scale_hidden).astype(np.int32)


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
    # Wide dtype before the matmul -- multiplying two int8 arrays directly
    # can silently overflow, same overflow risk as pool_2x2's uint8 sum.
    W1_wide = W1_q.astype(np.int64)                  # (64, 196)
    X_wide = X_calibration_images.astype(np.int64)   # (M, 196)
    bias_wide = bias_q_fc1.astype(np.int64)          # (64,)

    # acc1[m, n] = sum_i(W1_q[n, i] * X[m, i]) + bias_q_fc1[n], for every
    # calibration image m and every neuron n, computed all at once:
    # (M, 196) @ (196, 64) -> (M, 64), then the bias broadcasts over the
    # neuron axis. This is the RAW accumulator -- no shift/clamp/ReLU yet.
    acc1 = X_wide @ W1_wide.T + bias_wide             # (M, 64)

    p999 = np.percentile(np.abs(acc1), 99.9)
    shift1 = int(np.ceil(np.log2(p999 / 127)))
    shift1 = max(shift1, 0)  # a zero/negative shift would be degenerate -- guard against a bad calibration set

    return shift1


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
    # Widen dtypes before any arithmetic -- same overflow reasoning as
    # calibrate_shift1 and pool_2x2.
    x_wide = x_pixels_u8.astype(np.int64)          # (196,)
    W1_wide = W1_q.astype(np.int64)                 # (64, 196)
    bias1_wide = bias_q_fc1.astype(np.int64)        # (64,)

    # ---- FC1: the exact 5 steps, for all 64 neurons at once ----
    acc1 = W1_wide @ x_wide + bias1_wide             # step 1, (64,), int32-range values held in int64

    acc1_round = acc1 + (1 << (shift1 - 1))          # step 2: round-half-up prep
    acc1_shift = acc1_round >> shift1                # step 3: arithmetic shift (numpy's >> on signed ints), NOT real division
    acc1_clamp = np.clip(acc1_shift, -128, 127)       # step 4: saturate to int8 range
    a1 = np.where(acc1_clamp < 0, 0, acc1_clamp)      # step 5: ReLU
    a1_i8 = a1.astype(np.int8)                        # (64,) -- FC1's final output

    # ---- FC2: raw accumulate, NO requantization, straight to argmax ----
    W2_wide = W2_q.astype(np.int64)                  # (10, 64)
    bias2_wide = bias_q_fc2.astype(np.int64)          # (10,)
    a1_wide = a1_i8.astype(np.int64)

    acc2 = W2_wide @ a1_wide + bias2_wide              # (10,)

    pred = int(np.argmax(acc2))  # ties -> smallest index, matching Python's max() convention
    return pred


def validate_quantized_accuracy(X_test_pooled, y_test, W1_q, bias_q_fc1, W2_q, bias_q_fc2, shift1):
    """
    Step 6. Run quantized_inference_one() over the full test set and report
    accuracy. Target: >= 95% (see architecture_spec.md Section 1). If far below
    that, debug the quantization math before moving on -- do not lower the target.

    Returns
    -------
    float
    """
    X_flat = X_test_pooled.reshape(-1, N_INPUT)
    n = X_flat.shape[0]

    correct = 0
    for i in range(n):
        pred = quantized_inference_one(X_flat[i], W1_q, bias_q_fc1, W2_q, bias_q_fc2, shift1)
        if pred == y_test[i]:
            correct += 1

    return correct / n


def _to_hex_u8(v):
    """Two's-complement hex byte, e.g. -1 -> 'ff', 127 -> '7f'."""
    return f"{(int(v) & 0xFF):02x}"


def _to_hex_u32(v):
    """Two's-complement 8-hex-digit word, e.g. -1 -> 'ffffffff'."""
    return f"{(int(v) & 0xFFFFFFFF):08x}"


def _write_mem_u8(path, arr):
    with open(path, "w") as f:
        for v in arr.flatten():
            f.write(_to_hex_u8(v) + "\n")


def _write_mem_u32(path, arr):
    with open(path, "w") as f:
        for v in arr.flatten():
            f.write(_to_hex_u32(v) + "\n")


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
    _write_mem_u8(os.path.join(OUTDIR, "w1.mem"), W1_q)
    _write_mem_u32(os.path.join(OUTDIR, "b1.mem"), bias_q_fc1)
    _write_mem_u8(os.path.join(OUTDIR, "w2.mem"), W2_q)
    _write_mem_u32(os.path.join(OUTDIR, "b2.mem"), bias_q_fc2)

    # Pick 2 test images per digit class (0-9) = 20 total -- deterministic
    # (first 2 occurrences of each label), not an arbitrary first-20 slice.
    X_flat = X_test_pooled_u8.reshape(-1, N_INPUT)
    selected_idx = []
    for digit in range(N_CLASS):
        idx_for_digit = np.where(y_test == digit)[0][:2]
        selected_idx.extend(idx_for_digit.tolist())

    X_selected = X_flat[selected_idx]  # (20, 196)
    y_selected = y_test[selected_idx]  # (20,)

    _write_mem_u8(os.path.join(OUTDIR, "x_test.mem"), X_selected)

    with open(os.path.join(OUTDIR, "y_label.txt"), "w") as f:
        for v in y_selected:
            f.write(str(int(v)) + "\n")

    meta = {
        "N_INPUT": N_INPUT,
        "HIDDEN": HIDDEN,
        "N_CLASS": N_CLASS,
        "NTEST_STORED": NTEST_STORED,
        "SHIFT1": shift1,
        "float_test_accuracy": float_acc,
        "quantized_test_accuracy": quant_acc,
    }
    with open(os.path.join(OUTDIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    # TODO: wire together load_real_mnist -> pool_2x2 -> train_float_model ->
    # quantize_weights (both layers) -> calibrate_shift1 -> quantize_bias_fc1 /
    # quantize_bias_fc2 -> validate_quantized_accuracy -> export_mem_files.
    # Print both float_acc and quant_acc clearly at the end -- you'll want these
    # numbers for your README (Lab 7) and for sanity-checking against Lab 4's
    # RTL simulation results.
    X_train, y_train, X_test, y_test = load_real_mnist()
    X_train_p = pool_2x2(X_train)
    X_test_p = pool_2x2(X_test)

    W1, b1, W2, b2, float_acc = train_float_model(X_train_p, y_train, X_test_p, y_test)

    W1_q, w_scale_fc1 = quantize_weights(W1)
    W2_q, w_scale_fc2 = quantize_weights(W2)
    b1_q = quantize_bias_fc1(b1, w_scale_fc1)

    calib_idx = np.random.choice(X_train_p.shape[0], size=1000, replace=False)
    X_calib = X_train_p.reshape(-1, N_INPUT)[calib_idx]
    shift1 = calibrate_shift1(W1_q, b1_q, X_calib)

    b2_q = quantize_bias_fc2(b2, w_scale_fc1, w_scale_fc2, shift1)

    quant_acc = validate_quantized_accuracy(X_test_p, y_test, W1_q, b1_q, W2_q, b2_q, shift1)

    print(f"Float test accuracy:      {float_acc * 100:.2f}%")
    print(f"Quantized test accuracy:  {quant_acc * 100:.2f}%")
    print(f"Calibrated SHIFT1:        {shift1}")

    export_mem_files(W1_q, b1_q, W2_q, b2_q, shift1, X_test_p, y_test, float_acc, quant_acc)
    print(f"Exported .mem files + meta.json to {OUTDIR}/")


if __name__ == "__main__":
    main()
