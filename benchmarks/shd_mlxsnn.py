"""mlx-snn SHD (Spiking Heidelberg Digits) benchmark suite.

Runs on M3 Max with MLX.  Results saved as CSV for comparison
with snnTorch on V100 (shd_snntorch.py).

Models (mirrors shd_snntorch.py):
  M1. FC-SNN 2-layer:  700→256→20, LIF×2            (~185K params)
  M2. FC-SNN 3-layer:  700→256→128→20, LIF×3         (~220K params)
  M3. Recurrent SNN:   700→256→20, recurrent(256→256) (~250K params)
  M4. LSM:             700→(500 reservoir)→20          (~10K trainable)

Usage:
    python benchmarks/shd_mlxsnn.py
"""

import os
import csv
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mlxsnn

# ============================================================
# Config (identical to snnTorch benchmark)
# ============================================================

NUM_STEPS = 100
BATCH_SIZE = 128
NUM_EPOCHS = 30
LR = 1e-3
BETA = 0.9
SEEDS = [42, 123, 7]
SURROGATE = "arctan"
SURROGATE_SCALE = 2.0

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "shd")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_shd_mlx")
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ============================================================
# Data loading
# ============================================================

def load_shd(train=True):
    """Load SHD dataset using mlxsnn SHDDataset, return numpy arrays."""
    ds = mlxsnn.SHDDataset(root=DATA_ROOT, train=train, num_steps=NUM_STEPS)
    log(f"  Loaded SHD {'train' if train else 'test'}: {len(ds)} samples")
    return ds


def make_batches(dataset, batch_size, shuffle=True):
    """Yield (x_batch, y_batch) as mx.array from SHDDataset.

    x_batch: [T, B, 700] (time-first)
    y_batch: [B]
    """
    n = len(dataset)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)

    for i in range(0, n - batch_size + 1, batch_size):
        batch_idx = idx[i:i + batch_size]
        spikes_list = []
        labels_list = []
        for j in batch_idx:
            spk, lbl = dataset[j]
            # spk is mx.array [T, 700], convert to numpy for batching
            spikes_list.append(np.array(spk))
            labels_list.append(lbl)

        # Stack: [B, T, 700] -> transpose to [T, B, 700]
        x_np = np.stack(spikes_list, axis=0)  # [B, T, 700]
        x_np = np.transpose(x_np, (1, 0, 2))  # [T, B, 700]
        y_np = np.array(labels_list, dtype=np.int32)

        yield mx.array(x_np), mx.array(y_np)


# ============================================================
# M1: FC-SNN 2-layer
# ============================================================

class FCSNN2L(nn.Module):
    """700→256→20, LIF×2."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(700, 256)
        self.lif1 = mlxsnn.Leaky(
            beta=BETA, surrogate_fn=SURROGATE, surrogate_scale=SURROGATE_SCALE)
        self.fc2 = nn.Linear(256, 20)
        self.lif2 = mlxsnn.Leaky(
            beta=BETA, surrogate_fn=SURROGATE, surrogate_scale=SURROGATE_SCALE)

    def __call__(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        s1 = self.lif1.init_state(B, 256)
        s2 = self.lif2.init_state(B, 20)
        spk_rec = []

        for t in range(T):
            h = self.fc1(x_seq[t])
            spk, s1 = self.lif1(h, s1)
            h = self.fc2(spk)
            spk, s2 = self.lif2(h, s2)
            spk_rec.append(spk)

        return mx.stack(spk_rec)  # [T, B, 20]


# ============================================================
# M2: FC-SNN 3-layer
# ============================================================

class FCSNN3L(nn.Module):
    """700→256→128→20, LIF×3."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(700, 256)
        self.lif1 = mlxsnn.Leaky(
            beta=BETA, surrogate_fn=SURROGATE, surrogate_scale=SURROGATE_SCALE)
        self.fc2 = nn.Linear(256, 128)
        self.lif2 = mlxsnn.Leaky(
            beta=BETA, surrogate_fn=SURROGATE, surrogate_scale=SURROGATE_SCALE)
        self.fc3 = nn.Linear(128, 20)
        self.lif3 = mlxsnn.Leaky(
            beta=BETA, surrogate_fn=SURROGATE, surrogate_scale=SURROGATE_SCALE)

    def __call__(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        s1 = self.lif1.init_state(B, 256)
        s2 = self.lif2.init_state(B, 128)
        s3 = self.lif3.init_state(B, 20)
        spk_rec = []

        for t in range(T):
            h = self.fc1(x_seq[t])
            spk, s1 = self.lif1(h, s1)
            h = self.fc2(spk)
            spk, s2 = self.lif2(h, s2)
            h = self.fc3(spk)
            spk, s3 = self.lif3(h, s3)
            spk_rec.append(spk)

        return mx.stack(spk_rec)  # [T, B, 20]


# ============================================================
# M3: Recurrent SNN
# ============================================================

class RecurrentSNN(nn.Module):
    """700→256(recurrent)→20.

    Manual all-to-all recurrence: fc_rec(256→256) feeds previous spikes
    back as additional input, matching snnTorch RLeaky(all_to_all=True).
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(700, 256)
        self.fc_rec = nn.Linear(256, 256, bias=False)  # recurrent weight V
        self.lif1 = mlxsnn.Leaky(
            beta=BETA, surrogate_fn=SURROGATE, surrogate_scale=SURROGATE_SCALE)
        self.fc2 = nn.Linear(256, 20)
        self.lif2 = mlxsnn.Leaky(
            beta=BETA, surrogate_fn=SURROGATE, surrogate_scale=SURROGATE_SCALE)

    def __call__(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        s1 = self.lif1.init_state(B, 256)
        s2 = self.lif2.init_state(B, 20)
        prev_spk = mx.zeros((B, 256))
        spk_rec = []

        for t in range(T):
            h = self.fc1(x_seq[t]) + self.fc_rec(prev_spk)
            spk, s1 = self.lif1(h, s1)
            prev_spk = spk
            h = self.fc2(spk)
            spk, s2 = self.lif2(h, s2)
            spk_rec.append(spk)

        return mx.stack(spk_rec)  # [T, B, 20]


# ============================================================
# M4: LSM
# ============================================================

class LSMSNN(nn.Module):
    """Manual reservoir matching snnTorch construction (no Dale's law).

    700→(500 reservoir, frozen)→20 (trainable linear readout).
    """

    def __init__(self, seed=42):
        super().__init__()
        rng = np.random.default_rng(seed)

        # Random input weights (frozen) — matches snnTorch
        W_in = rng.standard_normal((700, 500)).astype(np.float32)
        W_in /= np.sqrt(700)
        self._W_in = mx.array(W_in)

        # Random sparse recurrent weights (frozen) — matches snnTorch
        connectivity = 0.1
        spectral_radius = 0.9
        mask = (rng.random((500, 500)) < connectivity).astype(np.float32)
        np.fill_diagonal(mask, 0)
        W_res = rng.standard_normal((500, 500)).astype(np.float32) * mask
        eigvals = np.abs(np.linalg.eigvals(W_res)).max()
        if eigvals > 0:
            W_res = W_res * (spectral_radius / eigvals)
        self._W_res = mx.array(W_res)

        self.threshold = 1.0
        self.readout = nn.Linear(500, 20)

    def __call__(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        mem = mx.zeros((B, 500))
        spk = mx.zeros((B, 500))
        out_list = []

        for t in range(T):
            I = x_seq[t] @ self._W_in + spk @ self._W_res
            mem = BETA * mem + I
            spk = mx.where(mem >= self.threshold, 1.0, 0.0)
            mem = mem - spk * self.threshold
            out_list.append(self.readout(spk))

        return mx.stack(out_list)  # [T, B, 20]


# ============================================================
# Training loop
# ============================================================

MODELS = [
    ("m1_fc2l", FCSNN2L),
    ("m2_fc3l", FCSNN3L),
    ("m3_recurrent", RecurrentSNN),
    ("m4_lsm", LSMSNN),
]


def train_model(model_name, model_cls, train_ds, test_ds, seeds=SEEDS):
    log(f"\n{'='*60}")
    log(f"Training {model_name}")
    log(f"{'='*60}")

    all_results = []

    for seed in seeds:
        np.random.seed(seed)
        mx.random.seed(seed)

        if model_cls == LSMSNN:
            model = model_cls(seed=seed)
        else:
            model = model_cls()

        # M4 (LSM) uses higher LR since only readout is trainable
        lr = 5e-3 if model_cls == LSMSNN else LR
        optimizer = optim.Adam(learning_rate=lr)

        def loss_fn(model, x_seq, y):
            spk_out = model(x_seq)
            logits = mx.sum(spk_out, axis=0)  # rate decode: spike count over time
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            total_loss, correct, total = 0.0, 0, 0

            for bx, by in make_batches(train_ds, BATCH_SIZE, shuffle=True):
                loss, grads = loss_and_grad(model, bx, by)
                grads, _ = optim.clip_grad_norm(grads, max_norm=1.0)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)

                # Train accuracy from same batch (extra forward, matches snnTorch)
                spk_out = model(bx)
                preds = mx.argmax(mx.sum(spk_out, axis=0), axis=-1)
                mx.eval(preds)

                total_loss += loss.item() * by.shape[0]
                correct += mx.sum(preds == by).item()
                total += by.shape[0]

            elapsed = time.time() - t0
            train_acc = correct / total

            # Test accuracy
            test_correct, test_total = 0, 0
            for bx, by in make_batches(test_ds, BATCH_SIZE, shuffle=False):
                spk_out = model(bx)
                preds = mx.argmax(mx.sum(spk_out, axis=0), axis=-1)
                mx.eval(preds)
                test_correct += mx.sum(preds == by).item()
                test_total += by.shape[0]
            test_acc = test_correct / test_total if test_total > 0 else 0.0

            log(f"  seed={seed} epoch={epoch+1:2d}/{NUM_EPOCHS}  "
                f"loss={total_loss/total:.4f}  train={train_acc:.4f}  "
                f"test={test_acc:.4f}  time={elapsed:.2f}s")

            all_results.append({
                "seed": seed, "epoch": epoch + 1,
                "train_loss": round(total_loss / total, 6),
                "train_acc": round(train_acc, 6),
                "test_acc": round(test_acc, 6),
                "epoch_time": round(elapsed, 3),
            })

    # Save CSV
    fname = f"{model_name}.csv"
    with open(os.path.join(OUT_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    log(f"  Saved {fname}")
    return all_results


# ============================================================
# Throughput benchmark
# ============================================================

def bench_throughput(train_ds):
    log(f"\n{'='*60}")
    log(f"Throughput benchmark (inference only)")
    log(f"{'='*60}")

    results = []

    for model_name, model_cls in MODELS:
        if model_cls == LSMSNN:
            model = model_cls(seed=42)
        else:
            model = model_cls()

        # Create a fixed input batch
        x = mx.random.normal((NUM_STEPS, BATCH_SIZE, 700))
        mx.eval(x)

        # Warmup
        for _ in range(5):
            out = model(x)
            mx.eval(out)

        # Timed
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            out = model(x)
            mx.eval(out)
            times.append(time.perf_counter() - t0)

        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        samples_per_sec = BATCH_SIZE / np.mean(times)

        log(f"  {model_name:15s}  "
            f"{avg_ms:7.2f} +/- {std_ms:.2f} ms  "
            f"{samples_per_sec:8.0f} samples/s")

        results.append({
            "model": model_name,
            "batch_size": BATCH_SIZE,
            "avg_ms": round(avg_ms, 3),
            "std_ms": round(std_ms, 3),
            "samples_per_sec": round(samples_per_sec, 1),
        })

    fname = "throughput.csv"
    with open(os.path.join(OUT_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    log(f"  Saved {fname}")
    return results


# ============================================================
# Main
# ============================================================

def main():
    log(f"mlx-snn SHD Benchmark Suite")
    log(f"mlx-snn {mlxsnn.__version__}")
    log(f"Seeds: {SEEDS}, Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, "
        f"T: {NUM_STEPS}, LR: {LR}, Beta: {BETA}")
    log(f"Surrogate: {SURROGATE} (scale={SURROGATE_SCALE})")
    log(f"Output: {OUT_DIR}/")

    # Load data once
    log(f"\nLoading SHD dataset...")
    train_ds = load_shd(train=True)
    test_ds = load_shd(train=False)

    # Train all models
    for model_name, model_cls in MODELS:
        train_model(model_name, model_cls, train_ds, test_ds)

    # Throughput
    bench_throughput(train_ds)

    log(f"\n{'='*60}")
    log(f"All benchmarks complete. Results in {OUT_DIR}/")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
