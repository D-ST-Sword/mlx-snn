"""mlx-snn v0.7 cross-platform benchmark suite.

Runs on M3 Max with MLX.  Results saved as CSV for comparison
with snnTorch on V100.

Benchmarks (mirrors snntorch_v07_bench.py):
  B1. FC SNN on MNIST (784-128-10, 25 timesteps, 5 seeds)
  B2. FC SNN on FashionMNIST
  B3. Conv SNN on FashionMNIST (2 conv blocks + FC)
  B4. Forward throughput (batch sweep, no training)
  B5. LSM reservoir + linear readout on MNIST
  B6. mx.compile speedup test

Usage:
    python mlxsnn_v07_bench.py
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

NUM_STEPS = 25
BATCH_SIZE = 128
NUM_EPOCHS = 10
LR = 1e-3
SEEDS = [42, 123, 7, 2024, 99]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v07_mlx")
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ============================================================
# Data (using torchvision via numpy, no torch dependency at runtime)
# ============================================================

def load_dataset(name="MNIST"):
    """Load MNIST/FashionMNIST using mlx-compatible numpy arrays."""
    try:
        from torchvision import datasets, transforms
        import torch
    except ImportError:
        raise ImportError("torchvision required for benchmark data loading")

    transform = transforms.Compose([transforms.ToTensor()])
    cls = datasets.MNIST if name == "MNIST" else datasets.FashionMNIST
    train_ds = cls("./data", train=True, download=True, transform=transform)
    test_ds = cls("./data", train=False, download=True, transform=transform)

    # Convert to numpy
    train_x = train_ds.data.numpy().astype(np.float32) / 255.0
    train_y = train_ds.targets.numpy().astype(np.int32)
    test_x = test_ds.data.numpy().astype(np.float32) / 255.0
    test_y = test_ds.targets.numpy().astype(np.int32)

    return (train_x, train_y), (test_x, test_y)


def make_batches(x, y, batch_size, shuffle=True):
    """Yield (x_batch, y_batch) numpy arrays."""
    n = len(y)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, n, batch_size):
        batch_idx = idx[i:i + batch_size]
        yield x[batch_idx], y[batch_idx]


# ============================================================
# B1/B2: FC SNN
# ============================================================

class FCSNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = mlxsnn.Leaky(beta=beta)
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = mlxsnn.Leaky(beta=beta)

    def __call__(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        s1 = self.lif1.init_state(B, 128)
        s2 = self.lif2.init_state(B, 10)
        spk_rec = []

        for t in range(T):
            h = self.fc1(x_seq[t].reshape(B, -1))
            spk, s1 = self.lif1(h, s1)
            h = self.fc2(spk)
            spk, s2 = self.lif2(h, s2)
            spk_rec.append(spk)

        return mx.stack(spk_rec)  # [T, B, 10]


def train_fc_snn(dataset_name, seeds=SEEDS):
    log(f"\n{'='*60}")
    log(f"B{'1' if dataset_name=='MNIST' else '2'}: FC SNN on {dataset_name}")
    log(f"{'='*60}")

    (train_x, train_y), (test_x, test_y) = load_dataset(dataset_name)
    all_results = []

    for seed in seeds:
        np.random.seed(seed)
        mx.random.seed(seed)

        model = FCSNN()
        optimizer = optim.Adam(learning_rate=LR)

        def loss_fn(model, x_seq, y):
            spk_out = model(x_seq)
            logits = mx.sum(spk_out, axis=0)  # rate decode
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            total_loss, correct, total = 0.0, 0, 0

            for bx, by in make_batches(train_x, train_y, BATCH_SIZE):
                bx_mx = mx.array(bx)
                by_mx = mx.array(by)
                # Rate encode
                x_seq = (mx.random.uniform(shape=(NUM_STEPS, *bx_mx.shape)) < mx.expand_dims(bx_mx, 0)).astype(mx.float32)

                loss, grads = loss_and_grad(model, x_seq, by_mx)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)

                total_loss += loss.item() * by.shape[0]
                spk_out = model(x_seq)
                preds = mx.argmax(mx.sum(spk_out, axis=0), axis=-1)
                correct += mx.sum(preds == by_mx).item()
                total += by.shape[0]

            elapsed = time.time() - t0
            train_acc = correct / total

            # Test
            test_correct, test_total = 0, 0
            for bx, by in make_batches(test_x, test_y, BATCH_SIZE, shuffle=False):
                bx_mx = mx.array(bx)
                by_mx = mx.array(by)
                x_seq = (mx.random.uniform(shape=(NUM_STEPS, *bx_mx.shape)) < mx.expand_dims(bx_mx, 0)).astype(mx.float32)
                spk_out = model(x_seq)
                mx.eval(spk_out)
                preds = mx.argmax(mx.sum(spk_out, axis=0), axis=-1)
                test_correct += mx.sum(preds == by_mx).item()
                test_total += by.shape[0]
            test_acc = test_correct / test_total

            log(f"  seed={seed} epoch={epoch+1:2d}/{NUM_EPOCHS}  "
                f"loss={total_loss/total:.4f}  train={train_acc:.4f}  "
                f"test={test_acc:.4f}  time={elapsed:.2f}s")

            all_results.append({
                "seed": seed, "epoch": epoch + 1,
                "train_loss": total_loss / total,
                "train_acc": train_acc, "test_acc": test_acc,
                "epoch_time": elapsed,
            })

    fname = f"b{'1' if dataset_name=='MNIST' else '2'}_fc_{dataset_name.lower()}.csv"
    with open(os.path.join(OUT_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    log(f"  Saved {fname}")
    return all_results


# ============================================================
# B3: Conv SNN on FashionMNIST
# ============================================================

class ConvSNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm(32)
        self.lif1 = mlxsnn.Leaky(beta=beta)
        self.pool1 = mlxsnn.SpikingMaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm(64)
        self.lif2 = mlxsnn.Leaky(beta=beta)
        self.pool2 = mlxsnn.SpikingMaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.lif_out = mlxsnn.Leaky(beta=beta)

    def __call__(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        s1 = self.lif1.init_state(B, 32 * 14 * 14)
        s2 = self.lif2.init_state(B, 64 * 7 * 7)
        s_out = self.lif_out.init_state(B, 10)
        spk_rec = []

        for t in range(T):
            # MLX uses NHWC: (B, 28, 28, 1)
            h = self.conv1(x_seq[t])
            h = self.bn1(h)
            h_flat = h.reshape(B, -1)
            spk, s1 = self.lif1(h_flat, s1)
            h = spk.reshape(B, 14, 14, 32)  # after pool equiv
            h = self.pool1(self.conv1(x_seq[t]))  # redo with pool

            # Simpler: manual conv+bn+lif+pool
            h = self.bn1(self.conv1(x_seq[t]))
            h = h.reshape(B, -1)
            spk1, s1 = self.lif1(h, s1)
            spk1_spatial = spk1.reshape(B, 14, 14, 32)
            h = self.pool1(spk1_spatial)

            h = self.bn2(self.conv2(h))
            h = h.reshape(B, -1)
            spk2, s2 = self.lif2(h, s2)
            spk2_spatial = spk2.reshape(B, 7, 7, 64)
            h = self.pool2(spk2_spatial)

            h = h.reshape(B, -1)
            spk_out, s_out = self.lif_out(self.fc(h), s_out)
            spk_rec.append(spk_out)

        return mx.stack(spk_rec)


class ConvSNNSimple(nn.Module):
    """Simplified Conv SNN using flattened LIF after conv."""
    def __init__(self, beta=0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.lif1 = mlxsnn.Leaky(beta=beta)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.lif2 = mlxsnn.Leaky(beta=beta)

        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.lif_out = mlxsnn.Leaky(beta=beta)

    def __call__(self, x_seq):
        """x_seq: [T, B, 28, 28, 1] (NHWC)"""
        T, B = x_seq.shape[0], x_seq.shape[1]
        s1 = self.lif1.init_state(B, 32 * 14 * 14)
        s2 = self.lif2.init_state(B, 64 * 7 * 7)
        s_out = self.lif_out.init_state(B, 10)
        spk_rec = []

        for t in range(T):
            # Conv1 + MaxPool (via reshape stride trick)
            h = self.conv1(x_seq[t])  # (B,28,28,32)
            # Simple max pool 2x2
            h = h.reshape(B, 14, 2, 14, 2, 32).max(axis=(2, 4))  # (B,14,14,32)
            h = h.reshape(B, -1)
            spk, s1 = self.lif1(h, s1)

            # Conv2 + MaxPool
            h = spk.reshape(B, 14, 14, 32)
            h = self.conv2(h)  # (B,14,14,64)
            h = h.reshape(B, 7, 2, 7, 2, 64).max(axis=(2, 4))  # (B,7,7,64)
            h = h.reshape(B, -1)
            spk, s2 = self.lif2(h, s2)

            # FC readout
            spk_out, s_out = self.lif_out(self.fc(spk), s_out)
            spk_rec.append(spk_out)

        return mx.stack(spk_rec)


def train_conv_snn(seeds=SEEDS):
    log(f"\n{'='*60}")
    log(f"B3: Conv SNN on FashionMNIST")
    log(f"{'='*60}")

    (train_x, train_y), (test_x, test_y) = load_dataset("FashionMNIST")
    all_results = []

    for seed in seeds:
        np.random.seed(seed)
        mx.random.seed(seed)

        model = ConvSNNSimple()
        optimizer = optim.Adam(learning_rate=LR)

        def loss_fn(model, x_seq, y):
            spk_out = model(x_seq)
            logits = mx.sum(spk_out, axis=0)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            total_loss, correct, total = 0.0, 0, 0

            for bx, by in make_batches(train_x, train_y, BATCH_SIZE):
                # Reshape to NHWC: (B, 28, 28, 1)
                bx_mx = mx.array(bx).reshape(-1, 28, 28, 1)
                by_mx = mx.array(by)
                x_seq = (mx.random.uniform(shape=(NUM_STEPS, *bx_mx.shape)) < mx.expand_dims(bx_mx, 0)).astype(mx.float32)

                loss, grads = loss_and_grad(model, x_seq, by_mx)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)

                total_loss += loss.item() * by.shape[0]
                spk_out = model(x_seq)
                preds = mx.argmax(mx.sum(spk_out, axis=0), axis=-1)
                correct += mx.sum(preds == by_mx).item()
                total += by.shape[0]

            elapsed = time.time() - t0
            train_acc = correct / total

            test_correct, test_total = 0, 0
            for bx, by in make_batches(test_x, test_y, BATCH_SIZE, shuffle=False):
                bx_mx = mx.array(bx).reshape(-1, 28, 28, 1)
                by_mx = mx.array(by)
                x_seq = (mx.random.uniform(shape=(NUM_STEPS, *bx_mx.shape)) < mx.expand_dims(bx_mx, 0)).astype(mx.float32)
                spk_out = model(x_seq)
                mx.eval(spk_out)
                preds = mx.argmax(mx.sum(spk_out, axis=0), axis=-1)
                test_correct += mx.sum(preds == by_mx).item()
                test_total += by.shape[0]
            test_acc = test_correct / test_total

            log(f"  seed={seed} epoch={epoch+1:2d}/{NUM_EPOCHS}  "
                f"loss={total_loss/total:.4f}  train={train_acc:.4f}  "
                f"test={test_acc:.4f}  time={elapsed:.2f}s")

            all_results.append({
                "seed": seed, "epoch": epoch + 1,
                "train_loss": total_loss / total,
                "train_acc": train_acc, "test_acc": test_acc,
                "epoch_time": elapsed,
            })

    fname = "b3_conv_fashionmnist.csv"
    with open(os.path.join(OUT_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    log(f"  Saved {fname}")
    return all_results


# ============================================================
# B4: Forward throughput
# ============================================================

def bench_throughput():
    log(f"\n{'='*60}")
    log(f"B4: Forward throughput (inference only)")
    log(f"{'='*60}")

    results = []
    T = 25

    for model_name, model_cls, input_fn in [
        ("FC-SNN", FCSNN, lambda B: mx.random.uniform(shape=(T, B, 28, 28))),
        ("Conv-SNN", ConvSNNSimple, lambda B: mx.random.uniform(shape=(T, B, 28, 28, 1))),
    ]:
        for batch_size in [32, 64, 128, 256, 512]:
            model = model_cls()
            x = input_fn(batch_size)
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
            samples_per_sec = batch_size / np.mean(times)

            log(f"  {model_name:10s}  B={batch_size:4d}  "
                f"{avg_ms:7.2f} ± {std_ms:.2f} ms  "
                f"{samples_per_sec:8.0f} samples/s")

            results.append({
                "model": model_name, "batch_size": batch_size,
                "avg_ms": avg_ms, "std_ms": std_ms,
                "samples_per_sec": samples_per_sec,
            })

    fname = "b4_throughput.csv"
    with open(os.path.join(OUT_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    log(f"  Saved {fname}")
    return results


# ============================================================
# B5: LSM reservoir + linear readout
# ============================================================

def train_reservoir(seeds=SEEDS):
    log(f"\n{'='*60}")
    log(f"B5: LSM Reservoir on MNIST")
    log(f"{'='*60}")

    (train_x, train_y), (test_x, test_y) = load_dataset("MNIST")
    all_results = []

    for seed in seeds:
        np.random.seed(seed)
        mx.random.seed(seed)

        lsm = mlxsnn.LSM(
            input_size=784, reservoir_size=500, output_size=10,
            readout="linear", seed=seed,
        )
        optimizer = optim.Adam(learning_rate=LR)

        def loss_fn(model, x_seq, y):
            out, _ = model.forward_sequence(x_seq)
            logits = mx.sum(out, axis=0)
            return mx.mean(nn.losses.cross_entropy(logits, y))

        loss_and_grad = nn.value_and_grad(lsm, loss_fn)

        for epoch in range(NUM_EPOCHS):
            t0 = time.time()
            total_loss, correct, total = 0.0, 0, 0

            for bx, by in make_batches(train_x, train_y, BATCH_SIZE):
                bx_mx = mx.array(bx.reshape(-1, 784))
                by_mx = mx.array(by)
                x_seq = (mx.random.uniform(shape=(NUM_STEPS, *bx_mx.shape)) < mx.expand_dims(bx_mx, 0)).astype(mx.float32)

                loss, grads = loss_and_grad(lsm, x_seq, by_mx)
                optimizer.update(lsm, grads)
                mx.eval(lsm.parameters(), optimizer.state, loss)

                total_loss += loss.item() * by.shape[0]
                out, _ = lsm.forward_sequence(x_seq)
                preds = mx.argmax(mx.sum(out, axis=0), axis=-1)
                correct += mx.sum(preds == by_mx).item()
                total += by.shape[0]

            elapsed = time.time() - t0
            train_acc = correct / total

            test_correct, test_total = 0, 0
            for bx, by in make_batches(test_x, test_y, BATCH_SIZE, shuffle=False):
                bx_mx = mx.array(bx.reshape(-1, 784))
                by_mx = mx.array(by)
                x_seq = (mx.random.uniform(shape=(NUM_STEPS, *bx_mx.shape)) < mx.expand_dims(bx_mx, 0)).astype(mx.float32)
                out, _ = lsm.forward_sequence(x_seq)
                mx.eval(out)
                preds = mx.argmax(mx.sum(out, axis=0), axis=-1)
                test_correct += mx.sum(preds == by_mx).item()
                test_total += by.shape[0]
            test_acc = test_correct / test_total

            log(f"  seed={seed} epoch={epoch+1:2d}/{NUM_EPOCHS}  "
                f"loss={total_loss/total:.4f}  train={train_acc:.4f}  "
                f"test={test_acc:.4f}  time={elapsed:.2f}s")

            all_results.append({
                "seed": seed, "epoch": epoch + 1,
                "train_loss": total_loss / total,
                "train_acc": train_acc, "test_acc": test_acc,
                "epoch_time": elapsed,
            })

    fname = "b5_reservoir_mnist.csv"
    with open(os.path.join(OUT_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    log(f"  Saved {fname}")
    return all_results


# ============================================================
# B6: mx.compile speedup
# ============================================================

def bench_compile():
    log(f"\n{'='*60}")
    log(f"B6: mx.compile speedup")
    log(f"{'='*60}")

    results = []
    T = 25
    B = 128

    lif = mlxsnn.Leaky(beta=0.9)
    state = lif.init_state(B, 128)
    x_seq = mx.random.normal((T, B, 128))
    mx.eval(x_seq)

    # Uncompiled
    for _ in range(5):
        s = lif.init_state(B, 128)
        for t in range(T):
            _, s = lif(x_seq[t], s)
        mx.eval(s["mem"])

    times_uncompiled = []
    for _ in range(20):
        s = lif.init_state(B, 128)
        t0 = time.perf_counter()
        for t in range(T):
            _, s = lif(x_seq[t], s)
        mx.eval(s["mem"])
        times_uncompiled.append(time.perf_counter() - t0)

    # Compiled
    step_fn = mlxsnn.compiled_step(lif)
    for _ in range(5):
        s = lif.init_state(B, 128)
        for t in range(T):
            _, s = step_fn(x_seq[t], s)
        mx.eval(s["mem"])

    times_compiled = []
    for _ in range(20):
        s = lif.init_state(B, 128)
        t0 = time.perf_counter()
        for t in range(T):
            _, s = step_fn(x_seq[t], s)
        mx.eval(s["mem"])
        times_compiled.append(time.perf_counter() - t0)

    avg_unc = np.mean(times_uncompiled) * 1000
    avg_comp = np.mean(times_compiled) * 1000
    speedup = avg_unc / avg_comp if avg_comp > 0 else 0

    log(f"  Uncompiled: {avg_unc:.2f} ms")
    log(f"  Compiled:   {avg_comp:.2f} ms")
    log(f"  Speedup:    {speedup:.2f}x")

    results.append({
        "mode": "uncompiled", "avg_ms": avg_unc,
        "std_ms": np.std(times_uncompiled) * 1000,
    })
    results.append({
        "mode": "compiled", "avg_ms": avg_comp,
        "std_ms": np.std(times_compiled) * 1000,
    })

    fname = "b6_compile_speedup.csv"
    with open(os.path.join(OUT_DIR, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    log(f"  Saved {fname}")


# ============================================================
# Main
# ============================================================

def main():
    log(f"mlx-snn v0.7 Benchmark Suite")
    log(f"mlx-snn {mlxsnn.__version__}")
    log(f"Seeds: {SEEDS}, Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, T: {NUM_STEPS}")

    train_fc_snn("MNIST")
    train_fc_snn("FashionMNIST")
    train_conv_snn()
    bench_throughput()
    train_reservoir()
    bench_compile()

    log(f"\n{'='*60}")
    log(f"All benchmarks complete. Results in {OUT_DIR}/")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
