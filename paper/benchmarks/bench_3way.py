"""3-way benchmark: mlx-snn (MLX GPU) vs snnTorch (MPS GPU) vs snnTorch (CPU).

All on the same M3 Max, same architecture, same hyperparameters.
Records: accuracy, wall-clock time per epoch, total time, peak memory.
"""

import sys
sys.path.insert(0, ".")

import json
import time
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_mnist():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        x = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int32)
        return x[:60000], y[:60000], x[60000:], y[60000:]
    except ImportError:
        pass
    import gzip, os, urllib.request
    url_base = "http://yann.lecun.com/exdb/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    data_dir = os.path.join(os.path.dirname(__file__), ".mnist_data")
    os.makedirs(data_dir, exist_ok=True)
    def dl_img(fname):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url_base + fname, path)
        with gzip.open(path, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
    def dl_lbl(fname):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url_base + fname, path)
        with gzip.open(path, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int32)
    return dl_img(files["train_images"]), dl_lbl(files["train_labels"]), \
           dl_img(files["test_images"]), dl_lbl(files["test_labels"])


def get_batches_np(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield x[idx], y[idx]


# ---------------------------------------------------------------------------
# Shared hyperparameters
# ---------------------------------------------------------------------------
NUM_EPOCHS = 15
BATCH_SIZE = 128
NUM_STEPS = 25
LR = 2e-3
BETA = 0.95
HIDDEN = 128


# ---------------------------------------------------------------------------
# 1. mlx-snn (MLX GPU)
# ---------------------------------------------------------------------------

def run_mlxsnn(x_train, y_train, x_test, y_test):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlxsnn

    print("\n" + "=" * 60)
    print("1. mlx-snn (MLX GPU on M3 Max)")
    print("=" * 60)

    class SpikingMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_steps = NUM_STEPS
            self.fc1 = nn.Linear(784, HIDDEN)
            self.lif1 = mlxsnn.Leaky(beta=BETA, threshold=1.0)
            self.fc2 = nn.Linear(HIDDEN, 10)
            self.lif2 = mlxsnn.Leaky(beta=BETA, threshold=1.0,
                                      reset_mechanism="none")

        def __call__(self, x):
            bs = x.shape[1]
            s1 = self.lif1.init_state(bs, HIDDEN)
            s2 = self.lif2.init_state(bs, 10)
            for t in range(self.num_steps):
                spk1, s1 = self.lif1(self.fc1(x[t]), s1)
                spk2, s2 = self.lif2(self.fc2(spk1), s2)
            return s2["mem"]

    # Reset peak memory counter
    try:
        mx.reset_peak_memory()
    except Exception:
        pass

    model = SpikingMLP()
    optimizer = optim.Adam(learning_rate=LR)

    def loss_fn(model, spk, tgt):
        return mx.mean(nn.losses.cross_entropy(model(spk), tgt))
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    epoch_accs, epoch_times = [], []
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        total_loss, nb = 0.0, 0
        for xb_np, yb_np in get_batches_np(x_train, y_train, BATCH_SIZE):
            xb = mx.array(xb_np)
            yb = mx.array(yb_np)
            sp = mlxsnn.rate_encode(xb, num_steps=NUM_STEPS)
            loss, grads = loss_and_grad(model, sp, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_loss += loss.item()
            nb += 1
        et = time.time() - t0

        correct, total = 0, 0
        for xb_np, yb_np in get_batches_np(x_test, y_test, BATCH_SIZE, shuffle=False):
            xb = mx.array(xb_np)
            yb = mx.array(yb_np)
            sp = mlxsnn.rate_encode(xb, num_steps=NUM_STEPS)
            mem = model(sp)
            mx.eval(mem)
            preds = mx.argmax(mem, axis=1)
            mx.eval(preds)
            correct += mx.sum(preds == yb).item()
            total += yb_np.shape[0]

        acc = correct / total
        epoch_accs.append(acc)
        epoch_times.append(et)
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} — "
              f"loss: {total_loss/nb:.4f}, acc: {acc:.2%}, time: {et:.1f}s")

    try:
        peak_mb = mx.get_peak_memory() / (1024 * 1024)
    except Exception:
        try:
            peak_mb = mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            peak_mb = -1

    return {
        "accs": epoch_accs,
        "times": epoch_times,
        "final_acc": epoch_accs[-1],
        "best_acc": max(epoch_accs),
        "total_time": sum(epoch_times),
        "avg_epoch_time": np.mean(epoch_times),
        "peak_memory_mb": peak_mb,
    }


# ---------------------------------------------------------------------------
# 2. snnTorch (PyTorch MPS GPU)
# ---------------------------------------------------------------------------

def run_snntorch_mps(x_train, y_train, x_test, y_test):
    import torch
    import torch.nn as tnn
    import snntorch as snn
    from snntorch import surrogate

    if not torch.backends.mps.is_available():
        print("\nMPS not available, skipping.")
        return None

    device = torch.device("mps")
    print("\n" + "=" * 60)
    print("2. snnTorch (PyTorch MPS GPU on M3 Max)")
    print("=" * 60)

    class SpikingMLP(tnn.Module):
        def __init__(self):
            super().__init__()
            sg = surrogate.fast_sigmoid(slope=25)
            self.fc1 = tnn.Linear(784, HIDDEN)
            self.lif1 = snn.Leaky(beta=BETA, spike_grad=sg)
            self.fc2 = tnn.Linear(HIDDEN, 10)
            self.lif2 = snn.Leaky(beta=BETA, spike_grad=sg,
                                  reset_mechanism="none")

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            for t in range(NUM_STEPS):
                spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
                spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            return mem2

    model = SpikingMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = tnn.CrossEntropyLoss()

    # Warmup MPS
    dummy = torch.randn(NUM_STEPS, 2, 784, device=device)
    _ = model(dummy)
    torch.mps.synchronize()

    epoch_accs, epoch_times = [], []
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0.0, 0
        for xb_np, yb_np in get_batches_np(x_train, y_train, BATCH_SIZE):
            xb = torch.tensor(xb_np, device=device)
            yb = torch.tensor(yb_np, dtype=torch.long, device=device)
            xb = xb.unsqueeze(0).repeat(NUM_STEPS, 1, 1)
            sp = torch.bernoulli(xb)

            optimizer.zero_grad()
            mem = model(sp)
            loss = criterion(mem, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        torch.mps.synchronize()
        et = time.time() - t0

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb_np, yb_np in get_batches_np(x_test, y_test, BATCH_SIZE, shuffle=False):
                xb = torch.tensor(xb_np, device=device)
                yb = torch.tensor(yb_np, dtype=torch.long, device=device)
                xb = xb.unsqueeze(0).repeat(NUM_STEPS, 1, 1)
                sp = torch.bernoulli(xb)
                mem = model(sp)
                preds = mem.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb_np.shape[0]

        acc = correct / total
        epoch_accs.append(acc)
        epoch_times.append(et)
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} — "
              f"loss: {total_loss/nb:.4f}, acc: {acc:.2%}, time: {et:.1f}s")

    # MPS memory
    try:
        peak_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
    except Exception:
        peak_mb = -1

    return {
        "accs": epoch_accs,
        "times": epoch_times,
        "final_acc": epoch_accs[-1],
        "best_acc": max(epoch_accs),
        "total_time": sum(epoch_times),
        "avg_epoch_time": np.mean(epoch_times),
        "peak_memory_mb": peak_mb,
    }


# ---------------------------------------------------------------------------
# 3. snnTorch (PyTorch CPU)
# ---------------------------------------------------------------------------

def run_snntorch_cpu(x_train, y_train, x_test, y_test):
    import torch
    import torch.nn as tnn
    import snntorch as snn
    from snntorch import surrogate

    device = torch.device("cpu")
    print("\n" + "=" * 60)
    print("3. snnTorch (PyTorch CPU on M3 Max)")
    print("=" * 60)

    class SpikingMLP(tnn.Module):
        def __init__(self):
            super().__init__()
            sg = surrogate.fast_sigmoid(slope=25)
            self.fc1 = tnn.Linear(784, HIDDEN)
            self.lif1 = snn.Leaky(beta=BETA, spike_grad=sg)
            self.fc2 = tnn.Linear(HIDDEN, 10)
            self.lif2 = snn.Leaky(beta=BETA, spike_grad=sg,
                                  reset_mechanism="none")

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            for t in range(NUM_STEPS):
                spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
                spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            return mem2

    model = SpikingMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = tnn.CrossEntropyLoss()

    epoch_accs, epoch_times = [], []
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        total_loss, nb = 0.0, 0
        for xb_np, yb_np in get_batches_np(x_train, y_train, BATCH_SIZE):
            xb = torch.tensor(xb_np, device=device)
            yb = torch.tensor(yb_np, dtype=torch.long, device=device)
            xb = xb.unsqueeze(0).repeat(NUM_STEPS, 1, 1)
            sp = torch.bernoulli(xb)

            optimizer.zero_grad()
            mem = model(sp)
            loss = criterion(mem, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        et = time.time() - t0

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb_np, yb_np in get_batches_np(x_test, y_test, BATCH_SIZE, shuffle=False):
                xb = torch.tensor(xb_np, device=device)
                yb = torch.tensor(yb_np, dtype=torch.long, device=device)
                xb = xb.unsqueeze(0).repeat(NUM_STEPS, 1, 1)
                sp = torch.bernoulli(xb)
                mem = model(sp)
                preds = mem.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb_np.shape[0]

        acc = correct / total
        epoch_accs.append(acc)
        epoch_times.append(et)
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} — "
              f"loss: {total_loss/nb:.4f}, acc: {acc:.2%}, time: {et:.1f}s")

    return {
        "accs": epoch_accs,
        "times": epoch_times,
        "final_acc": epoch_accs[-1],
        "best_acc": max(epoch_accs),
        "total_time": sum(epoch_times),
        "avg_epoch_time": np.mean(epoch_times),
        "peak_memory_mb": -1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_3way(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, NUM_EPOCHS + 1)

    colors = {"mlx_snn": "#2196F3", "snntorch_mps": "#4CAF50", "snntorch_cpu": "#FF5722"}
    labels = {"mlx_snn": "mlx-snn (MLX GPU)", "snntorch_mps": "snnTorch (MPS GPU)",
              "snntorch_cpu": "snnTorch (CPU)"}
    markers = {"mlx_snn": "o", "snntorch_mps": "^", "snntorch_cpu": "s"}

    for key in ["mlx_snn", "snntorch_mps", "snntorch_cpu"]:
        if key not in results or results[key] is None:
            continue
        r = results[key]
        ax1.plot(epochs, [a * 100 for a in r["accs"]],
                 f"{markers[key]}-", label=labels[key],
                 color=colors[key], linewidth=1.5, markersize=4)
        ax2.plot(epochs, r["times"],
                 f"{markers[key]}-", label=labels[key],
                 color=colors[key], linewidth=1.5, markersize=4)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("MNIST Classification Accuracy", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Training Time per Epoch", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("paper/figures/training_curves.pdf", dpi=150, bbox_inches="tight")
    print("\nSaved: paper/figures/training_curves.pdf")


if __name__ == "__main__":
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")

    results = {}
    results["mlx_snn"] = run_mlxsnn(x_train, y_train, x_test, y_test)
    results["snntorch_mps"] = run_snntorch_mps(x_train, y_train, x_test, y_test)
    results["snntorch_cpu"] = run_snntorch_cpu(x_train, y_train, x_test, y_test)

    print("\n" + "=" * 60)
    print("3-WAY BENCHMARK SUMMARY")
    print("=" * 60)
    for key, label in [("mlx_snn", "mlx-snn (MLX GPU)"),
                        ("snntorch_mps", "snnTorch (MPS GPU)"),
                        ("snntorch_cpu", "snnTorch (CPU)")]:
        r = results.get(key)
        if r is None:
            print(f"  {label:30s}  SKIPPED")
            continue
        mem_str = f"{r['peak_memory_mb']:.0f}MB" if r['peak_memory_mb'] > 0 else "N/A"
        print(f"  {label:30s}  acc={r['best_acc']:.2%}  "
              f"time={r['total_time']:.1f}s  "
              f"avg_epoch={r['avg_epoch_time']:.1f}s  "
              f"mem={mem_str}")

    with open("paper/figures/bench_3way_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    plot_3way(results)
    print("\nResults saved to paper/figures/bench_3way_results.json")
