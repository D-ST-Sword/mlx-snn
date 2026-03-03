"""Benchmark: MNIST classification — mlx-snn vs snnTorch.

Trains the same 2-layer SNN architecture on MNIST using both mlx-snn and
snnTorch, recording accuracy, training time, and peak memory usage.

Output:
    paper/figures/training_curves.pdf — accuracy vs epoch comparison
    paper/figures/mnist_results.json — detailed benchmark data
"""

import sys
sys.path.insert(0, ".")

import json
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mlxsnn


# ---------------------------------------------------------------------------
# Architecture: 784 -> 128 (LIF) -> 10 (LIF, no reset)
# ---------------------------------------------------------------------------

class SpikingMLP_MLX(nn.Module):
    """mlx-snn version of the benchmark model."""

    def __init__(self, num_steps=25, beta=0.95):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = mlxsnn.Leaky(beta=beta, threshold=1.0)
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = mlxsnn.Leaky(beta=beta, threshold=1.0, reset_mechanism="none")

    def __call__(self, x):
        batch_size = x.shape[1]
        state1 = self.lif1.init_state(batch_size, 128)
        state2 = self.lif2.init_state(batch_size, 10)

        for t in range(self.num_steps):
            h = self.fc1(x[t])
            spk1, state1 = self.lif1(h, state1)
            h = self.fc2(spk1)
            spk2, state2 = self.lif2(h, state2)

        return state2["mem"]


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

    try:
        import gzip
        import os
        import urllib.request

        url_base = "http://yann.lecun.com/exdb/mnist/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images": "t10k-images-idx3-ubyte.gz",
            "test_labels": "t10k-labels-idx1-ubyte.gz",
        }
        data_dir = os.path.join(os.path.dirname(__file__), ".mnist_data")
        os.makedirs(data_dir, exist_ok=True)

        def download_and_parse_images(fname):
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                urllib.request.urlretrieve(url_base + fname, path)
            with gzip.open(path, "rb") as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 784).astype(np.float32) / 255.0

        def download_and_parse_labels(fname):
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                urllib.request.urlretrieve(url_base + fname, path)
            with gzip.open(path, "rb") as f:
                return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int32)

        x_train = download_and_parse_images(files["train_images"])
        y_train = download_and_parse_labels(files["train_labels"])
        x_test = download_and_parse_images(files["test_images"])
        y_test = download_and_parse_labels(files["test_labels"])
        return x_train, y_train, x_test, y_test
    except Exception as e:
        raise RuntimeError(f"Could not load MNIST: {e}")


def get_batches(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield x[idx], y[idx]


# ---------------------------------------------------------------------------
# mlx-snn training
# ---------------------------------------------------------------------------

def train_mlxsnn(x_train, y_train, x_test, y_test,
                 num_epochs=15, batch_size=128, num_steps=25):
    print("\n" + "=" * 60)
    print("Training with mlx-snn")
    print("=" * 60)

    model = SpikingMLP_MLX(num_steps=num_steps)
    optimizer = optim.Adam(learning_rate=2e-3)

    def loss_fn(model, spikes_in, targets):
        mem_out = model(spikes_in)
        return mx.mean(nn.losses.cross_entropy(mem_out, targets))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    epoch_accs = []
    epoch_losses = []
    epoch_times = []

    for epoch in range(num_epochs):
        t0 = time.time()
        total_loss = 0.0
        num_batches = 0

        for x_batch_np, y_batch_np in get_batches(x_train, y_train, batch_size):
            x_batch = mx.array(x_batch_np)
            y_batch = mx.array(y_batch_np)
            spikes_in = mlxsnn.rate_encode(x_batch, num_steps=num_steps)
            loss, grads = loss_and_grad(model, spikes_in, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_loss += loss.item()
            num_batches += 1

        epoch_time = time.time() - t0
        avg_loss = total_loss / num_batches

        # Evaluate
        correct, total = 0, 0
        for x_batch_np, y_batch_np in get_batches(x_test, y_test, batch_size, shuffle=False):
            x_batch = mx.array(x_batch_np)
            y_batch = mx.array(y_batch_np)
            spikes_in = mlxsnn.rate_encode(x_batch, num_steps=num_steps)
            mem_out = model(spikes_in)
            mx.eval(mem_out)
            preds = mx.argmax(mem_out, axis=1)
            mx.eval(preds)
            correct += mx.sum(preds == y_batch).item()
            total += y_batch_np.shape[0]

        acc = correct / total
        epoch_accs.append(acc)
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch+1:2d}/{num_epochs} — loss: {avg_loss:.4f}, "
              f"acc: {acc:.2%}, time: {epoch_time:.1f}s")

    # Peak memory (MLX Metal)
    try:
        peak_mem_bytes = mx.get_peak_memory()
        peak_mem_mb = peak_mem_bytes / (1024 * 1024)
    except Exception:
        peak_mem_mb = -1.0

    return {
        "accs": epoch_accs,
        "losses": epoch_losses,
        "times": epoch_times,
        "total_time": sum(epoch_times),
        "final_acc": epoch_accs[-1],
        "peak_memory_mb": peak_mem_mb,
    }


# ---------------------------------------------------------------------------
# snnTorch training (optional — requires snntorch + torch installed)
# ---------------------------------------------------------------------------

def train_snntorch(x_train, y_train, x_test, y_test,
                   num_epochs=15, batch_size=128, num_steps=25):
    try:
        import torch
        import torch.nn as tnn
        import snntorch as snn
        from snntorch import surrogate
    except ImportError:
        print("\nsnnTorch or PyTorch not installed. Skipping snnTorch benchmark.")
        return None

    print("\n" + "=" * 60)
    print("Training with snnTorch (PyTorch CPU)")
    print("=" * 60)

    class SpikingMLP_snnTorch(tnn.Module):
        def __init__(self, num_steps=25, beta=0.95):
            super().__init__()
            self.num_steps = num_steps
            spike_grad = surrogate.fast_sigmoid(slope=25)
            self.fc1 = tnn.Linear(784, 128)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.fc2 = tnn.Linear(128, 10)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                                  reset_mechanism="none")

        def forward(self, x):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()

            for t in range(self.num_steps):
                h = self.fc1(x[t])
                spk1, mem1 = self.lif1(h, mem1)
                h = self.fc2(spk1)
                spk2, mem2 = self.lif2(h, mem2)

            return mem2

    device = torch.device("cpu")
    model = SpikingMLP_snnTorch(num_steps=num_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    criterion = tnn.CrossEntropyLoss()

    epoch_accs = []
    epoch_losses = []
    epoch_times = []

    for epoch in range(num_epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for x_batch_np, y_batch_np in get_batches(x_train, y_train, batch_size):
            x_batch = torch.tensor(x_batch_np, dtype=torch.float32, device=device)
            y_batch = torch.tensor(y_batch_np, dtype=torch.long, device=device)

            # Rate encode: Poisson spikes
            x_batch = x_batch.unsqueeze(0).repeat(num_steps, 1, 1)
            spikes_in = torch.bernoulli(x_batch)

            optimizer.zero_grad()
            mem_out = model(spikes_in)
            loss = criterion(mem_out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_time = time.time() - t0
        avg_loss = total_loss / num_batches

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch_np, y_batch_np in get_batches(x_test, y_test, batch_size, shuffle=False):
                x_batch = torch.tensor(x_batch_np, dtype=torch.float32, device=device)
                y_batch = torch.tensor(y_batch_np, dtype=torch.long, device=device)

                x_batch = x_batch.unsqueeze(0).repeat(num_steps, 1, 1)
                spikes_in = torch.bernoulli(x_batch)

                mem_out = model(spikes_in)
                preds = mem_out.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch_np.shape[0]

        acc = correct / total
        epoch_accs.append(acc)
        epoch_losses.append(avg_loss)
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch+1:2d}/{num_epochs} — loss: {avg_loss:.4f}, "
              f"acc: {acc:.2%}, time: {epoch_time:.1f}s")

    return {
        "accs": epoch_accs,
        "losses": epoch_losses,
        "times": epoch_times,
        "total_time": sum(epoch_times),
        "final_acc": epoch_accs[-1],
        "peak_memory_mb": -1.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_training_curves(mlx_results, snntorch_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(mlx_results["accs"]) + 1)

    # Accuracy
    ax1.plot(epochs, [a * 100 for a in mlx_results["accs"]],
             "o-", label="mlx-snn (M3 Max)", color="#2196F3", linewidth=1.5)
    if snntorch_results:
        ax1.plot(epochs, [a * 100 for a in snntorch_results["accs"]],
                 "s--", label="snnTorch (CPU)", color="#FF5722", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("MNIST Classification Accuracy", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Training time per epoch
    ax2.bar(
        [e - 0.15 for e in epochs], mlx_results["times"],
        width=0.3, label="mlx-snn", color="#2196F3", alpha=0.8,
    )
    if snntorch_results:
        ax2.bar(
            [e + 0.15 for e in epochs], snntorch_results["times"],
            width=0.3, label="snnTorch", color="#FF5722", alpha=0.8,
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Training Time per Epoch", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("paper/figures/training_curves.pdf", dpi=150, bbox_inches="tight")
    print("\nSaved: paper/figures/training_curves.pdf")


def main():
    num_epochs = 15
    batch_size = 128
    num_steps = 25

    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")

    # Run mlx-snn benchmark
    mlx_results = train_mlxsnn(
        x_train, y_train, x_test, y_test,
        num_epochs=num_epochs, batch_size=batch_size, num_steps=num_steps,
    )

    # Run snnTorch benchmark (if available)
    snntorch_results = train_snntorch(
        x_train, y_train, x_test, y_test,
        num_epochs=num_epochs, batch_size=batch_size, num_steps=num_steps,
    )

    # Summary
    print("\n" + "=" * 60)
    print("MNIST BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  mlx-snn:   acc={mlx_results['final_acc']:.2%}  "
          f"time={mlx_results['total_time']:.1f}s  "
          f"mem={mlx_results['peak_memory_mb']:.0f}MB")
    if snntorch_results:
        print(f"  snnTorch:  acc={snntorch_results['final_acc']:.2%}  "
              f"time={snntorch_results['total_time']:.1f}s")

    # Save results
    results = {"mlx_snn": mlx_results}
    if snntorch_results:
        results["snntorch"] = snntorch_results

    with open("paper/figures/mnist_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    plot_training_curves(mlx_results, snntorch_results)

    print("\nResults saved to paper/figures/mnist_results.json")


if __name__ == "__main__":
    main()
