"""Benchmark: Surrogate gradient comparison on MNIST.

Trains the same SpikingMLP architecture with different surrogate gradient
functions and compares final accuracy.

Output:
    paper/figures/surrogates.pdf — surrogate function plots
    paper/figures/surrogate_results.npz — accuracy data

Surrogate functions tested:
    1. fast_sigmoid (scale=25)
    2. arctan (scale=2)
    3. straight_through (scale=1)
"""

import sys
sys.path.insert(0, ".")

import time
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mlxsnn
from mlxsnn.surrogate import fast_sigmoid_surrogate, arctan_surrogate, straight_through_surrogate


# ---------------------------------------------------------------------------
# Model (same as quickstart, parameterized surrogate)
# ---------------------------------------------------------------------------

class SpikingMLP(nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden_size=128,
        output_size=10,
        beta=0.95,
        num_steps=25,
        surrogate_fn="fast_sigmoid",
    ):
        super().__init__()
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = mlxsnn.Leaky(
            beta=beta, threshold=1.0, surrogate_fn=surrogate_fn,
        )
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = mlxsnn.Leaky(
            beta=beta, threshold=1.0, reset_mechanism="none",
            surrogate_fn=surrogate_fn,
        )

    def __call__(self, x):
        batch_size = x.shape[1]
        state1 = self.lif1.init_state(batch_size, self.hidden_size)
        state2 = self.lif2.init_state(batch_size, self.output_size)

        for t in range(self.num_steps):
            h = self.fc1(x[t])
            spk1, state1 = self.lif1(h, state1)
            h = self.fc2(spk1)
            spk2, state2 = self.lif2(h, state2)

        return state2["mem"]


# ---------------------------------------------------------------------------
# Data loading (reuse from quickstart)
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
        yield mx.array(x[idx]), mx.array(y[idx])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(surrogate_name, x_train, y_train, x_test, y_test,
                       num_epochs=10, batch_size=128, num_steps=25):
    print(f"\n{'='*60}")
    print(f"Surrogate: {surrogate_name}")
    print(f"{'='*60}")

    model = SpikingMLP(surrogate_fn=surrogate_name, num_steps=num_steps)
    optimizer = optim.Adam(learning_rate=2e-3)

    def loss_fn(model, spikes_in, targets):
        mem_out = model(spikes_in)
        return mx.mean(nn.losses.cross_entropy(mem_out, targets))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    epoch_accs = []
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in get_batches(x_train, y_train, batch_size):
            spikes_in = mlxsnn.rate_encode(x_batch, num_steps=num_steps)
            loss, grads = loss_and_grad(model, spikes_in, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Test accuracy
        correct, total = 0, 0
        for x_batch, y_batch in get_batches(x_test, y_test, batch_size, shuffle=False):
            spikes_in = mlxsnn.rate_encode(x_batch, num_steps=num_steps)
            mem_out = model(spikes_in)
            mx.eval(mem_out)
            predictions = mx.argmax(mem_out, axis=1)
            mx.eval(predictions)
            correct += mx.sum(predictions == y_batch).item()
            total += y_batch.shape[0]

        acc = correct / total
        epoch_accs.append(acc)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}, acc: {acc:.2%}")

    return epoch_accs, epoch_losses


def plot_surrogates():
    """Plot the 3 surrogate gradient functions (forward + backward)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping surrogate plots.")
        return

    x_np = np.linspace(-2, 2, 500)
    x_mx = mx.array(x_np.astype(np.float32))

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    surrogates = [
        ("Fast Sigmoid", fast_sigmoid_surrogate(25.0), 25.0),
        ("Arctan", arctan_surrogate(2.0), 2.0),
        ("Straight-Through", straight_through_surrogate(1.0), 1.0),
    ]

    for ax, (name, fn, scale) in zip(axes, surrogates):
        # Forward: Heaviside
        fwd = np.where(x_np >= 0, 1.0, 0.0)

        # Backward: smooth approximation gradient
        if "Sigmoid" in name:
            # Rational fast sigmoid: scale / (2 * (1 + scale*|x|)^2)
            grad = scale / (2.0 * (1.0 + scale * np.abs(x_np)) ** 2)
        elif "Arctan" in name:
            grad = scale / (2 * (1 + (np.pi / 2 * scale * x_np) ** 2))
        else:  # Straight-Through
            grad = np.where(np.abs(x_np) <= 0.5 / scale, scale, 0.0)

        ax.plot(x_np, fwd, "b-", linewidth=1.5, label="Forward (Heaviside)")
        ax.plot(x_np, grad, "r--", linewidth=1.5, label="Backward (gradient)")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("$x$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, max(np.max(grad), 1.0) * 1.2)

    plt.tight_layout()
    plt.savefig("paper/figures/surrogates.pdf", dpi=150, bbox_inches="tight")
    print("\nSaved: paper/figures/surrogates.pdf")


def main():
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()

    surrogates = ["fast_sigmoid", "arctan", "straight_through"]
    all_results = {}

    for sg in surrogates:
        t0 = time.time()
        accs, losses = train_and_evaluate(
            sg, x_train, y_train, x_test, y_test,
            num_epochs=10, batch_size=128, num_steps=25,
        )
        elapsed = time.time() - t0
        all_results[sg] = {
            "accs": accs,
            "losses": losses,
            "final_acc": accs[-1],
            "time_s": elapsed,
        }
        print(f"  Total time: {elapsed:.1f}s")

    # Save results
    np.savez(
        "paper/figures/surrogate_results.npz",
        **{f"{k}_accs": np.array(v["accs"]) for k, v in all_results.items()},
        **{f"{k}_losses": np.array(v["losses"]) for k, v in all_results.items()},
    )

    # Save summary as JSON for easy reading
    summary = {k: {"final_acc": v["final_acc"], "time_s": v["time_s"]}
               for k, v in all_results.items()}
    with open("paper/figures/surrogate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SURROGATE COMPARISON SUMMARY")
    print("=" * 60)
    for sg, res in all_results.items():
        print(f"  {sg:20s}  acc={res['final_acc']:.2%}  time={res['time_s']:.1f}s")

    # Plot surrogate functions
    plot_surrogates()


if __name__ == "__main__":
    main()
