"""Quickstart: LIF network on MNIST with rate coding.

A minimal spiking neural network trained on MNIST using:
- Rate coding to convert pixel intensities to spike trains
- Two-layer feedforward SNN (Linear -> LIF -> Linear -> LIF)
- BPTT training with surrogate gradients
- Output decoded via final membrane potential (cross-entropy loss)

Usage:
    python examples/01_quickstart.py

Requirements:
    pip install -e .
    # MNIST data is fetched automatically
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mlxsnn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpikingMLP(nn.Module):
    """Two-layer spiking MLP for MNIST classification.

    Architecture:
        Input (784) -> Linear -> LIF -> Linear -> LIF -> Output (10)

    The output layer uses reset_mechanism="none" so that membrane potential
    accumulates freely — this is used as the classification logit.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10,
        beta: float = 0.95,
        num_steps: int = 25,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = mlxsnn.Leaky(beta=beta, threshold=1.0)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = mlxsnn.Leaky(beta=beta, threshold=1.0, reset_mechanism="none")

    def __call__(self, x: mx.array):
        """Forward pass over all timesteps.

        Args:
            x: Rate-encoded spikes [num_steps, batch, 784].

        Returns:
            mem_out: Final output membrane potential [batch, 10].
            spk_count: Total output spike count [batch, 10].
        """
        batch_size = x.shape[1]
        state1 = self.lif1.init_state(batch_size, self.hidden_size)
        state2 = self.lif2.init_state(batch_size, self.output_size)

        spk_count = mx.zeros((batch_size, self.output_size))

        for t in range(self.num_steps):
            h = self.fc1(x[t])
            spk1, state1 = self.lif1(h, state1)

            h = self.fc2(spk1)
            spk2, state2 = self.lif2(h, state2)

            spk_count = spk_count + spk2

        return state2["mem"], spk_count


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist():
    """Load MNIST dataset.

    Attempts sklearn, then raw download, then synthetic fallback.

    Returns:
        (x_train, y_train, x_test, y_test) as numpy arrays.
    """
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        x = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int32)
        x_train, x_test = x[:60000], x[60000:]
        y_train, y_test = y[:60000], y[60000:]
        return x_train, y_train, x_test, y_test
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
        print(f"Could not download MNIST: {e}")
        print("Using synthetic data for demonstration.")
        rng = np.random.default_rng(42)
        x_train = rng.random((1000, 784)).astype(np.float32)
        y_train = rng.integers(0, 10, size=1000).astype(np.int32)
        x_test = rng.random((200, 784)).astype(np.float32)
        y_test = rng.integers(0, 10, size=200).astype(np.int32)
        return x_train, y_train, x_test, y_test


def get_batches(x, y, batch_size, shuffle=True):
    """Yield mini-batches from data arrays."""
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield mx.array(x[idx]), mx.array(y[idx])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    # Hyperparameters
    num_steps = 25
    batch_size = 128
    num_epochs = 5
    learning_rate = 2e-3
    beta = 0.95

    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")

    # Build model
    model = SpikingMLP(
        input_size=784,
        hidden_size=128,
        output_size=10,
        beta=beta,
        num_steps=num_steps,
    )
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, _loss_fn)

    print(f"\nTraining SNN for {num_epochs} epochs...")
    print(f"  num_steps={num_steps}, beta={beta}, batch_size={batch_size}")
    print()

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in get_batches(x_train, y_train, batch_size):
            # Rate encode the batch: [num_steps, batch, 784]
            spikes_in = mlxsnn.rate_encode(x_batch, num_steps=num_steps)

            loss, grads = loss_and_grad(model, spikes_in, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Evaluate on test set
        test_acc = evaluate(model, x_test, y_test, num_steps, batch_size)
        print(f"  Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}, "
              f"test acc: {test_acc:.2%}")

    print("\nDone!")


def _loss_fn(model, spikes_in, targets):
    """Compute cross-entropy loss on output membrane potential."""
    mem_out, _ = model(spikes_in)
    return mx.mean(nn.losses.cross_entropy(mem_out, targets))


def evaluate(model, x_test, y_test, num_steps, batch_size):
    """Evaluate model accuracy on test set."""
    correct = 0
    total = 0

    for x_batch, y_batch in get_batches(x_test, y_test, batch_size, shuffle=False):
        spikes_in = mlxsnn.rate_encode(x_batch, num_steps=num_steps)
        mem_out, _ = model(spikes_in)
        mx.eval(mem_out)

        predictions = mx.argmax(mem_out, axis=1)
        mx.eval(predictions)

        correct += mx.sum(predictions == y_batch).item()
        total += y_batch.shape[0]

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    main()
