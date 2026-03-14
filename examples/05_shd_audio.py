"""SHD Audio Classification: spiking network for spoken digit recognition.

Trains a fully-connected SNN on the Spiking Heidelberg Digits (SHD)
dataset — 700-channel cochlea spike trains, 20 classes (digits 0-9
in English and German).

Usage:
    python examples/05_shd_audio.py --data_dir /path/to/shd

Requirements:
    pip install mlx-snn[datasets]
    # SHD data (shd_train.h5, shd_test.h5) must be pre-downloaded
    # from https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import mlxsnn
from mlxsnn.datasets import SHDDataset, create_dataloader


# ---------------------------------------------------------------------------
# Model: 2-layer FC SNN for SHD
# ---------------------------------------------------------------------------

class SHDSNN(nn.Module):
    """Fully-connected SNN for Spiking Heidelberg Digits.

    Architecture: 700 → 256 (LIF) → 128 (LIF) → 20 (readout LIF)
    """

    def __init__(self, num_classes=20):
        super().__init__()
        self.fc1 = nn.Linear(700, 256)
        self.lif1 = mlxsnn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(256, 128)
        self.lif2 = mlxsnn.Leaky(beta=0.9)
        self.fc_out = nn.Linear(128, num_classes)
        self.lif_out = mlxsnn.Leaky(beta=0.9)

    def __call__(self, x_seq):
        """x_seq: (T, B, 700) spike frames."""
        T, B = x_seq.shape[0], x_seq.shape[1]

        s1 = self.lif1.init_state(B, 256)
        s2 = self.lif2.init_state(B, 128)
        s_out = self.lif_out.init_state(B, 20)

        all_mem = []
        for t in range(T):
            h = self.fc1(x_seq[t])
            spk, s1 = self.lif1(h, s1)

            h = self.fc2(spk)
            spk, s2 = self.lif2(h, s2)

            h = self.fc_out(spk)
            spk, s_out = self.lif_out(h, s_out)
            all_mem.append(s_out["mem"])

        return mx.stack(all_mem)  # (T, B, 20)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, loss_and_grad_fn):
    total_loss, correct, total = 0.0, 0, 0
    for batch_x, batch_y in dataloader:
        # batch_x: (B, T, 700) → transpose to (T, B, 700)
        x_seq = mx.array(batch_x).transpose(1, 0, 2)
        y = mx.array(batch_y)

        loss, grads = loss_and_grad_fn(model, x_seq, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        total_loss += loss.item() * y.shape[0]

        # Accuracy from membrane potentials
        mem_out = model(x_seq)
        preds = mx.argmax(mx.mean(mem_out, axis=0), axis=-1)
        correct += mx.sum(preds == y).item()
        total += y.shape[0]

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, dataloader):
    correct, total = 0, 0
    for batch_x, batch_y in dataloader:
        x_seq = mx.array(batch_x).transpose(1, 0, 2)
        y = mx.array(batch_y)

        mem_out = model(x_seq)
        mx.eval(mem_out)
        preds = mx.argmax(mx.mean(mem_out, axis=0), axis=-1)
        correct += mx.sum(preds == y).item()
        total += y.shape[0]

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to directory with shd_train.h5/shd_test.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--T", type=int, default=100, help="Number of timesteps")
    args = parser.parse_args()

    print("SHD Audio Classification with SNN")
    print("=" * 50)

    # Load dataset
    print(f"Loading SHD from {args.data_dir}...")
    train_ds = SHDDataset(root=args.data_dir, train=True, num_steps=args.T)
    test_ds = SHDDataset(root=args.data_dir, train=False, num_steps=args.T)
    print(f"  Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")
    print(f"  Classes: {train_ds.NUM_CLASSES}, Channels: {train_ds.num_units}")
    print(f"  Timesteps: {args.T}")

    train_loader = create_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = create_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = SHDSNN(num_classes=train_ds.NUM_CLASSES)

    def loss_fn(model, x_seq, y):
        mem_out = model(x_seq)
        return mlxsnn.mse_membrane_loss(mem_out, y)

    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Train
    best_acc = 0.0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_and_grad
        )
        test_acc = evaluate(model, test_loader)
        elapsed = time.time() - t0
        best_acc = max(best_acc, test_acc)

        print(f"Epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  "
              f"train_acc={train_acc:.1%}  "
              f"test_acc={test_acc:.1%}  "
              f"time={elapsed:.1f}s")

    print(f"\nBest test accuracy: {best_acc:.1%}")


if __name__ == "__main__":
    main()
