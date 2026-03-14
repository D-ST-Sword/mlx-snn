"""Conv SNN on DVS-Gesture: event-driven gesture recognition.

Trains a convolutional SNN on the DVS128 Gesture dataset using
SpikingConv2d layers with max-pooling and surrogate gradient BPTT.

Usage:
    python examples/02_conv_snn_dvsg.py --data_dir /path/to/DVSGesture

Requirements:
    pip install mlx-snn
    # DVS-Gesture data must be pre-downloaded and extracted
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

import mlxsnn
from mlxsnn.datasets import DVSGestureDataset, create_dataloader


# ---------------------------------------------------------------------------
# Model: 3-block Conv SNN
# ---------------------------------------------------------------------------

class ConvSNN(nn.Module):
    """Conv SNN for DVS-Gesture (11 classes, 128x128 → pooled)."""

    def __init__(self, num_classes=11, T=16):
        super().__init__()
        self.T = T

        # Block 1: 2 → 32 channels, 128→64
        self.conv1 = mlxsnn.SpikingConv2d(2, 32, kernel_size=3, padding=1)
        self.pool1 = mlxsnn.SpikingMaxPool2d(kernel_size=2, stride=2)

        # Block 2: 32 → 64 channels, 64→32
        self.conv2 = mlxsnn.SpikingConv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = mlxsnn.SpikingMaxPool2d(kernel_size=2, stride=2)

        # Block 3: 64 → 128 channels, 32→16
        self.conv3 = mlxsnn.SpikingConv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = mlxsnn.SpikingMaxPool2d(kernel_size=2, stride=2)

        # Classifier
        self.flatten = mlxsnn.SpikingFlatten()
        self.fc = nn.Linear(128 * 16 * 16, num_classes)
        self.lif_out = mlxsnn.Leaky(beta=0.9)

    def __call__(self, x_seq):
        """x_seq: (T, B, H, W, C) event frames."""
        T = x_seq.shape[0]
        B = x_seq.shape[1]

        # Init states
        s1 = {"mem": mx.zeros((B, 64, 64, 32))}
        s2 = {"mem": mx.zeros((B, 32, 32, 64))}
        s3 = {"mem": mx.zeros((B, 16, 16, 128))}
        s_out = self.lif_out.init_state(B, 11)

        all_mem = []
        for t in range(T):
            # Conv blocks
            h, s1 = self.conv1(x_seq[t], s1)
            h = self.pool1(h)
            h, s2 = self.conv2(h, s2)
            h = self.pool2(h)
            h, s3 = self.conv3(h, s3)
            h = self.pool3(h)

            # Classify
            h = self.flatten(h)
            h = self.fc(h)
            spk, s_out = self.lif_out(h, s_out)
            all_mem.append(s_out["mem"])

        return mx.stack(all_mem)  # (T, B, 11)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, loss_and_grad_fn):
    total_loss, correct, total = 0.0, 0, 0
    for batch_x, batch_y in dataloader:
        # batch_x: (B, T, H, W, C) → transpose to (T, B, H, W, C)
        x_seq = mx.array(batch_x).transpose(1, 0, 2, 3, 4)
        y = mx.array(batch_y)

        loss, grads = loss_and_grad_fn(model, x_seq, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        total_loss += loss.item() * y.shape[0]
        preds = mx.argmax(mx.mean(x_seq, axis=0), axis=-1)  # placeholder
        total += y.shape[0]

    return total_loss / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to DVSGesture directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--T", type=int, default=16, help="Number of timesteps")
    args = parser.parse_args()

    print("Conv SNN on DVS-Gesture")
    print("=" * 50)

    # Load dataset
    print(f"Loading DVS-Gesture from {args.data_dir}...")
    train_ds = DVSGestureDataset(root=args.data_dir, train=True,
                                  num_steps=args.T, spatial_size=128)
    test_ds = DVSGestureDataset(root=args.data_dir, train=False,
                                 num_steps=args.T, spatial_size=128)
    print(f"  Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    train_loader = create_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = create_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = ConvSNN(num_classes=11, T=args.T)

    def loss_fn(model, x_seq, y):
        mem_out = model(x_seq)
        return mlxsnn.mse_membrane_loss(mem_out, y)

    optimizer = optim.Adam(learning_rate=args.lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Train
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, loss_and_grad)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"time={elapsed:.1f}s")


if __name__ == "__main__":
    main()
