"""snnTorch v0.7 cross-platform benchmark suite.

Runs on V100 GPU with PyTorch + snnTorch.  Results saved as CSV for
comparison with mlx-snn on M3 Max.

Benchmarks:
  B1. FC SNN on MNIST (784-128-10, 25 timesteps, 5 seeds)
  B2. FC SNN on FashionMNIST (same architecture)
  B3. Conv SNN on FashionMNIST (2 conv blocks + FC)
  B4. Forward throughput (batch sweep, no training)
  B5. LSM-style reservoir + linear readout on MNIST

Usage:
    CUDA_VISIBLE_DEVICES=0 python snntorch_v07_bench.py
"""

import os
import csv
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import snntorch as snn
from snntorch import surrogate
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# Config
# ============================================================

NUM_STEPS = 25
BATCH_SIZE = 128
NUM_EPOCHS = 10
LR = 1e-3
SEEDS = [42, 123, 7, 2024, 99]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v07")
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)


# ============================================================
# Data
# ============================================================

def load_dataset(name="MNIST"):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    cls = datasets.MNIST if name == "MNIST" else datasets.FashionMNIST
    train_ds = cls("./data", train=True, download=True, transform=transform)
    test_ds = cls("./data", train=False, download=True, transform=transform)
    return train_ds, test_ds


# ============================================================
# B1/B2: FC SNN
# ============================================================

class FCSNN(nn.Module):
    def __init__(self, beta=0.9):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25))
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(slope=25))

    def forward(self, x_seq):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []
        mem_rec = []
        for t in range(x_seq.shape[0]):
            cur1 = self.fc1(x_seq[t].view(x_seq.shape[1], -1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)
        return torch.stack(spk_rec), torch.stack(mem_rec)


def train_fc_snn(dataset_name, seeds=SEEDS):
    log(f"\n{'='*60}")
    log(f"B{'1' if dataset_name=='MNIST' else '2'}: FC SNN on {dataset_name}")
    log(f"{'='*60}")

    train_ds, test_ds = load_dataset(dataset_name)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    all_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = FCSNN().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(NUM_EPOCHS):
            model.train()
            t0 = time.time()
            total_loss, correct, total = 0.0, 0, 0

            for data, targets in train_loader:
                data = data.to(DEVICE)
                targets = targets.to(DEVICE)
                # Rate encode
                x_seq = (torch.rand(NUM_STEPS, *data.shape, device=DEVICE) < data.unsqueeze(0)).float()

                optimizer.zero_grad()
                spk_out, mem_out = model(x_seq)

                # Sum spikes over time for rate decoding
                loss = loss_fn(spk_out.sum(0), targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * targets.shape[0]
                preds = spk_out.sum(0).argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.shape[0]

            elapsed = time.time() - t0
            train_acc = correct / total

            # Test
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for data, targets in test_loader:
                    data = data.to(DEVICE)
                    targets = targets.to(DEVICE)
                    x_seq = (torch.rand(NUM_STEPS, *data.shape, device=DEVICE) < data.unsqueeze(0)).float()
                    spk_out, _ = model(x_seq)
                    preds = spk_out.sum(0).argmax(dim=-1)
                    test_correct += (preds == targets).sum().item()
                    test_total += targets.shape[0]
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

    # Save
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
        sg = surrogate.fast_sigmoid(slope=25)
        # Block 1: 1->32, 28->14
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)
        self.pool1 = nn.MaxPool2d(2)

        # Block 2: 32->64, 14->7
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=sg)
        self.pool2 = nn.MaxPool2d(2)

        # Classifier: 64*7*7 -> 10
        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=sg)

    def forward(self, x_seq):
        """x_seq: [T, B, 1, 28, 28]"""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()
        spk_rec = []

        for t in range(x_seq.shape[0]):
            h = self.pool1(self.lif1(self.bn1(self.conv1(x_seq[t])), mem1)[0])
            _, mem1 = self.lif1(self.bn1(self.conv1(x_seq[t])), mem1)
            spk1, mem1 = self.lif1(self.bn1(self.conv1(x_seq[t])), mem1)
            h = self.pool1(spk1)

            spk2, mem2 = self.lif2(self.bn2(self.conv2(h)), mem2)
            h = self.pool2(spk2)

            h = h.view(h.size(0), -1)
            spk_out, mem_out = self.lif_out(self.fc(h), mem_out)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)


class ConvSNNClean(nn.Module):
    """Clean Conv SNN without redundant calls."""
    def __init__(self, beta=0.9):
        super().__init__()
        sg = surrogate.fast_sigmoid(slope=25)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=sg)
        self.pool2 = nn.MaxPool2d(2)

        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=sg)

    def forward(self, x_seq):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()
        spk_rec = []

        for t in range(x_seq.shape[0]):
            spk1, mem1 = self.lif1(self.bn1(self.conv1(x_seq[t])), mem1)
            h = self.pool1(spk1)
            spk2, mem2 = self.lif2(self.bn2(self.conv2(h)), mem2)
            h = self.pool2(spk2)
            h = h.view(h.size(0), -1)
            spk_out, mem_out = self.lif_out(self.fc(h), mem_out)
            spk_rec.append(spk_out)

        return torch.stack(spk_rec)


def train_conv_snn(seeds=SEEDS):
    log(f"\n{'='*60}")
    log(f"B3: Conv SNN on FashionMNIST")
    log(f"{'='*60}")

    train_ds, test_ds = load_dataset("FashionMNIST")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    all_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = ConvSNNClean().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(NUM_EPOCHS):
            model.train()
            t0 = time.time()
            total_loss, correct, total = 0.0, 0, 0

            for data, targets in train_loader:
                data = data.to(DEVICE)
                targets = targets.to(DEVICE)
                x_seq = (torch.rand(NUM_STEPS, *data.shape, device=DEVICE) < data.unsqueeze(0)).float()

                optimizer.zero_grad()
                spk_out = model(x_seq)
                loss = loss_fn(spk_out.sum(0), targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * targets.shape[0]
                preds = spk_out.sum(0).argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.shape[0]

            elapsed = time.time() - t0
            train_acc = correct / total

            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for data, targets in test_loader:
                    data = data.to(DEVICE)
                    targets = targets.to(DEVICE)
                    x_seq = (torch.rand(NUM_STEPS, *data.shape, device=DEVICE) < data.unsqueeze(0)).float()
                    spk_out = model(x_seq)
                    preds = spk_out.sum(0).argmax(dim=-1)
                    test_correct += (preds == targets).sum().item()
                    test_total += targets.shape[0]
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
# B4: Forward throughput benchmark
# ============================================================

def bench_throughput():
    log(f"\n{'='*60}")
    log(f"B4: Forward throughput (inference only)")
    log(f"{'='*60}")

    results = []
    T = 25

    for model_name, model_cls, input_shape_fn in [
        ("FC-SNN", FCSNN, lambda B: (T, B, 1, 28, 28)),
        ("Conv-SNN", ConvSNNClean, lambda B: (T, B, 1, 28, 28)),
    ]:
        for batch_size in [32, 64, 128, 256, 512]:
            model = model_cls().to(DEVICE).eval()
            shape = input_shape_fn(batch_size)
            x = torch.rand(*shape, device=DEVICE)

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    if model_name == "FC-SNN":
                        model(x)
                    else:
                        model(x)

            # Timed
            torch.cuda.synchronize()
            times = []
            with torch.no_grad():
                for _ in range(20):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    if model_name == "FC-SNN":
                        model(x)
                    else:
                        model(x)
                    torch.cuda.synchronize()
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
# B5: Reservoir (LSM-style) + linear readout on MNIST
# ============================================================

class ReservoirSNN(nn.Module):
    """Fixed random reservoir with trainable linear readout.

    Mirrors mlx-snn LSM: random sparse input projection + recurrent
    weights, only the readout is trained.
    """

    def __init__(self, input_size=784, reservoir_size=500, output_size=10,
                 connectivity=0.1, beta=0.9, spectral_radius=0.9):
        super().__init__()
        self.reservoir_size = reservoir_size

        # Random input weights (frozen)
        W_in = torch.randn(input_size, reservoir_size) / np.sqrt(input_size)
        self.register_buffer("W_in", W_in)

        # Random sparse recurrent weights (frozen) with spectral radius scaling
        mask = (torch.rand(reservoir_size, reservoir_size) < connectivity).float()
        mask.fill_diagonal_(0)
        W_res = torch.randn(reservoir_size, reservoir_size) * mask
        eigvals = torch.linalg.eigvals(W_res).abs().max().item()
        if eigvals > 0:
            W_res = W_res * (spectral_radius / eigvals)
        self.register_buffer("W_res", W_res)

        self.threshold = 1.0
        self.beta = beta

        # Trainable readout
        self.readout = nn.Linear(reservoir_size, output_size)

    def forward(self, x_seq):
        """x_seq: [T, B, 784]"""
        T, B = x_seq.shape[0], x_seq.shape[1]
        mem = torch.zeros(B, self.reservoir_size, device=x_seq.device)
        spk = torch.zeros(B, self.reservoir_size, device=x_seq.device)
        out_list = []

        for t in range(T):
            I = x_seq[t] @ self.W_in + spk @ self.W_res
            mem = self.beta * mem + I
            spk = (mem >= self.threshold).float()
            mem = mem - spk * self.threshold
            out_list.append(self.readout(spk))

        return torch.stack(out_list)  # [T, B, 10]


def train_reservoir(seeds=SEEDS):
    log(f"\n{'='*60}")
    log(f"B5: Reservoir (LSM-style) on MNIST")
    log(f"{'='*60}")

    train_ds, test_ds = load_dataset("MNIST")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    all_results = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = ReservoirSNN(reservoir_size=500).to(DEVICE)
        # Only train readout
        optimizer = optim.Adam(model.readout.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(NUM_EPOCHS):
            model.train()
            t0 = time.time()
            total_loss, correct, total = 0.0, 0, 0

            for data, targets in train_loader:
                data = data.to(DEVICE).view(-1, 784)
                targets = targets.to(DEVICE)
                x_seq = (torch.rand(NUM_STEPS, *data.shape, device=DEVICE) < data.unsqueeze(0)).float()

                optimizer.zero_grad()
                out = model(x_seq)  # [T, B, 10]
                # Rate decode: sum over time
                logits = out.sum(0)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * targets.shape[0]
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.shape[0]

            elapsed = time.time() - t0
            train_acc = correct / total

            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for data, targets in test_loader:
                    data = data.to(DEVICE).view(-1, 784)
                    targets = targets.to(DEVICE)
                    x_seq = (torch.rand(NUM_STEPS, *data.shape, device=DEVICE) < data.unsqueeze(0)).float()
                    out = model(x_seq)
                    preds = out.sum(0).argmax(dim=-1)
                    test_correct += (preds == targets).sum().item()
                    test_total += targets.shape[0]
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
# Main
# ============================================================

def main():
    log(f"snnTorch v0.7 Benchmark Suite")
    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"PyTorch {torch.__version__}, snnTorch {snn.__version__}")
    log(f"Seeds: {SEEDS}, Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, T: {NUM_STEPS}")

    train_fc_snn("MNIST")
    train_fc_snn("FashionMNIST")
    train_conv_snn()
    bench_throughput()
    train_reservoir()

    log(f"\n{'='*60}")
    log(f"All benchmarks complete. Results in {OUT_DIR}/")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
