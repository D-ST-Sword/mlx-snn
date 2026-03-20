"""snnTorch SHD (Spiking Heidelberg Digits) benchmark suite.

Runs on V100 GPU with PyTorch + snnTorch.  Results saved as CSV for
comparison with mlx-snn on M3 Max (shd_mlxsnn.py).

Models (mirrors shd_mlxsnn.py):
  M1. FC-SNN 2-layer:  700→256→20, LIF×2            (~185K params)
  M2. FC-SNN 3-layer:  700→256→128→20, LIF×3         (~220K params)
  M3. Recurrent SNN:   700→256→20, RLeaky(all_to_all) (~250K params)
  M4. Reservoir SNN:   700→(500 reservoir)→20          (~10K trainable)

Usage:
    CUDA_VISIBLE_DEVICES=0 python shd_snntorch.py
"""

import os
import csv
import time

import torch
import torch.nn as tnn
import torch.optim as optim
import numpy as np
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Config (identical to mlx-snn benchmark)
# ============================================================

NUM_STEPS = 100
BATCH_SIZE = 128
NUM_EPOCHS = 30
LR = 1e-3
BETA = 0.9
SEEDS = [42, 123, 7]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_shd_snntorch")
os.makedirs(OUT_DIR, exist_ok=True)

SPIKE_GRAD = surrogate.atan(alpha=2.0)


def log(msg):
    print(msg, flush=True)


# ============================================================
# Data loading (manual HDF5, no mlxsnn dependency)
# ============================================================

class SHDDataset(Dataset):
    """SHD dataset loaded directly from HDF5 files."""

    def __init__(self, h5_path, num_steps=100, num_units=700, max_time=None):
        import h5py

        with h5py.File(h5_path, "r") as f:
            times_ds = f["spikes/times"]
            units_ds = f["spikes/units"]

            if hasattr(times_ds, "keys"):
                keys = sorted(times_ds.keys(), key=int)
                self.spike_times = [np.array(times_ds[k]) for k in keys]
                self.spike_units = [
                    np.array(units_ds[k]).astype(int) for k in keys
                ]
            else:
                n = times_ds.shape[0]
                self.spike_times = [np.array(times_ds[i]) for i in range(n)]
                self.spike_units = [
                    np.array(units_ds[i]).astype(int) for i in range(n)
                ]
            self.labels = np.array(f["labels"]).astype(int)

        self.num_steps = num_steps
        self.num_units = num_units

        if max_time is None:
            self.max_time = max(
                t.max() for t in self.spike_times if t.size > 0
            )
        else:
            self.max_time = max_time

        log(f"  Loaded {h5_path}: {len(self.labels)} samples")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        times = self.spike_times[idx]
        units = self.spike_units[idx]

        frame = np.zeros((self.num_steps, self.num_units), dtype=np.float32)
        if times.size > 0:
            valid = times <= self.max_time
            times = times[valid]
            units = units[valid]
            bin_idx = np.floor(times / self.max_time * self.num_steps).astype(int)
            bin_idx = np.clip(bin_idx, 0, self.num_steps - 1)
            units = np.clip(units, 0, self.num_units - 1)
            frame[bin_idx, units] = 1.0

        return torch.tensor(frame), self.labels[idx]


def collate_time_first(batch):
    """Custom collate: stack to [B, T, C] then transpose to [T, B, C]."""
    spikes, labels = zip(*batch)
    x = torch.stack(spikes, dim=0)        # [B, T, C]
    x = x.permute(1, 0, 2).contiguous()  # [T, B, C]
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


# ============================================================
# M1: FC-SNN 2-layer
# ============================================================

class FCSNN2L(tnn.Module):
    """700→256→20, LIF×2."""

    def __init__(self):
        super().__init__()
        self.fc1 = tnn.Linear(700, 256)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)
        self.fc2 = tnn.Linear(256, 20)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)

    def forward(self, x_seq):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []

        for t in range(x_seq.shape[0]):
            cur1 = self.fc1(x_seq[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)

        return torch.stack(spk_rec)  # [T, B, 20]


# ============================================================
# M2: FC-SNN 3-layer
# ============================================================

class FCSNN3L(tnn.Module):
    """700→256→128→20, LIF×3."""

    def __init__(self):
        super().__init__()
        self.fc1 = tnn.Linear(700, 256)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)
        self.fc2 = tnn.Linear(256, 128)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)
        self.fc3 = tnn.Linear(128, 20)
        self.lif3 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)

    def forward(self, x_seq):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_rec = []

        for t in range(x_seq.shape[0]):
            cur1 = self.fc1(x_seq[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk_rec.append(spk3)

        return torch.stack(spk_rec)  # [T, B, 20]


# ============================================================
# M3: Recurrent SNN
# ============================================================

class RecurrentSNN(tnn.Module):
    """700→256(recurrent, all_to_all)→20.

    Uses snnTorch RLeaky with all_to_all=True, linear_features=256.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = tnn.Linear(700, 256)
        self.rlif1 = snn.RLeaky(
            beta=BETA,
            all_to_all=True,
            linear_features=256,
            spike_grad=SPIKE_GRAD,
        )
        self.fc2 = tnn.Linear(256, 20)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=SPIKE_GRAD)

    def forward(self, x_seq):
        spk1, mem1 = self.rlif1.init_rleaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []

        for t in range(x_seq.shape[0]):
            cur1 = self.fc1(x_seq[t])
            spk1, mem1 = self.rlif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)

        return torch.stack(spk_rec)  # [T, B, 20]


# ============================================================
# M4: Reservoir SNN (LSM-style)
# ============================================================

class ReservoirSNN(tnn.Module):
    """Fixed random reservoir (500 LIF neurons) + trainable linear readout.

    Mirrors mlx-snn LSM: random input projection + sparse recurrent
    weights, only the readout is trained.
    """

    def __init__(self, seed=42):
        super().__init__()
        self.reservoir_size = 500

        rng = np.random.default_rng(seed)

        # Random input weights (frozen)
        W_in = rng.standard_normal((700, 500)).astype(np.float32)
        W_in /= np.sqrt(700)
        self.register_buffer("W_in", torch.tensor(W_in))

        # Random sparse recurrent weights (frozen)
        connectivity = 0.1
        spectral_radius = 0.9
        mask = (rng.random((500, 500)) < connectivity).astype(np.float32)
        np.fill_diagonal(mask, 0)
        W_res = rng.standard_normal((500, 500)).astype(np.float32) * mask
        eigvals = np.abs(np.linalg.eigvals(W_res)).max()
        if eigvals > 0:
            W_res = W_res * (spectral_radius / eigvals)
        self.register_buffer("W_res", torch.tensor(W_res))

        self.threshold = 1.0
        self.beta = BETA

        # Trainable readout
        self.readout = tnn.Linear(500, 20)

    def forward(self, x_seq):
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

        return torch.stack(out_list)  # [T, B, 20]


# ============================================================
# Training loop
# ============================================================

MODELS = [
    ("m1_fc2l", FCSNN2L),
    ("m2_fc3l", FCSNN3L),
    ("m3_recurrent", RecurrentSNN),
    ("m4_lsm", ReservoirSNN),
]


def train_model(model_name, model_cls, train_loader, test_loader, seeds=SEEDS):
    log(f"\n{'='*60}")
    log(f"Training {model_name}")
    log(f"{'='*60}")

    all_results = []
    loss_fn = tnn.CrossEntropyLoss()

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if model_cls == ReservoirSNN:
            model = model_cls(seed=seed).to(DEVICE)
            # Only train readout, higher LR
            optimizer = optim.Adam(model.readout.parameters(), lr=5e-3)
        else:
            model = model_cls().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(NUM_EPOCHS):
            model.train()
            t0 = time.time()
            total_loss, correct, total = 0.0, 0, 0

            for x_seq, targets in train_loader:
                x_seq = x_seq.to(DEVICE)
                targets = targets.to(DEVICE)

                optimizer.zero_grad()
                spk_out = model(x_seq)
                logits = spk_out.sum(0)  # rate decode: spike count over time
                loss = loss_fn(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * targets.shape[0]
                preds = spk_out.sum(0).argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.shape[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            train_acc = correct / total

            # Test
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for x_seq, targets in test_loader:
                    x_seq = x_seq.to(DEVICE)
                    targets = targets.to(DEVICE)
                    spk_out = model(x_seq)
                    preds = spk_out.sum(0).argmax(dim=-1)
                    test_correct += (preds == targets).sum().item()
                    test_total += targets.shape[0]
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

def bench_throughput():
    log(f"\n{'='*60}")
    log(f"Throughput benchmark (inference only)")
    log(f"{'='*60}")

    results = []

    for model_name, model_cls in MODELS:
        if model_cls == ReservoirSNN:
            model = model_cls(seed=42).to(DEVICE).eval()
        else:
            model = model_cls().to(DEVICE).eval()

        x = torch.rand(NUM_STEPS, BATCH_SIZE, 700, device=DEVICE)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(x)

        # Timed
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        with torch.no_grad():
            for _ in range(20):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
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
    log(f"snnTorch SHD Benchmark Suite")
    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"PyTorch {torch.__version__}, snnTorch {snn.__version__}")
    log(f"Seeds: {SEEDS}, Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, "
        f"T: {NUM_STEPS}, LR: {LR}, Beta: {BETA}")
    log(f"Surrogate: atan (alpha=2.0)")
    log(f"Output: {OUT_DIR}/")

    # Load data
    log(f"\nLoading SHD dataset...")
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "shd")
    train_ds = SHDDataset(os.path.join(data_root, "shd_train.h5"), num_steps=NUM_STEPS)
    test_ds = SHDDataset(os.path.join(data_root, "shd_test.h5"), num_steps=NUM_STEPS)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_time_first,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_time_first,
    )

    # Train all models
    for model_name, model_cls in MODELS:
        train_model(model_name, model_cls, train_loader, test_loader)

    # Throughput
    bench_throughput()

    log(f"\n{'='*60}")
    log(f"All benchmarks complete. Results in {OUT_DIR}/")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
