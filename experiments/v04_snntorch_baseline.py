"""snnTorch baseline experiments for cross-framework comparison.

Mirrors v04_benchmarks.py but uses snnTorch + PyTorch on GPU.
Runs on V100 with CUDA_VISIBLE_DEVICES=2,3.

Experiments:
1. Neuron type comparison (Leaky vs RLeaky vs Synaptic vs RSynaptic)
2. Learnable parameter ablation (learn_beta, learn_threshold)
3. Surrogate gradient comparison (fast_sigmoid, arctan, sigmoid, triangular, straight_through)
4. Loss function comparison (ce_rate, ce_count, mse_membrane)
5. Encoding comparison (rate, latency, delta)

Same config: Linear(784,128) -> Neuron -> Linear(128,10) -> Neuron
Batch 128, 25 epochs, lr=1e-3, Adam, 25 timesteps, 3 seeds.

Usage:
    CUDA_VISIBLE_DEVICES=2,3 python v04_snntorch_baseline.py
"""

import os
import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import snntorch as snn
from snntorch import surrogate, utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ============================================================
# Config (identical to mlx-snn)
# ============================================================

NUM_STEPS = 25
BATCH_SIZE = 128
NUM_EPOCHS = 25
LR = 1e-3
SEEDS = [42, 123, 7]
HIDDEN = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v04_snntorch")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Data loading
# ============================================================

def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten to 784
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return train_ds, test_ds


# ============================================================
# Generic SNN model (mirrors mlx-snn SNN class)
# ============================================================

class SNN(nn.Module):
    def __init__(self, neuron_cls1, neuron_cls2, neuron_kwargs1, neuron_kwargs2):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.n1 = neuron_cls1(**neuron_kwargs1)
        self.fc2 = nn.Linear(HIDDEN, 10)
        self.n2 = neuron_cls2(**neuron_kwargs2)

    def forward(self, x_seq):
        """x_seq: [T, batch, 784]"""
        batch_size = x_seq.shape[1]

        # Initialize hidden states
        mem1 = torch.zeros(batch_size, HIDDEN, device=x_seq.device)
        mem2 = torch.zeros(batch_size, 10, device=x_seq.device)

        # For recurrent neurons
        spk1_prev = torch.zeros(batch_size, HIDDEN, device=x_seq.device)
        spk2_prev = torch.zeros(batch_size, 10, device=x_seq.device)

        # For synaptic neurons
        syn1 = torch.zeros(batch_size, HIDDEN, device=x_seq.device)
        syn2 = torch.zeros(batch_size, 10, device=x_seq.device)

        spk_rec = []
        mem_rec = []

        for t in range(x_seq.shape[0]):
            h = self.fc1(x_seq[t])

            # Hidden layer
            if isinstance(self.n1, snn.RLeaky):
                spk1, mem1 = self.n1(h, spk1_prev, mem1)
                spk1_prev = spk1
            elif isinstance(self.n1, snn.RSynaptic):
                spk1, syn1, mem1 = self.n1(h, spk1_prev, syn1, mem1)
                spk1_prev = spk1
            elif isinstance(self.n1, snn.Synaptic):
                spk1, syn1, mem1 = self.n1(h, syn1, mem1)
            else:
                spk1, mem1 = self.n1(h, mem1)

            h = self.fc2(spk1)

            # Output layer
            if isinstance(self.n2, snn.RLeaky):
                spk2, mem2 = self.n2(h, spk2_prev, mem2)
                spk2_prev = spk2
            elif isinstance(self.n2, snn.RSynaptic):
                spk2, syn2, mem2 = self.n2(h, spk2_prev, syn2, mem2)
                spk2_prev = spk2
            elif isinstance(self.n2, snn.Synaptic):
                spk2, syn2, mem2 = self.n2(h, syn2, mem2)
            else:
                spk2, mem2 = self.n2(h, mem2)

            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec), torch.stack(mem_rec)


# ============================================================
# Loss functions (mirroring mlx-snn)
# ============================================================

def ce_rate_loss(spk_out, targets):
    """CE on mean spike rate."""
    spike_rate = spk_out.mean(dim=0)  # [batch, 10]
    return nn.functional.cross_entropy(spike_rate, targets)


def ce_count_loss(spk_out, targets):
    """CE on spike count."""
    spike_count = spk_out.sum(dim=0)  # [batch, 10]
    return nn.functional.cross_entropy(spike_count, targets)


def mse_membrane_loss(mem_out, targets, on_target=1.0, off_target=0.0):
    """MSE on last-timestep membrane with one-hot targets."""
    mem_last = mem_out[-1]  # [batch, 10]
    num_classes = mem_last.shape[-1]
    one_hot = torch.zeros(targets.shape[0], num_classes, device=targets.device)
    one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
    target_vals = one_hot * on_target + (1.0 - one_hot) * off_target
    return ((mem_last - target_vals) ** 2).mean()


# ============================================================
# Encoding functions
# ============================================================

def rate_encode(x, num_steps):
    """Poisson rate encoding."""
    return torch.stack([torch.bernoulli(x) for _ in range(num_steps)])


def latency_encode(x, num_steps):
    """Latency encoding: spike time = (1 - x) * num_steps."""
    spike_times = ((1.0 - x.clamp(1e-6, 1.0)) * (num_steps - 1)).long()
    spikes = torch.zeros(num_steps, *x.shape, device=x.device)
    for t in range(num_steps):
        spikes[t] = (spike_times == t).float()
    return spikes


def delta_encode(x, num_steps, threshold=0.1):
    """Delta modulation on static images with noise."""
    x_tiled = x.unsqueeze(0).expand(num_steps, -1, -1)
    noise = torch.randn_like(x_tiled) * 0.05
    x_noisy = (x_tiled + noise).clamp(0, 1)
    diff = torch.zeros_like(x_noisy)
    diff[1:] = x_noisy[1:] - x_noisy[:-1]
    return (diff.abs() > threshold).float()


# ============================================================
# Training / evaluation
# ============================================================

LOSS_FNS = {
    "ce_rate_loss": lambda spk, mem, t: ce_rate_loss(spk, t),
    "ce_count_loss": lambda spk, mem, t: ce_count_loss(spk, t),
    "mse_membrane_loss": lambda spk, mem, t: mse_membrane_loss(mem, t),
}


def train_one_epoch(model, optimizer, loss_fn_name, train_loader, encode_fn, seed_offset):
    model.train()
    total_loss = 0.0
    n_batches = 0
    torch.manual_seed(seed_offset)

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        spk_in = encode_fn(xb, NUM_STEPS).to(DEVICE)

        optimizer.zero_grad()
        spk_out, mem_out = model(spk_in)
        loss = LOSS_FNS[loss_fn_name](spk_out, mem_out, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, test_loader, encode_fn):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            spk_in = encode_fn(xb, NUM_STEPS).to(DEVICE)
            spk_out, mem_out = model(spk_in)
            preds = mem_out[-1].argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.shape[0]
    return correct / total if total > 0 else 0.0


def run_config(name, neuron_cls, neuron_kwargs, train_loader, test_loader,
               loss_fn_name="ce_rate_loss", encode_fn=None, seeds=SEEDS):
    if encode_fn is None:
        encode_fn = rate_encode

    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        nk1 = dict(neuron_kwargs)
        nk2 = dict(neuron_kwargs)
        nk2["reset_mechanism"] = "none"

        model = SNN(neuron_cls, neuron_cls, nk1, nk2).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        epoch_data = []
        t_start = time.time()
        for epoch in range(NUM_EPOCHS):
            t_ep = time.time()
            avg_loss = train_one_epoch(model, optimizer, loss_fn_name,
                                       train_loader, encode_fn, seed + epoch)
            test_acc = evaluate(model, test_loader, encode_fn)
            ep_time = time.time() - t_ep
            epoch_data.append({
                "epoch": epoch + 1, "loss": avg_loss,
                "test_acc": test_acc, "epoch_time": ep_time,
            })
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    [{name}] seed={seed} epoch={epoch+1:2d}  "
                      f"loss={avg_loss:.4f}  acc={test_acc:.4f}  "
                      f"t={ep_time:.1f}s", flush=True)

        total_time = time.time() - t_start
        final_acc = epoch_data[-1]["test_acc"]
        results.append({
            "seed": seed, "final_acc": final_acc,
            "total_time": total_time,
            "time_per_epoch": total_time / NUM_EPOCHS,
            "epochs": epoch_data,
        })
        print(f"    [{name}] seed={seed} DONE — acc={final_acc:.4f} "
              f"time={total_time:.1f}s", flush=True)

    return results


def summarize(results):
    accs = [r["final_acc"] for r in results]
    times = [r["time_per_epoch"] for r in results]
    return {
        "acc_mean": np.mean(accs),
        "acc_std": np.std(accs),
        "time_mean": np.mean(times),
        "time_std": np.std(times),
    }


def save_curves(name, results, out_dir=OUT_DIR):
    path = os.path.join(out_dir, f"curves_{name}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "epoch", "loss", "test_acc", "epoch_time"])
        for r in results:
            for ed in r["epochs"]:
                writer.writerow([r["seed"], ed["epoch"], f"{ed['loss']:.6f}",
                                 f"{ed['test_acc']:.6f}", f"{ed['epoch_time']:.3f}"])


def print_table(title, rows):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    print(f"  {header_line}")
    print(f"  {'-' * len(header_line)}")
    for r in rows:
        line = " | ".join(str(r[h]).ljust(widths[h]) for h in headers)
        print(f"  {line}")
    print()


def print_latex_table(title, rows):
    headers = list(rows[0].keys())
    print(f"\n% {title}")
    print(r"\begin{tabular}{" + "l" * len(headers) + "}")
    print(r"\toprule")
    print(" & ".join(headers) + r" \\")
    print(r"\midrule")
    for r in rows:
        print(" & ".join(str(r[h]) for h in headers) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ============================================================
# Experiments
# ============================================================

def get_surrogate_fn(name):
    """Map surrogate name to snnTorch surrogate."""
    fns = {
        "fast_sigmoid": surrogate.fast_sigmoid(slope=25),
        "arctan": surrogate.atan(alpha=2.0),
        "sigmoid": surrogate.sigmoid(slope=25),
        "triangular": surrogate.triangular(threshold=1.0),
        "straight_through": surrogate.straight_through_estimator(),
    }
    return fns.get(name, surrogate.fast_sigmoid(slope=25))


def experiment_1(train_loader, test_loader):
    """Neuron type comparison."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Neuron Type Comparison (snnTorch)")
    print("=" * 70, flush=True)

    configs = {
        "Leaky": (snn.Leaky, {"beta": 0.9}),
        "RLeaky": (snn.RLeaky, {"beta": 0.9, "V": 1.0, "all_to_all": False}),
        "Synaptic": (snn.Synaptic, {"alpha": 0.8, "beta": 0.9}),
        "RSynaptic": (snn.RSynaptic, {"alpha": 0.8, "beta": 0.9, "V": 1.0, "all_to_all": False}),
    }

    all_results = {}
    table_rows = []
    for name, (cls, kwargs) in configs.items():
        print(f"\n  --- {name} ---", flush=True)
        results = run_config(name, cls, kwargs, train_loader, test_loader)
        all_results[name] = results
        save_curves(f"exp1_{name}", results)
        s = summarize(results)
        table_rows.append({
            "Neuron": name,
            "Accuracy": f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}",
            "Time/epoch (s)": f"{s['time_mean']:.1f} ± {s['time_std']:.1f}",
        })

    print_table("Experiment 1: Neuron Type Comparison (snnTorch)", table_rows)
    print_latex_table("Experiment 1: Neuron Type Comparison (snnTorch)", table_rows)
    return all_results


def experiment_2(train_loader, test_loader):
    """Learnable parameter ablation."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Learnable Parameter Ablation (snnTorch)")
    print("=" * 70, flush=True)

    configs = {
        "baseline": {"beta": 0.9, "learn_beta": False, "learn_threshold": False},
        "learn_beta": {"beta": 0.9, "learn_beta": True, "learn_threshold": False},
        "learn_thresh": {"beta": 0.9, "learn_beta": False, "learn_threshold": True},
        "learn_both": {"beta": 0.9, "learn_beta": True, "learn_threshold": True},
    }

    all_results = {}
    table_rows = []
    for name, kwargs in configs.items():
        print(f"\n  --- {name} ---", flush=True)
        results = run_config(name, snn.Leaky, kwargs, train_loader, test_loader)
        all_results[name] = results
        save_curves(f"exp2_{name}", results)
        s = summarize(results)
        table_rows.append({
            "Config": name,
            "learn_beta": str(kwargs["learn_beta"]),
            "learn_thresh": str(kwargs["learn_threshold"]),
            "Accuracy": f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}",
            "Time/epoch (s)": f"{s['time_mean']:.1f} ± {s['time_std']:.1f}",
        })

    print_table("Experiment 2: Learnable Parameter Ablation (snnTorch)", table_rows)
    print_latex_table("Experiment 2: Learnable Parameter Ablation (snnTorch)", table_rows)
    return all_results


def experiment_3(train_loader, test_loader):
    """Surrogate gradient comparison."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Surrogate Gradient Comparison (snnTorch)")
    print("=" * 70, flush=True)

    surrogates = ["fast_sigmoid", "arctan", "sigmoid", "triangular", "straight_through"]

    all_results = {}
    table_rows = []
    for sg in surrogates:
        print(f"\n  --- {sg} ---", flush=True)
        kwargs = {"beta": 0.9, "spike_grad": get_surrogate_fn(sg)}
        results = run_config(sg, snn.Leaky, kwargs, train_loader, test_loader)
        all_results[sg] = results
        save_curves(f"exp3_{sg}", results)
        s = summarize(results)
        table_rows.append({
            "Surrogate": sg,
            "Accuracy": f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}",
            "Time/epoch (s)": f"{s['time_mean']:.1f} ± {s['time_std']:.1f}",
        })

    print_table("Experiment 3: Surrogate Gradient Comparison (snnTorch)", table_rows)
    print_latex_table("Experiment 3: Surrogate Gradient Comparison (snnTorch)", table_rows)
    return all_results


def experiment_4(train_loader, test_loader):
    """Loss function comparison."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Loss Function Comparison (snnTorch)")
    print("=" * 70, flush=True)

    losses = ["ce_rate_loss", "ce_count_loss", "mse_membrane_loss"]

    all_results = {}
    table_rows = []
    for loss_name in losses:
        print(f"\n  --- {loss_name} ---", flush=True)
        kwargs = {"beta": 0.9}
        results = run_config(loss_name, snn.Leaky, kwargs, train_loader, test_loader,
                             loss_fn_name=loss_name)
        all_results[loss_name] = results
        save_curves(f"exp4_{loss_name}", results)
        s = summarize(results)
        table_rows.append({
            "Loss": loss_name,
            "Accuracy": f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}",
            "Time/epoch (s)": f"{s['time_mean']:.1f} ± {s['time_std']:.1f}",
        })

    print_table("Experiment 4: Loss Function Comparison (snnTorch)", table_rows)
    print_latex_table("Experiment 4: Loss Function Comparison (snnTorch)", table_rows)
    return all_results


def experiment_5(train_loader, test_loader):
    """Encoding comparison."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: Encoding Comparison (snnTorch)")
    print("=" * 70, flush=True)

    encodings = {
        "rate": rate_encode,
        "latency": latency_encode,
        "delta": lambda x, ns: delta_encode(x, ns, threshold=0.1),
    }

    all_results = {}
    table_rows = []
    for enc_name, enc_fn in encodings.items():
        print(f"\n  --- {enc_name} ---", flush=True)
        kwargs = {"beta": 0.9}
        results = run_config(enc_name, snn.Leaky, kwargs, train_loader, test_loader,
                             encode_fn=enc_fn)
        all_results[enc_name] = results
        save_curves(f"exp5_{enc_name}", results)
        s = summarize(results)
        table_rows.append({
            "Encoding": enc_name,
            "Accuracy": f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}",
            "Time/epoch (s)": f"{s['time_mean']:.1f} ± {s['time_std']:.1f}",
        })

    print_table("Experiment 5: Encoding Comparison (snnTorch)", table_rows)
    print_latex_table("Experiment 5: Encoding Comparison (snnTorch)", table_rows)
    return all_results


# ============================================================
# Main
# ============================================================

def main():
    print(f"snnTorch Baseline Experiments (device={DEVICE})")
    print(f"Config: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, "
          f"{len(SEEDS)} seeds, {NUM_STEPS} timesteps")
    print(f"Output: {OUT_DIR}", flush=True)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    print("\nLoading MNIST...", flush=True)
    train_ds, test_ds = load_mnist()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}", flush=True)

    t_total = time.time()

    r1 = experiment_1(train_loader, test_loader)
    r2 = experiment_2(train_loader, test_loader)
    r3 = experiment_3(train_loader, test_loader)
    r4 = experiment_4(train_loader, test_loader)
    r5 = experiment_5(train_loader, test_loader)

    total_time = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  Total experiment time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
