"""v0.4.0 benchmark experiments for arXiv v2.

Runs 5 experiments on MNIST with 3 seeds each:
1. Neuron type comparison (Leaky vs RLeaky vs Synaptic vs RSynaptic)
2. Learnable parameter ablation (learn_beta, learn_threshold)
3. Surrogate gradient comparison (6 surrogates)
4. Loss function comparison (ce_rate, ce_count, mse_membrane)
5. Encoding comparison (rate, latency, delta)

Base architecture: Linear(784,128) → Neuron → Linear(128,10) → Neuron
Batch size 128, 25 epochs, lr=1e-3, Adam, 25 timesteps.

Usage:
    python experiments/v04_benchmarks.py
"""

import os
import sys
import csv
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlxsnn

# ============================================================
# Config
# ============================================================

NUM_STEPS = 25
BATCH_SIZE = 128
NUM_EPOCHS = 25
LR = 1e-3
SEEDS = [42, 123, 7]
HIDDEN = 128

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v04")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Data loading (reused from quickstart)
# ============================================================

def load_mnist():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        x = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int32)
        return x[:60000], y[:60000], x[60000:], y[60000:]
    except ImportError:
        pass

    import gzip
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

    def load_images(fname):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url_base + fname, path)
        with gzip.open(path, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0

    def load_labels(fname):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url_base + fname, path)
        with gzip.open(path, "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int32)

    return (load_images(files["train_images"]), load_labels(files["train_labels"]),
            load_images(files["test_images"]), load_labels(files["test_labels"]))


def get_batches(x, y, batch_size, shuffle=True, seed=None):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield mx.array(x[idx]), mx.array(y[idx])


# ============================================================
# Generic SNN model
# ============================================================

class SNN(nn.Module):
    def __init__(self, neuron_cls1, neuron_cls2, neuron_kwargs1, neuron_kwargs2):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.n1 = neuron_cls1(**neuron_kwargs1)
        self.fc2 = nn.Linear(HIDDEN, 10)
        self.n2 = neuron_cls2(**neuron_kwargs2)

    def __call__(self, x):
        batch_size = x.shape[1]
        state1 = self.n1.init_state(batch_size, HIDDEN)
        state2 = self.n2.init_state(batch_size, 10)
        spk_rec = []
        mem_rec = []
        for t in range(x.shape[0]):
            h = self.fc1(x[t])
            spk1, state1 = self.n1(h, state1)
            h = self.fc2(spk1)
            spk2, state2 = self.n2(h, state2)
            spk_rec.append(spk2)
            mem_rec.append(state2["mem"])
        return mx.stack(spk_rec), mx.stack(mem_rec)


# ============================================================
# Training / evaluation
# ============================================================

def train_one_epoch(model, optimizer, loss_fn_name, x_train, y_train,
                    encode_fn, seed_offset):
    loss_and_grad = nn.value_and_grad(model, _make_loss_fn(loss_fn_name))
    total_loss = 0.0
    n_batches = 0
    for xb, yb in get_batches(x_train, y_train, BATCH_SIZE, seed=seed_offset):
        spk_in = encode_fn(xb)
        loss, grads = loss_and_grad(model, spk_in, yb)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def evaluate(model, x_test, y_test, encode_fn):
    correct = 0
    total = 0
    for xb, yb in get_batches(x_test, y_test, BATCH_SIZE, shuffle=False):
        spk_in = encode_fn(xb)
        spk_out, mem_out = model(spk_in)
        mx.eval(mem_out)
        preds = mx.argmax(mem_out[-1], axis=1)
        mx.eval(preds)
        correct += mx.sum(preds == yb).item()
        total += yb.shape[0]
    return correct / total if total > 0 else 0.0


def _make_loss_fn(name):
    def loss_fn(model, spk_in, targets):
        spk_out, mem_out = model(spk_in)
        if name == "ce_rate_loss":
            return mlxsnn.ce_rate_loss(spk_out, targets)
        elif name == "ce_count_loss":
            return mlxsnn.ce_count_loss(spk_out, targets)
        elif name == "mse_membrane_loss":
            return mlxsnn.mse_membrane_loss(mem_out[-1], targets)
        elif name == "membrane_loss":
            return mlxsnn.membrane_loss(mem_out, targets)
        else:
            raise ValueError(f"Unknown loss: {name}")
    return loss_fn


def rate_encode_fn(x):
    return mlxsnn.rate_encode(x, num_steps=NUM_STEPS)


def latency_encode_fn(x):
    return mlxsnn.latency_encode(x, num_steps=NUM_STEPS)


def delta_encode_fn(x):
    # delta_encode expects [T, B, F], so we first tile the static image
    # across time then compute temporal differences
    x_tiled = mx.broadcast_to(mx.expand_dims(x, 0), (NUM_STEPS, *x.shape))
    # Add small noise per timestep to create temporal variation
    noise = mx.random.normal(x_tiled.shape) * 0.05
    x_noisy = mx.clip(x_tiled + noise, 0.0, 1.0)
    return mlxsnn.delta_encode(x_noisy, threshold=0.1)


def run_config(name, neuron_cls, neuron_kwargs, x_train, y_train, x_test, y_test,
               loss_fn_name="ce_rate_loss", encode_fn=None, seeds=SEEDS):
    """Run a single config across multiple seeds. Returns list of per-seed results."""
    if encode_fn is None:
        encode_fn = rate_encode_fn

    results = []
    for seed in seeds:
        np.random.seed(seed)
        mx.random.seed(seed)

        # Output neuron: same type but with reset_mechanism="none"
        nk1 = dict(neuron_kwargs)
        nk2 = dict(neuron_kwargs)
        nk2["reset_mechanism"] = "none"

        model = SNN(neuron_cls, neuron_cls, nk1, nk2)
        mx.eval(model.parameters())
        optimizer = optim.Adam(learning_rate=LR)

        epoch_data = []
        t_start = time.time()
        for epoch in range(NUM_EPOCHS):
            t_ep = time.time()
            avg_loss = train_one_epoch(model, optimizer, loss_fn_name,
                                       x_train, y_train, encode_fn,
                                       seed_offset=seed + epoch)
            test_acc = evaluate(model, x_test, y_test, encode_fn)
            ep_time = time.time() - t_ep
            epoch_data.append({
                "epoch": epoch + 1, "loss": avg_loss,
                "test_acc": test_acc, "epoch_time": ep_time,
            })
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    [{name}] seed={seed} epoch={epoch+1:2d}  "
                      f"loss={avg_loss:.4f}  acc={test_acc:.4f}  "
                      f"t={ep_time:.1f}s")

        total_time = time.time() - t_start
        final_acc = epoch_data[-1]["test_acc"]
        results.append({
            "seed": seed, "final_acc": final_acc,
            "total_time": total_time,
            "time_per_epoch": total_time / NUM_EPOCHS,
            "epochs": epoch_data,
        })
        print(f"    [{name}] seed={seed} DONE — acc={final_acc:.4f} "
              f"time={total_time:.1f}s")

    return results


def summarize(results):
    """Compute mean ± std of final accuracy and timing."""
    accs = [r["final_acc"] for r in results]
    times = [r["time_per_epoch"] for r in results]
    return {
        "acc_mean": np.mean(accs),
        "acc_std": np.std(accs),
        "time_mean": np.mean(times),
        "time_std": np.std(times),
    }


def save_curves(name, results, out_dir=OUT_DIR):
    """Save per-epoch training curves as CSV."""
    path = os.path.join(out_dir, f"curves_{name}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "epoch", "loss", "test_acc", "epoch_time"])
        for r in results:
            for ed in r["epochs"]:
                writer.writerow([r["seed"], ed["epoch"], f"{ed['loss']:.6f}",
                                 f"{ed['test_acc']:.6f}", f"{ed['epoch_time']:.3f}"])


def print_table(title, rows):
    """Print a formatted table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    # Find max widths
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
    """Print LaTeX table."""
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

def experiment_1(x_train, y_train, x_test, y_test):
    """Neuron type comparison."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: Neuron Type Comparison")
    print("=" * 70)

    configs = {
        "Leaky": (mlxsnn.Leaky, {"beta": 0.9}),
        "RLeaky": (mlxsnn.RLeaky, {"beta": 0.9}),
        "Synaptic": (mlxsnn.Synaptic, {"alpha": 0.8, "beta": 0.9}),
        "RSynaptic": (mlxsnn.RSynaptic, {"alpha": 0.8, "beta": 0.9}),
    }

    all_results = {}
    table_rows = []
    for name, (cls, kwargs) in configs.items():
        print(f"\n  --- {name} ---")
        results = run_config(name, cls, kwargs, x_train, y_train, x_test, y_test)
        all_results[name] = results
        save_curves(f"exp1_{name}", results)
        s = summarize(results)
        table_rows.append({
            "Neuron": name,
            "Accuracy": f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}",
            "Time/epoch (s)": f"{s['time_mean']:.1f} ± {s['time_std']:.1f}",
        })

    print_table("Experiment 1: Neuron Type Comparison", table_rows)
    print_latex_table("Experiment 1: Neuron Type Comparison", table_rows)
    return all_results


def experiment_2(x_train, y_train, x_test, y_test):
    """Learnable parameter ablation."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Learnable Parameter Ablation")
    print("=" * 70)

    configs = {
        "baseline": {"beta": 0.9, "learn_beta": False, "learn_threshold": False},
        "learn_beta": {"beta": 0.9, "learn_beta": True, "learn_threshold": False},
        "learn_thresh": {"beta": 0.9, "learn_beta": False, "learn_threshold": True},
        "learn_both": {"beta": 0.9, "learn_beta": True, "learn_threshold": True},
    }

    all_results = {}
    table_rows = []
    for name, kwargs in configs.items():
        print(f"\n  --- {name} ---")
        results = run_config(name, mlxsnn.Leaky, kwargs,
                             x_train, y_train, x_test, y_test)
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

    print_table("Experiment 2: Learnable Parameter Ablation", table_rows)
    print_latex_table("Experiment 2: Learnable Parameter Ablation", table_rows)
    return all_results


def experiment_3(x_train, y_train, x_test, y_test):
    """Surrogate gradient comparison."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Surrogate Gradient Comparison")
    print("=" * 70)

    surrogates = ["fast_sigmoid", "arctan", "sigmoid", "triangular", "straight_through"]

    all_results = {}
    table_rows = []
    for sg in surrogates:
        print(f"\n  --- {sg} ---")
        kwargs = {"beta": 0.9, "surrogate_fn": sg}
        results = run_config(sg, mlxsnn.Leaky, kwargs,
                             x_train, y_train, x_test, y_test)
        all_results[sg] = results
        save_curves(f"exp3_{sg}", results)
        s = summarize(results)
        table_rows.append({
            "Surrogate": sg,
            "Accuracy": f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}",
            "Time/epoch (s)": f"{s['time_mean']:.1f} ± {s['time_std']:.1f}",
        })

    print_table("Experiment 3: Surrogate Gradient Comparison", table_rows)
    print_latex_table("Experiment 3: Surrogate Gradient Comparison", table_rows)
    return all_results


def experiment_4(x_train, y_train, x_test, y_test):
    """Loss function comparison."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Loss Function Comparison")
    print("=" * 70)

    losses = ["ce_rate_loss", "ce_count_loss", "mse_membrane_loss"]

    all_results = {}
    table_rows = []
    for loss_name in losses:
        print(f"\n  --- {loss_name} ---")
        kwargs = {"beta": 0.9}
        results = run_config(loss_name, mlxsnn.Leaky, kwargs,
                             x_train, y_train, x_test, y_test,
                             loss_fn_name=loss_name)
        all_results[loss_name] = results
        save_curves(f"exp4_{loss_name}", results)
        s = summarize(results)
        table_rows.append({
            "Loss": loss_name,
            "Accuracy": f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}",
            "Time/epoch (s)": f"{s['time_mean']:.1f} ± {s['time_std']:.1f}",
        })

    print_table("Experiment 4: Loss Function Comparison", table_rows)
    print_latex_table("Experiment 4: Loss Function Comparison", table_rows)
    return all_results


def experiment_5(x_train, y_train, x_test, y_test):
    """Encoding comparison."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: Encoding Comparison")
    print("=" * 70)

    encodings = {
        "rate": rate_encode_fn,
        "latency": latency_encode_fn,
        "delta": delta_encode_fn,
    }

    all_results = {}
    table_rows = []
    for enc_name, enc_fn in encodings.items():
        print(f"\n  --- {enc_name} ---")
        kwargs = {"beta": 0.9}
        results = run_config(enc_name, mlxsnn.Leaky, kwargs,
                             x_train, y_train, x_test, y_test,
                             encode_fn=enc_fn)
        all_results[enc_name] = results
        save_curves(f"exp5_{enc_name}", results)
        s = summarize(results)
        table_rows.append({
            "Encoding": enc_name,
            "Accuracy": f"{s['acc_mean']:.4f} ± {s['acc_std']:.4f}",
            "Time/epoch (s)": f"{s['time_mean']:.1f} ± {s['time_std']:.1f}",
        })

    print_table("Experiment 5: Encoding Comparison", table_rows)
    print_latex_table("Experiment 5: Encoding Comparison", table_rows)
    return all_results


# ============================================================
# Main
# ============================================================

def main():
    print("mlx-snn v0.4.0 Benchmark Experiments")
    print(f"Config: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}, "
          f"{len(SEEDS)} seeds, {NUM_STEPS} timesteps")
    print(f"Output: {OUT_DIR}")

    print("\nLoading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}")

    t_total = time.time()

    r1 = experiment_1(x_train, y_train, x_test, y_test)
    r2 = experiment_2(x_train, y_train, x_test, y_test)
    r3 = experiment_3(x_train, y_train, x_test, y_test)
    r4 = experiment_4(x_train, y_train, x_test, y_test)
    r5 = experiment_5(x_train, y_train, x_test, y_test)

    total_time = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  Total experiment time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
