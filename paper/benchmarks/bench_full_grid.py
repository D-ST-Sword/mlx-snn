"""Full grid benchmark: 5 configs x 3 backends = 15 runs.

Output: LaTeX table + JSON results.
"""

import sys
sys.path.insert(0, ".")

import json
import time
import numpy as np

# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS = [
    {"name": "C1", "beta": 0.85, "hidden": 256, "lr": 1e-3, "epochs": 25, "bs": 128},
    {"name": "C2", "beta": 0.9,  "hidden": 256, "lr": 1e-3, "epochs": 25, "bs": 128},
    {"name": "C3", "beta": 0.9,  "hidden": 256, "lr": 1e-3, "epochs": 25, "bs": 256},
    {"name": "C4", "beta": 0.9,  "hidden": 128, "lr": 1e-3, "epochs": 25, "bs": 128},
    {"name": "C5", "beta": 0.95, "hidden": 128, "lr": 2e-3, "epochs": 15, "bs": 128},
]

NUM_STEPS = 25

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
    fnames = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
              "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    data_dir = os.path.join(os.path.dirname(__file__), ".mnist_data")
    os.makedirs(data_dir, exist_ok=True)
    def dl(fname, offset, shape=None):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url_base + fname, path)
        with gzip.open(path, "rb") as f:
            d = np.frombuffer(f.read(), np.uint8, offset=offset)
        return d.reshape(shape).astype(np.float32) / 255.0 if shape else d.astype(np.int32)
    return (dl(fnames[0], 16, (-1, 784)), dl(fnames[1], 8),
            dl(fnames[2], 16, (-1, 784)), dl(fnames[3], 8))


def get_batches_np(x, y, bs, shuffle=True):
    idx = np.arange(len(x))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(x), bs):
        j = idx[i:i+bs]
        yield x[j], y[j]


# ---------------------------------------------------------------------------
# mlx-snn backend
# ---------------------------------------------------------------------------

def run_mlxsnn(cfg, x_train, y_train, x_test, y_test):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlxsnn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, cfg["hidden"])
            self.lif1 = mlxsnn.Leaky(beta=cfg["beta"], threshold=1.0)
            self.fc2 = nn.Linear(cfg["hidden"], 10)
            self.lif2 = mlxsnn.Leaky(beta=cfg["beta"], threshold=1.0,
                                      reset_mechanism="none")
        def __call__(self, x):
            bs = x.shape[1]
            s1 = self.lif1.init_state(bs, cfg["hidden"])
            s2 = self.lif2.init_state(bs, 10)
            for t in range(NUM_STEPS):
                sp1, s1 = self.lif1(self.fc1(x[t]), s1)
                sp2, s2 = self.lif2(self.fc2(sp1), s2)
            return s2["mem"]

    try:
        mx.reset_peak_memory()
    except Exception:
        pass

    model = Model()
    optimizer = optim.Adam(learning_rate=cfg["lr"])
    def loss_fn(m, s, t):
        return mx.mean(nn.losses.cross_entropy(m(s), t))
    lg = nn.value_and_grad(model, loss_fn)

    epoch_times = []
    best_acc = 0
    final_acc = 0
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        for xb, yb in get_batches_np(x_train, y_train, cfg["bs"]):
            sp = mlxsnn.rate_encode(mx.array(xb), num_steps=NUM_STEPS)
            loss, grads = lg(model, sp, mx.array(yb))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
        et = time.time() - t0
        epoch_times.append(et)

        correct, total = 0, 0
        for xb, yb in get_batches_np(x_test, y_test, cfg["bs"], shuffle=False):
            mem = model(mlxsnn.rate_encode(mx.array(xb), num_steps=NUM_STEPS))
            mx.eval(mem)
            preds = mx.argmax(mem, axis=1)
            mx.eval(preds)
            correct += mx.sum(preds == mx.array(yb)).item()
            total += len(yb)
        acc = correct / total
        best_acc = max(best_acc, acc)
        final_acc = acc

    try:
        peak_mb = mx.get_peak_memory() / (1024 * 1024)
    except Exception:
        try:
            peak_mb = mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            peak_mb = -1

    return {
        "best_acc": best_acc, "final_acc": final_acc,
        "total_time": sum(epoch_times),
        "avg_epoch": np.mean(epoch_times),
        "peak_mb": peak_mb,
    }


# ---------------------------------------------------------------------------
# snnTorch backends
# ---------------------------------------------------------------------------

def run_snntorch(cfg, x_train, y_train, x_test, y_test, device_name):
    import torch
    import torch.nn as tnn
    import snntorch as snn
    from snntorch import surrogate

    device = torch.device(device_name)

    class Model(tnn.Module):
        def __init__(self):
            super().__init__()
            sg = surrogate.fast_sigmoid(slope=25)
            self.fc1 = tnn.Linear(784, cfg["hidden"])
            self.lif1 = snn.Leaky(beta=cfg["beta"], spike_grad=sg)
            self.fc2 = tnn.Linear(cfg["hidden"], 10)
            self.lif2 = snn.Leaky(beta=cfg["beta"], spike_grad=sg,
                                  reset_mechanism="none")
        def forward(self, x):
            m1 = self.lif1.init_leaky()
            m2 = self.lif2.init_leaky()
            for t in range(NUM_STEPS):
                sp1, m1 = self.lif1(self.fc1(x[t]), m1)
                sp2, m2 = self.lif2(self.fc2(sp1), m2)
            return m2

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = tnn.CrossEntropyLoss()

    # Warmup for MPS
    if device_name == "mps":
        dummy = torch.randn(NUM_STEPS, 2, 784, device=device)
        _ = model(dummy)
        torch.mps.synchronize()

    epoch_times = []
    best_acc = 0
    final_acc = 0
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        for xb, yb in get_batches_np(x_train, y_train, cfg["bs"]):
            xb_t = torch.tensor(xb, device=device)
            yb_t = torch.tensor(yb, dtype=torch.long, device=device)
            sp = torch.bernoulli(xb_t.unsqueeze(0).repeat(NUM_STEPS, 1, 1))
            optimizer.zero_grad()
            loss = criterion(model(sp), yb_t)
            loss.backward()
            optimizer.step()
        if device_name == "mps":
            torch.mps.synchronize()
        et = time.time() - t0
        epoch_times.append(et)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in get_batches_np(x_test, y_test, cfg["bs"], shuffle=False):
                xb_t = torch.tensor(xb, device=device)
                yb_t = torch.tensor(yb, dtype=torch.long, device=device)
                sp = torch.bernoulli(xb_t.unsqueeze(0).repeat(NUM_STEPS, 1, 1))
                preds = model(sp).argmax(dim=1)
                correct += (preds == yb_t).sum().item()
                total += len(yb)
        acc = correct / total
        best_acc = max(best_acc, acc)
        final_acc = acc

    peak_mb = -1
    if device_name == "mps":
        try:
            peak_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
        except Exception:
            pass

    return {
        "best_acc": best_acc, "final_acc": final_acc,
        "total_time": sum(epoch_times),
        "avg_epoch": np.mean(epoch_times),
        "peak_mb": peak_mb,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading MNIST...")
    x_train, y_train, x_test, y_test = load_mnist()
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}\n")

    all_results = {}

    for ci, cfg in enumerate(CONFIGS):
        label = (f"{cfg['name']}: β={cfg['beta']}, h={cfg['hidden']}, "
                 f"lr={cfg['lr']}, ep={cfg['epochs']}, bs={cfg['bs']}")
        print(f"\n{'#'*70}")
        print(f"  CONFIG {ci+1}/5: {label}")
        print(f"{'#'*70}")

        key = cfg["name"]
        all_results[key] = {}

        # mlx-snn
        print(f"\n  [mlx-snn]")
        r = run_mlxsnn(cfg, x_train, y_train, x_test, y_test)
        all_results[key]["mlx"] = r
        print(f"    best={r['best_acc']:.2%}  final={r['final_acc']:.2%}  "
              f"time={r['total_time']:.1f}s  epoch={r['avg_epoch']:.1f}s  "
              f"mem={r['peak_mb']:.0f}MB")

        # snnTorch MPS
        print(f"\n  [snnTorch MPS]")
        r = run_snntorch(cfg, x_train, y_train, x_test, y_test, "mps")
        all_results[key]["mps"] = r
        print(f"    best={r['best_acc']:.2%}  final={r['final_acc']:.2%}  "
              f"time={r['total_time']:.1f}s  epoch={r['avg_epoch']:.1f}s  "
              f"mem={r['peak_mb']:.0f}MB")

        # snnTorch CPU
        print(f"\n  [snnTorch CPU]")
        r = run_snntorch(cfg, x_train, y_train, x_test, y_test, "cpu")
        all_results[key]["cpu"] = r
        print(f"    best={r['best_acc']:.2%}  final={r['final_acc']:.2%}  "
              f"time={r['total_time']:.1f}s  epoch={r['avg_epoch']:.1f}s")

    # Save JSON
    with open("paper/figures/full_grid_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print LaTeX table
    print("\n\n" + "=" * 90)
    print("LATEX TABLE")
    print("=" * 90)
    print(r"""
\begin{table}[t]
\centering
\caption{Full benchmark: 5 hyperparameter configurations $\times$ 3 backends on MNIST.
All use the same 784--$h$--10 feedforward SNN architecture with $T{=}25$ timesteps and
fast sigmoid surrogate gradient.
Acc.\ = best test accuracy; Time = average seconds per epoch; Mem.\ = peak GPU memory.}
\label{tab:full_grid}
\small
\begin{tabular}{llcccccc}
\toprule
& & \multicolumn{2}{c}{\textbf{mlx-snn (MLX)}} & \multicolumn{2}{c}{\textbf{snnTorch (MPS)}} & \multicolumn{2}{c}{\textbf{snnTorch (CPU)}} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}
\textbf{Config} & \textbf{Params} & Acc.\,(\%) & Time\,(s) & Acc.\,(\%) & Time\,(s) & Acc.\,(\%) & Time\,(s) \\
\midrule""")

    for cfg in CONFIGS:
        k = cfg["name"]
        params = f"$\\beta$={cfg['beta']}, $h$={cfg['hidden']}, lr={cfg['lr']}"
        if cfg["bs"] != 128:
            params += f", bs={cfg['bs']}"
        if cfg["epochs"] != 25:
            params += f", {cfg['epochs']}ep"

        mlx_r = all_results[k]["mlx"]
        mps_r = all_results[k]["mps"]
        cpu_r = all_results[k]["cpu"]

        # Bold the best accuracy across all 3 backends for this config
        accs = [mlx_r["best_acc"], mps_r["best_acc"], cpu_r["best_acc"]]
        best_idx = int(np.argmax(accs))

        def fmt_acc(val, idx, bi):
            s = f"{val*100:.2f}"
            return f"\\textbf{{{s}}}" if idx == bi else s

        line = (f"{k} & {params} & "
                f"{fmt_acc(mlx_r['best_acc'], 0, best_idx)} & {mlx_r['avg_epoch']:.1f} & "
                f"{fmt_acc(mps_r['best_acc'], 1, best_idx)} & {mps_r['avg_epoch']:.1f} & "
                f"{fmt_acc(cpu_r['best_acc'], 2, best_idx)} & {cpu_r['avg_epoch']:.1f} \\\\")
        print(line)

    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    # Summary
    print("\n\n" + "=" * 90)
    print("SUMMARY TABLE (plain text)")
    print("=" * 90)
    header = f"{'Config':<8} {'Params':<35} {'mlx acc':>8} {'mlx s/ep':>8} {'MPS acc':>8} {'MPS s/ep':>8} {'CPU acc':>8} {'CPU s/ep':>8}"
    print(header)
    print("-" * len(header))
    for cfg in CONFIGS:
        k = cfg["name"]
        params = f"β={cfg['beta']} h={cfg['hidden']} lr={cfg['lr']}"
        if cfg["bs"] != 128:
            params += f" bs={cfg['bs']}"
        if cfg["epochs"] != 25:
            params += f" {cfg['epochs']}ep"
        m = all_results[k]["mlx"]
        p = all_results[k]["mps"]
        c = all_results[k]["cpu"]
        print(f"{k:<8} {params:<35} {m['best_acc']:>7.2%} {m['avg_epoch']:>7.1f}s "
              f"{p['best_acc']:>7.2%} {p['avg_epoch']:>7.1f}s "
              f"{c['best_acc']:>7.2%} {c['avg_epoch']:>7.1f}s")
