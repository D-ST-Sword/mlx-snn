#!/usr/bin/env python3
"""Compute 5-seed statistics by merging original, fixed, and extra seed results."""

import os
import csv
from collections import defaultdict

# Directories
BASE = os.path.dirname(os.path.abspath(__file__))
DIR_ORIG = os.path.join(BASE, "results_v04")
DIR_FIXED = os.path.join(BASE, "results_v04_fixed")
DIR_5SEEDS = os.path.join(BASE, "results_v04_5seeds")

# Mapping: config_key -> list of (directory, filename) to merge
# For configs with bugs in original, use fixed + extra
# For configs without bugs, use original + extra
CONFIG_MAP = {
    # Exp 1
    "exp1_Leaky":              [(DIR_ORIG, "curves_exp1_Leaky.csv"),
                                (DIR_5SEEDS, "curves_exp1_Leaky.csv")],
    "exp1_RLeaky_V0.5":        [(DIR_FIXED, "curves_exp1_RLeaky_V0.5_fixed.csv"),
                                (DIR_5SEEDS, "curves_exp1_RLeaky_V0.5.csv")],
    "exp1_RLeaky_V0.1_learn":  [(DIR_FIXED, "curves_exp1_RLeaky_V0.1_learn.csv"),
                                (DIR_5SEEDS, "curves_exp1_RLeaky_V0.1_learn.csv")],
    "exp1_Synaptic":           [(DIR_ORIG, "curves_exp1_Synaptic.csv"),
                                (DIR_5SEEDS, "curves_exp1_Synaptic.csv")],
    "exp1_RSynaptic_V0.5":     [(DIR_FIXED, "curves_exp1_RSynaptic_V0.5_fixed.csv"),
                                (DIR_5SEEDS, "curves_exp1_RSynaptic_V0.5.csv")],
    "exp1_RSynaptic_V0.1_learn": [(DIR_FIXED, "curves_exp1_RSynaptic_V0.1_learn.csv"),
                                  (DIR_5SEEDS, "curves_exp1_RSynaptic_V0.1_learn.csv")],
    # Exp 2
    "exp2_baseline":           [(DIR_ORIG, "curves_exp2_baseline.csv"),
                                (DIR_5SEEDS, "curves_exp2_baseline.csv")],
    "exp2_learn_beta":         [(DIR_ORIG, "curves_exp2_learn_beta.csv"),
                                (DIR_5SEEDS, "curves_exp2_learn_beta.csv")],
    "exp2_learn_thresh":       [(DIR_ORIG, "curves_exp2_learn_thresh.csv"),
                                (DIR_5SEEDS, "curves_exp2_learn_thresh.csv")],
    "exp2_learn_both":         [(DIR_ORIG, "curves_exp2_learn_both.csv"),
                                (DIR_5SEEDS, "curves_exp2_learn_both.csv")],
    # Exp 3
    "exp3_fast_sigmoid":       [(DIR_ORIG, "curves_exp3_fast_sigmoid.csv"),
                                (DIR_5SEEDS, "curves_exp3_fast_sigmoid.csv")],
    "exp3_arctan":             [(DIR_ORIG, "curves_exp3_arctan.csv"),
                                (DIR_5SEEDS, "curves_exp3_arctan.csv")],
    "exp3_sigmoid":            [(DIR_ORIG, "curves_exp3_sigmoid.csv"),
                                (DIR_5SEEDS, "curves_exp3_sigmoid.csv")],
    "exp3_triangular":         [(DIR_FIXED, "curves_exp3_triangular.csv"),
                                (DIR_5SEEDS, "curves_exp3_triangular.csv")],
    "exp3_straight_through":   [(DIR_FIXED, "curves_exp3_straight_through.csv"),
                                (DIR_5SEEDS, "curves_exp3_straight_through.csv")],
    # Exp 4
    "exp4_ce_rate_loss":       [(DIR_ORIG, "curves_exp4_ce_rate_loss.csv"),
                                (DIR_5SEEDS, "curves_exp4_ce_rate_loss.csv")],
    "exp4_ce_count_loss":      [(DIR_ORIG, "curves_exp4_ce_count_loss.csv"),
                                (DIR_5SEEDS, "curves_exp4_ce_count_loss.csv")],
    "exp4_mse_membrane_loss":  [(DIR_ORIG, "curves_exp4_mse_membrane_loss.csv"),
                                (DIR_5SEEDS, "curves_exp4_mse_membrane_loss.csv")],
    # Exp 5
    "exp5_rate":               [(DIR_ORIG, "curves_exp5_rate.csv"),
                                (DIR_5SEEDS, "curves_exp5_rate.csv")],
    "exp5_latency":            [(DIR_ORIG, "curves_exp5_latency.csv"),
                                (DIR_5SEEDS, "curves_exp5_latency.csv")],
    "exp5_delta":              [(DIR_ORIG, "curves_exp5_delta.csv"),
                                (DIR_5SEEDS, "curves_exp5_delta.csv")],
}


def load_csv(path):
    """Load CSV and return list of dicts."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def merge_data(sources):
    """Merge data from multiple (dir, filename) pairs, dedup by (seed, epoch)."""
    seen = set()
    all_rows = []
    for dirpath, filename in sources:
        path = os.path.join(dirpath, filename) if not os.path.isabs(filename) else filename
        rows = load_csv(path)
        for row in rows:
            key = (int(row["seed"]), int(row["epoch"]))
            if key not in seen:
                seen.add(key)
                all_rows.append(row)
    return all_rows


def compute_stats(rows):
    """Compute best test_acc per seed, then mean ± std across seeds."""
    if not rows:
        return None
    # Group by seed
    seed_data = defaultdict(list)
    for row in rows:
        seed_data[int(row["seed"])].append(float(row["test_acc"]))

    # Best acc per seed
    best_accs = []
    for seed in sorted(seed_data.keys()):
        best_accs.append(max(seed_data[seed]))

    n = len(best_accs)
    mean_acc = sum(best_accs) / n
    if n > 1:
        var = sum((x - mean_acc) ** 2 for x in best_accs) / (n - 1)
        std_acc = var ** 0.5
    else:
        std_acc = 0.0

    # Mean epoch time
    times = [float(row["epoch_time"]) for row in rows if "epoch_time" in row]
    mean_time = sum(times) / len(times) if times else 0.0

    return {
        "n_seeds": n,
        "seeds": sorted(seed_data.keys()),
        "best_accs": best_accs,
        "mean_acc": mean_acc * 100,
        "std_acc": std_acc * 100,
        "mean_time": mean_time,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("  5-Seed Statistics for mlx-snn v0.4 Paper")
    print("=" * 70)

    for config_key, sources in CONFIG_MAP.items():
        rows = merge_data(sources)
        stats = compute_stats(rows)
        if stats is None:
            print(f"\n{config_key}: NO DATA")
            continue

        exp = config_key.split("_")[0]
        name = "_".join(config_key.split("_")[1:])

        print(f"\n{config_key}:")
        print(f"  Seeds: {stats['seeds']} (n={stats['n_seeds']})")
        print(f"  Best accs: {[f'{a:.2f}%' for a in [x*100 for x in [a/100 for a in [stats['mean_acc']]]]]}... per seed: {[f'{a*100:.2f}%' for a in stats['best_accs']]}")
        print(f"  Mean ± Std: {stats['mean_acc']:.2f} ± {stats['std_acc']:.2f} %")
        print(f"  Mean epoch time: {stats['mean_time']:.1f} s")

    # LaTeX-ready output
    print("\n" + "=" * 70)
    print("  LaTeX-ready values")
    print("=" * 70)

    for config_key, sources in CONFIG_MAP.items():
        rows = merge_data(sources)
        stats = compute_stats(rows)
        if stats is None:
            continue
        print(f"  {config_key:35s}  ${stats['mean_acc']:.2f} \\pm {stats['std_acc']:.2f}$  ({stats['n_seeds']} seeds, {stats['mean_time']:.1f} s/ep)")
