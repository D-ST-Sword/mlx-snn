"""TAC Speedup: benchmarking Temporal Aggregated Convolution vs baseline.

Demonstrates the TAC operator's speedup over standard per-timestep
convolution. TAC aggregates K timesteps into a single conv call,
achieving near-linear speedup with minimal accuracy loss.

Usage:
    python examples/03_tac_speedup.py

Requirements:
    pip install mlx-snn
"""

import time

import mlx.core as mx
import mlx.nn as nn

import mlxsnn
from mlxsnn.operators import TemporalAggregatedConv


# ---------------------------------------------------------------------------
# Baseline: per-timestep convolution
# ---------------------------------------------------------------------------

class BaselineConvSNN(nn.Module):
    """Standard Conv SNN: one conv call per timestep."""

    def __init__(self, in_ch=2, out_ch=32, kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=1)
        self.bn = nn.BatchNorm(out_ch)
        self.lif = mlxsnn.Leaky(beta=0.9)

    def __call__(self, x_seq):
        """x_seq: (T, B, H, W, C)."""
        T = x_seq.shape[0]
        B = x_seq.shape[1]
        state = self.lif.init_state(B, 32 * 32 * 32)
        spk_list = []
        for t in range(T):
            out = self.conv(x_seq[t])
            out = self.bn(out)
            out_flat = out.reshape(B, -1)
            spk, state = self.lif(out_flat, state)
            spk_list.append(spk)
        return mx.stack(spk_list)


# ---------------------------------------------------------------------------
# TAC: aggregated convolution
# ---------------------------------------------------------------------------

class TACConvSNN(nn.Module):
    """TAC Conv SNN: one conv call per K timesteps."""

    def __init__(self, in_ch=2, out_ch=32, kernel=3, chunk_size=4):
        super().__init__()
        self.tac = TemporalAggregatedConv(
            in_channels=in_ch, out_channels=out_ch, kernel_size=kernel,
            chunk_size=chunk_size, padding=1, bn=True,
        )

    def __call__(self, x_seq):
        """x_seq: (T, B, H, W, C)."""
        B = x_seq.shape[1]
        state = self.tac.init_state(B, (32, 32))
        spk_seq, _ = self.tac(x_seq, state)
        return spk_seq


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(model, x_seq, name, warmup=3, runs=10):
    """Time forward pass with warmup."""
    # Warmup
    for _ in range(warmup):
        out = model(x_seq)
        mx.eval(out)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        out = model(x_seq)
        mx.eval(out)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"  {name:30s}  {avg*1000:7.1f} ms  (output shape: {out.shape})")
    return avg


def main():
    print("TAC Speedup Benchmark")
    print("=" * 60)

    T, B, H, W, C = 16, 8, 32, 32, 2
    x_seq = mx.random.normal((T, B, H, W, C))
    mx.eval(x_seq)

    print(f"\nInput: T={T}, B={B}, H={H}, W={W}, C={C}")
    print(f"{'':30s}  {'Time':>7s}  Output")
    print("-" * 60)

    # Baseline
    baseline = BaselineConvSNN(in_ch=C, out_ch=32)
    t_base = benchmark(baseline, x_seq, "Baseline (per-timestep)")

    # TAC with different chunk sizes
    for K in [2, 4, 8, 16]:
        if T % K == 0:
            tac_model = TACConvSNN(in_ch=C, out_ch=32, chunk_size=K)
            t_tac = benchmark(tac_model, x_seq, f"TAC (K={K})")
            speedup = t_base / t_tac if t_tac > 0 else float("inf")
            print(f"  {'':30s}  Speedup: {speedup:.1f}x")

    print("\nNote: TAC reduces conv calls from T to T/K while")
    print("preserving gradient flow through surrogate gradients.")


if __name__ == "__main__":
    main()
