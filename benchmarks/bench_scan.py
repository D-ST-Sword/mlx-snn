#!/usr/bin/env python3
"""Benchmark: linear scan (cumsum) vs for-loop for LIF membrane dynamics.

The LIF linear recurrence is:
    mem[t] = beta * mem[t-1] + x[t]

This can be computed without a for-loop using the identity:
    mem[t] = sum_{s=0}^{t} beta^{t-s} * x[s]
           = beta^t * cumsum(x[s] / beta^s, axis=0)

This benchmark compares:
1. for-loop: standard sequential computation
2. cumsum-scan: vectorized via mx.cumsum (no loop)
3. chunked-scan: chunked version for numerical stability at large T

Run:
    python benchmarks/bench_scan.py
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

print(f"Python: {sys.version}")
print(f"MLX version: {mx.__version__}")
print()


# ── Implementations ──────────────────────────────────────────────────


def lif_loop(x: mx.array, beta: float) -> mx.array:
    """Standard for-loop LIF membrane dynamics (linear part only)."""
    T = x.shape[0]
    mem = mx.zeros(x.shape[1:])
    mems = []
    for t in range(T):
        mem = beta * mem + x[t]
        mems.append(mem)
    return mx.stack(mems)


def lif_cumsum_scan(x: mx.array, beta: float) -> mx.array:
    """Vectorized LIF via cumsum. Numerically unstable for large T."""
    T = x.shape[0]
    # t_idx shape: [T, 1, 1, ...] for broadcasting
    extra_dims = x.ndim - 1
    shape = [T] + [1] * extra_dims
    t_idx = mx.arange(T, dtype=x.dtype).reshape(shape)

    beta_powers = beta ** t_idx           # beta^0, beta^1, ..., beta^{T-1}
    inv_beta_powers = 1.0 / beta_powers   # beta^{-t} — unstable for large T

    # x[s] / beta^s, then cumsum, then multiply by beta^t
    scaled = x * inv_beta_powers
    cumulative = mx.cumsum(scaled, axis=0)
    result = cumulative * beta_powers
    return result


def lif_chunked_scan(x: mx.array, beta: float, chunk_size: int = 64) -> mx.array:
    """Chunked cumsum scan for numerical stability at large T.

    Processes chunks of chunk_size steps via cumsum, then carries the
    last membrane value to the next chunk.
    """
    T = x.shape[0]
    all_mems = []
    carry = mx.zeros(x.shape[1:])

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = x[start:end]
        C = chunk.shape[0]

        extra_dims = chunk.ndim - 1
        shape = [C] + [1] * extra_dims
        t_idx = mx.arange(C, dtype=chunk.dtype).reshape(shape)

        beta_powers = beta ** t_idx
        inv_beta_powers = 1.0 / beta_powers

        # Include carry from previous chunk
        # mem_chunk[0] = beta * carry + x[start]
        # Adjust: carry contributes beta^{t+1} * carry to step t within chunk
        carry_contrib = carry * (beta ** (t_idx + 1))

        scaled = chunk * inv_beta_powers
        cumulative = mx.cumsum(scaled, axis=0)
        chunk_mems = cumulative * beta_powers + carry_contrib

        all_mems.append(chunk_mems)
        carry = chunk_mems[-1]

    return mx.concatenate(all_mems, axis=0)


# ── Verification ─────────────────────────────────────────────────────


def verify_correctness():
    """Verify scan implementations match the for-loop reference."""
    print("Verifying correctness...")

    for beta in [0.5, 0.9, 0.95, 0.99]:
        for T in [8, 32, 128, 256]:
            x = mx.random.normal((T, 4, 16))
            ref = lif_loop(x, beta)
            scan = lif_cumsum_scan(x, beta)
            chunked = lif_chunked_scan(x, beta, chunk_size=32)
            mx.eval(ref, scan, chunked)

            # Check cumsum scan
            if mx.allclose(ref, scan, atol=1e-4).item():
                scan_ok = "OK"
            else:
                max_diff = mx.max(mx.abs(ref - scan)).item()
                scan_ok = f"DIFF={max_diff:.2e}"

            # Check chunked scan
            if mx.allclose(ref, chunked, atol=1e-4).item():
                chunked_ok = "OK"
            else:
                max_diff = mx.max(mx.abs(ref - chunked)).item()
                chunked_ok = f"DIFF={max_diff:.2e}"

            print(
                f"  beta={beta}, T={T:>4}: "
                f"cumsum={scan_ok:<12} chunked={chunked_ok}"
            )

    # Test large T for numerical stability
    print("\n  Numerical stability at large T (beta=0.9):")
    for T in [512, 1024, 2048, 4096]:
        x = mx.random.normal((T, 4, 16)) * 0.1
        ref = lif_loop(x, 0.9)
        mx.eval(ref)
        try:
            scan = lif_cumsum_scan(x, 0.9)
            mx.eval(scan)
            if mx.any(mx.isnan(scan)).item() or mx.any(mx.isinf(scan)).item():
                scan_ok = "NaN/Inf!"
            elif mx.allclose(ref, scan, atol=1e-3).item():
                scan_ok = "OK"
            else:
                max_diff = mx.max(mx.abs(ref - scan)).item()
                scan_ok = f"DIFF={max_diff:.2e}"
        except Exception as e:
            scan_ok = f"ERROR: {e}"

        try:
            chunked = lif_chunked_scan(x, 0.9, chunk_size=64)
            mx.eval(chunked)
            if mx.any(mx.isnan(chunked)).item() or mx.any(mx.isinf(chunked)).item():
                chunked_ok = "NaN/Inf!"
            elif mx.allclose(ref, chunked, atol=1e-3).item():
                chunked_ok = "OK"
            else:
                max_diff = mx.max(mx.abs(ref - chunked)).item()
                chunked_ok = f"DIFF={max_diff:.2e}"
        except Exception as e:
            chunked_ok = f"ERROR: {e}"

        print(f"  T={T:>5}: cumsum={scan_ok:<16} chunked={chunked_ok}")


# ── Benchmark ────────────────────────────────────────────────────────


def benchmark():
    """Benchmark for-loop vs scan at various T and hidden_dim."""
    print("\nBenchmark: for-loop vs cumsum-scan vs chunked-scan")
    print("=" * 75)

    beta = 0.9
    batch_size = 32
    warmup = 3
    repeats = 10

    configs = [
        # (T, hidden_dim)
        (16, 128),
        (16, 1024),
        (64, 128),
        (64, 1024),
        (256, 128),
        (256, 1024),
        (256, 4096),
        (1024, 128),
        (1024, 1024),
        (4096, 128),
        (4096, 1024),
    ]

    print(f"\n{'T':>6} {'H':>6} {'loop (ms)':>10} {'scan (ms)':>10} "
          f"{'chunk (ms)':>11} {'scan/loop':>10} {'chunk/loop':>11}")
    print("-" * 75)

    for T, H in configs:
        x = mx.random.normal((T, batch_size, H))

        # ── for-loop ──
        for _ in range(warmup):
            r = lif_loop(x, beta)
            mx.eval(r)

        t0 = time.perf_counter()
        for _ in range(repeats):
            r = lif_loop(x, beta)
            mx.eval(r)
        t_loop = (time.perf_counter() - t0) / repeats * 1000

        # ── cumsum scan ──
        for _ in range(warmup):
            r = lif_cumsum_scan(x, beta)
            mx.eval(r)

        t0 = time.perf_counter()
        for _ in range(repeats):
            r = lif_cumsum_scan(x, beta)
            mx.eval(r)
        t_scan = (time.perf_counter() - t0) / repeats * 1000

        # ── chunked scan ──
        for _ in range(warmup):
            r = lif_chunked_scan(x, beta, chunk_size=64)
            mx.eval(r)

        t0 = time.perf_counter()
        for _ in range(repeats):
            r = lif_chunked_scan(x, beta, chunk_size=64)
            mx.eval(r)
        t_chunk = (time.perf_counter() - t0) / repeats * 1000

        speedup_scan = t_loop / t_scan if t_scan > 0 else float("inf")
        speedup_chunk = t_loop / t_chunk if t_chunk > 0 else float("inf")

        print(
            f"{T:>6} {H:>6} {t_loop:>10.2f} {t_scan:>10.2f} "
            f"{t_chunk:>11.2f} {speedup_scan:>9.2f}x {speedup_chunk:>10.2f}x"
        )

    print()
    print("Note: scan only computes the LINEAR recurrence (no fire/reset).")
    print("Full LIF with fire/reset still requires a sequential loop.")


if __name__ == "__main__":
    print("=" * 75)
    print("LIF Linear Scan Benchmark")
    print("=" * 75)
    print()

    verify_correctness()
    benchmark()
