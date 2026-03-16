#!/usr/bin/env python3
"""Verify whether mx.custom_function + matmul backward shape bug is fixed.

This script tests the known issue where mx.custom_function VJP loses
dimensions when composed with matmul backward passes. If all tests pass,
we can migrate surrogate gradient functions from the STE pattern to the
cleaner mx.custom_function + .vjp pattern.

Run:
    python benchmarks/test_custom_vjp.py
"""

from __future__ import annotations

import sys
import time

import mlx.core as mx

print(f"Python: {sys.version}")
print(f"MLX version: {mx.__version__}")
print()


def test_basic_custom_function():
    """Test 1: Basic mx.custom_function without matmul."""
    print("Test 1: Basic custom_function (no matmul) ... ", end="", flush=True)

    @mx.custom_function
    def heaviside(x):
        return mx.where(x >= 0, 1.0, 0.0)

    @heaviside.vjp
    def heaviside_vjp(primals, cotangents, outputs):
        x = primals[0]
        grad = cotangents[0]
        # Fast sigmoid surrogate gradient
        scale = 25.0
        surrogate_grad = scale / (2.0 * (1.0 + scale * mx.abs(x)) ** 2)
        return (grad * surrogate_grad,)

    x = mx.random.normal((4, 8))

    def loss_fn(x):
        return mx.sum(heaviside(x - 0.5))

    try:
        grads = mx.grad(loss_fn)(x)
        mx.eval(grads)

        if grads.shape != (4, 8):
            print(f"FAILED  Expected (4, 8), got {grads.shape}")
            return False
        if not mx.any(grads != 0).item():
            print("FAILED  All gradients are zero")
            return False
        print(f"PASSED  shape={grads.shape}")
        return True
    except Exception as e:
        print(f"FAILED  {type(e).__name__}: {e}")
        return False


def test_custom_function_with_matmul():
    """Test 2: custom_function composed with matmul backward — the known bug."""
    print("Test 2: custom_function + matmul backward ... ", end="", flush=True)

    @mx.custom_function
    def heaviside(x):
        return mx.where(x >= 0, 1.0, 0.0)

    @heaviside.vjp
    def heaviside_vjp(primals, cotangents, outputs):
        x = primals[0]
        grad = cotangents[0]
        scale = 25.0
        surrogate_grad = scale / (2.0 * (1.0 + scale * mx.abs(x)) ** 2)
        return (grad * surrogate_grad,)

    W = mx.random.normal((128, 10))

    def loss_fn(W):
        x = mx.random.normal((4, 128))
        h = x @ W  # matmul
        spk = heaviside(h - 1.0)  # surrogate
        return mx.sum(spk)

    try:
        grads = mx.grad(loss_fn)(W)
        mx.eval(grads)
        assert grads.shape == (128, 10), f"Expected (128, 10), got {grads.shape}"
        assert mx.any(grads != 0).item(), "Expected non-zero gradients"
        print(f"PASSED  shape={grads.shape}")
        return True
    except Exception as e:
        print(f"FAILED  {type(e).__name__}: {e}")
        return False


def test_custom_function_lif_loop():
    """Test 3: custom_function inside a LIF time loop with matmul."""
    print("Test 3: custom_function in LIF loop + matmul ... ", end="", flush=True)

    @mx.custom_function
    def heaviside(x):
        return mx.where(x >= 0, 1.0, 0.0)

    @heaviside.vjp
    def heaviside_vjp(primals, cotangents, outputs):
        x = primals[0]
        grad = cotangents[0]
        scale = 25.0
        surrogate_grad = scale / (2.0 * (1.0 + scale * mx.abs(x)) ** 2)
        return (grad * surrogate_grad,)

    W = mx.random.normal((32, 16))
    beta = 0.9
    threshold = 1.0

    def loss_fn(W):
        x = mx.random.normal((5, 4, 32))  # [T, B, in]
        mem = mx.zeros((4, 16))
        total = mx.array(0.0)
        for t in range(5):
            h = x[t:t + 1].squeeze(0) @ W
            mem = beta * mem + h
            spk = heaviside(mem - threshold)
            mem = mem - spk * threshold
            total = total + mx.sum(spk)
        return total

    try:
        grads = mx.grad(loss_fn)(W)
        mx.eval(grads)
        assert grads.shape == (32, 16), f"Expected (32, 16), got {grads.shape}"
        assert mx.any(grads != 0).item(), "Expected non-zero gradients"
        print(f"PASSED  shape={grads.shape}")
        return True
    except Exception as e:
        print(f"FAILED  {type(e).__name__}: {e}")
        return False


def test_custom_function_multiple_surrogates():
    """Test 4: Multiple different surrogate functions in same graph."""
    print("Test 4: Multiple surrogates in one graph ... ", end="", flush=True)

    def make_surrogate(scale):
        @mx.custom_function
        def heaviside(x):
            return mx.where(x >= 0, 1.0, 0.0)

        @heaviside.vjp
        def heaviside_vjp(primals, cotangents, outputs):
            x = primals[0]
            grad = cotangents[0]
            surrogate_grad = scale / (2.0 * (1.0 + scale * mx.abs(x)) ** 2)
            return (grad * surrogate_grad,)

        return heaviside

    surr_a = make_surrogate(25.0)
    surr_b = make_surrogate(10.0)

    W1 = mx.random.normal((8, 8))
    W2 = mx.random.normal((8, 4))

    def loss_fn(W1, W2):
        x = mx.random.normal((2, 8))
        h1 = x @ W1
        s1 = surr_a(h1 - 1.0)
        h2 = s1 @ W2
        s2 = surr_b(h2 - 0.5)
        return mx.sum(s2)

    try:
        grad_W1, grad_W2 = mx.grad(loss_fn, argnums=(0, 1))(W1, W2)
        mx.eval(grad_W1, grad_W2)
        assert grad_W1.shape == (8, 8), f"W1 grad: expected (8,8), got {grad_W1.shape}"
        assert grad_W2.shape == (8, 4), f"W2 grad: expected (8,4), got {grad_W2.shape}"
        print(f"PASSED  W1={grad_W1.shape}, W2={grad_W2.shape}")
        return True
    except Exception as e:
        print(f"FAILED  {type(e).__name__}: {e}")
        return False


def bench_ste_vs_custom(iterations=100):
    """Benchmark: STE pattern vs custom_function overhead."""
    print("\nBenchmark: STE vs custom_function overhead")
    print("-" * 50)

    scale = 25.0

    # STE pattern (current implementation)
    def ste_forward(x):
        heaviside = mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        approx = 0.5 * scale * x / (1.0 + scale * mx.abs(x)) + 0.5
        return mx.stop_gradient(heaviside - approx) + approx

    # custom_function pattern
    @mx.custom_function
    def cf_forward(x):
        return mx.where(x >= 0, 1.0, 0.0)

    @cf_forward.vjp
    def cf_backward(primals, cotangents, outputs):
        x = primals[0]
        grad = cotangents[0]
        surrogate_grad = scale / (2.0 * (1.0 + scale * mx.abs(x)) ** 2)
        return (grad * surrogate_grad,)

    x = mx.random.normal((64, 1024))
    W = mx.random.normal((1024, 512))

    def loss_ste(W):
        h = x @ W
        return mx.sum(ste_forward(h - 1.0))

    def loss_cf(W):
        h = x @ W
        return mx.sum(cf_forward(h - 1.0))

    # Warmup
    for _ in range(5):
        g1 = mx.grad(loss_ste)(W)
        g2 = mx.grad(loss_cf)(W)
        mx.eval(g1, g2)

    # Benchmark STE
    t0 = time.perf_counter()
    for _ in range(iterations):
        g = mx.grad(loss_ste)(W)
        mx.eval(g)
    t_ste = time.perf_counter() - t0

    # Benchmark custom_function
    t0 = time.perf_counter()
    for _ in range(iterations):
        g = mx.grad(loss_cf)(W)
        mx.eval(g)
    t_cf = time.perf_counter() - t0

    print(f"  STE pattern:        {t_ste*1000/iterations:.2f} ms/iter")
    print(f"  custom_function:    {t_cf*1000/iterations:.2f} ms/iter")
    print(f"  Speedup:            {t_ste/t_cf:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("mx.custom_function VJP Compatibility Tests")
    print("=" * 60)
    print()

    results = []
    results.append(("Basic custom_function", test_basic_custom_function()))
    results.append(("+ matmul backward", test_custom_function_with_matmul()))
    results.append(("LIF loop + matmul", test_custom_function_lif_loop()))
    results.append(("Multiple surrogates", test_custom_function_multiple_surrogates()))

    print()
    print("=" * 60)
    all_passed = all(r[1] for r in results)
    passed = sum(1 for r in results if r[1])
    print(f"Results: {passed}/{len(results)} passed")

    if all_passed:
        print("\nAll tests PASSED — mx.custom_function VJP bug appears FIXED.")
        print("Recommendation: Migrate surrogate functions from STE to custom_function.")
        bench_ste_vs_custom()
    else:
        print("\nSome tests FAILED — mx.custom_function VJP bug is still present.")
        print("Recommendation: Keep STE pattern; optimize intermediate arrays.")

    print()
