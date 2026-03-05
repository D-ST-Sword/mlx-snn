"""Tests for v0.4 surrogate gradient functions (sigmoid, triangular)."""

import mlx.core as mx
import numpy as np
import pytest

from mlxsnn.surrogate import get_surrogate
from mlxsnn.surrogate.sigmoid import sigmoid_surrogate
from mlxsnn.surrogate.triangular import triangular_surrogate


class TestSigmoidSurrogate:
    """Tests for the sigmoid surrogate gradient."""

    def test_forward_heaviside(self):
        fn = sigmoid_surrogate(slope=25.0)
        x = mx.array([-1.0, -0.1, 0.0, 0.1, 1.0])
        result = fn(x)
        mx.eval(result)
        expected = mx.array([0.0, 0.0, 1.0, 1.0, 1.0])
        assert mx.allclose(result, expected).item()

    def test_backward_nonzero(self):
        fn = sigmoid_surrogate(slope=25.0)
        x = mx.array([0.0])
        grad_fn = mx.grad(lambda x: mx.sum(fn(x)))
        grad = grad_fn(x)
        mx.eval(grad)
        assert mx.abs(grad).item() > 0

    def test_gradient_peak_at_zero(self):
        fn = sigmoid_surrogate(slope=25.0)
        grad_fn = mx.grad(lambda x: mx.sum(fn(x)))
        grad_at_zero = mx.abs(grad_fn(mx.array([0.0])))
        grad_at_one = mx.abs(grad_fn(mx.array([1.0])))
        mx.eval(grad_at_zero, grad_at_one)
        assert grad_at_zero.item() > grad_at_one.item()

    def test_registry(self):
        fn = get_surrogate("sigmoid", scale=25.0)
        x = mx.array([0.5])
        result = fn(x)
        mx.eval(result)
        assert result.item() == 1.0


class TestTriangularSurrogate:
    """Tests for the triangular surrogate gradient."""

    def test_forward_heaviside(self):
        fn = triangular_surrogate(scale=1.0)
        x = mx.array([-2.0, -0.1, 0.0, 0.1, 2.0])
        result = fn(x)
        mx.eval(result)
        expected = mx.array([0.0, 0.0, 1.0, 1.0, 1.0])
        assert mx.allclose(result, expected).item()

    def test_backward_nonzero(self):
        """Gradient at threshold boundary should be 1.0 (tent peak)."""
        fn = triangular_surrogate(scale=1.0)
        grad_fn = mx.grad(lambda x: mx.sum(fn(x)))
        # At threshold (x=0): gradient = 1.0 (peak of tent)
        grad_at_zero = grad_fn(mx.array([0.0]))
        mx.eval(grad_at_zero)
        assert np.isclose(grad_at_zero.item(), 1.0, atol=1e-5)
        # Midway: gradient = 0.5 (linear decay)
        grad_half = grad_fn(mx.array([0.5]))
        mx.eval(grad_half)
        assert np.isclose(grad_half.item(), 0.5, atol=1e-5)

    def test_gradient_zero_outside_window(self):
        """Gradient should be zero for |x| >= 1 (tent support ends)."""
        fn = triangular_surrogate(scale=1.0)
        grad_fn = mx.grad(lambda x: mx.sum(fn(x)))
        grad_far = grad_fn(mx.array([2.0]))
        mx.eval(grad_far)
        assert np.isclose(grad_far.item(), 0.0, atol=1e-5)
        grad_neg_far = grad_fn(mx.array([-2.0]))
        mx.eval(grad_neg_far)
        assert np.isclose(grad_neg_far.item(), 0.0, atol=1e-5)

    def test_registry(self):
        fn = get_surrogate("triangular", scale=1.0)
        x = mx.array([0.5])
        result = fn(x)
        mx.eval(result)
        assert result.item() == 1.0
