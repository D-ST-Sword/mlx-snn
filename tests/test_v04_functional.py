"""Tests for v0.4 functional API additions (losses, spike utilities)."""

import mlx.core as mx
import numpy as np
import pytest

from mlxsnn.functional.loss import (
    ce_rate_loss,
    ce_count_loss,
    mse_membrane_loss,
    spike_rate,
    spike_count,
)


class TestCERateLoss:
    """Tests for ce_rate_loss."""

    def test_output_scalar(self):
        spk = mx.ones((10, 4, 5))
        targets = mx.array([0, 1, 2, 3])
        loss = ce_rate_loss(spk, targets)
        mx.eval(loss)
        assert loss.ndim == 0

    def test_differentiable(self):
        spk = mx.random.uniform(shape=(10, 4, 5))
        targets = mx.array([0, 1, 2, 3])
        mx.eval(spk)
        grad_fn = mx.grad(lambda s: ce_rate_loss(s, targets))
        grads = grad_fn(spk)
        mx.eval(grads)
        assert grads.shape == spk.shape

    def test_correct_class_lower_loss(self):
        """Spike rate concentrated in correct class should give lower loss."""
        targets = mx.array([0])
        # Good prediction: high rate for class 0
        spk_good = mx.zeros((10, 1, 3))
        spk_good = spk_good.at[:, :, 0].add(1.0)
        # Bad prediction: uniform rate
        spk_bad = mx.ones((10, 1, 3)) * 0.3
        loss_good = ce_rate_loss(spk_good, targets)
        loss_bad = ce_rate_loss(spk_bad, targets)
        mx.eval(loss_good, loss_bad)
        assert loss_good.item() < loss_bad.item()


class TestCECountLoss:
    """Tests for ce_count_loss."""

    def test_output_scalar(self):
        spk = mx.ones((10, 4, 5))
        targets = mx.array([0, 1, 2, 3])
        loss = ce_count_loss(spk, targets)
        mx.eval(loss)
        assert loss.ndim == 0

    def test_differentiable(self):
        spk = mx.random.uniform(shape=(10, 4, 5))
        targets = mx.array([0, 1, 2, 3])
        mx.eval(spk)
        grad_fn = mx.grad(lambda s: ce_count_loss(s, targets))
        grads = grad_fn(spk)
        mx.eval(grads)
        assert grads.shape == spk.shape


class TestMSEMembraneLoss:
    """Tests for mse_membrane_loss."""

    def test_output_scalar(self):
        mem = mx.random.normal((4, 5))
        targets = mx.array([0, 1, 2, 3])
        mx.eval(mem)
        loss = mse_membrane_loss(mem, targets)
        mx.eval(loss)
        assert loss.ndim == 0

    def test_target_encoding(self):
        """Correct class should have on_target, others off_target."""
        mem = mx.zeros((1, 3))
        targets = mx.array([1])  # Class 1
        # mem=[0,0,0], target=[0,1,0] -> MSE = (0^2 + 1^2 + 0^2) / 3
        loss = mse_membrane_loss(mem, targets, on_target=1.0, off_target=0.0)
        mx.eval(loss)
        expected = 1.0 / 3.0
        assert np.isclose(loss.item(), expected, atol=1e-5)

    def test_perfect_prediction(self):
        """Loss should be zero when mem matches targets exactly."""
        mem = mx.array([[0.0, 1.0, 0.0]])
        targets = mx.array([1])
        loss = mse_membrane_loss(mem, targets, on_target=1.0, off_target=0.0)
        mx.eval(loss)
        assert np.isclose(loss.item(), 0.0, atol=1e-5)

    def test_custom_on_off(self):
        mem = mx.array([[0.8, -0.2, -0.2]])
        targets = mx.array([0])
        loss = mse_membrane_loss(mem, targets, on_target=0.8, off_target=-0.2)
        mx.eval(loss)
        assert np.isclose(loss.item(), 0.0, atol=1e-5)


class TestSpikeRateAndCount:
    """Tests for spike_rate and spike_count."""

    def test_spike_rate_shape(self):
        spk = mx.ones((10, 4, 5))
        rate = spike_rate(spk)
        mx.eval(rate)
        assert rate.shape == (4, 5)

    def test_spike_rate_value(self):
        spk = mx.ones((10, 2, 3))
        rate = spike_rate(spk)
        mx.eval(rate)
        assert mx.allclose(rate, mx.ones((2, 3))).item()

    def test_spike_rate_half(self):
        spk = mx.zeros((10, 1, 1))
        spk = spk.at[:5].add(1.0)
        rate = spike_rate(spk)
        mx.eval(rate)
        assert np.isclose(rate.item(), 0.5, atol=1e-5)

    def test_spike_count_shape(self):
        spk = mx.ones((10, 4, 5))
        count = spike_count(spk)
        mx.eval(count)
        assert count.shape == (4, 5)

    def test_spike_count_value(self):
        spk = mx.ones((10, 2, 3))
        count = spike_count(spk)
        mx.eval(count)
        assert mx.allclose(count, mx.full((2, 3), 10.0)).item()

    def test_rate_count_consistency(self):
        """spike_rate * T should equal spike_count."""
        T = 20
        spk = mx.random.uniform(shape=(T, 4, 5))
        spk = mx.where(spk > 0.5, mx.ones_like(spk), mx.zeros_like(spk))
        mx.eval(spk)
        rate = spike_rate(spk)
        count = spike_count(spk)
        mx.eval(rate, count)
        assert mx.allclose(rate * T, count, atol=1e-5).item()
