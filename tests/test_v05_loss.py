"""Tests for v0.5 activity regularization loss functions."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlxsnn.functional.loss import activity_reg_loss, l1_spike_loss, l2_spike_loss


class TestActivityRegLoss:
    def test_scalar_output(self):
        spk = mx.ones((10, 4, 5)) * 0.5
        loss = activity_reg_loss(spk, target_rate=0.1)
        mx.eval(loss)
        assert loss.ndim == 0

    def test_perfect_rate(self):
        """Loss should be zero when mean rate matches target."""
        spk = mx.ones((10, 4, 5)) * 0.3
        loss = activity_reg_loss(spk, target_rate=0.3)
        mx.eval(loss)
        assert loss.item() < 1e-6

    def test_higher_rate_higher_loss(self):
        spk_low = mx.ones((10, 4, 5)) * 0.1
        spk_high = mx.ones((10, 4, 5)) * 0.9
        loss_low = activity_reg_loss(spk_low, target_rate=0.1)
        loss_high = activity_reg_loss(spk_high, target_rate=0.1)
        mx.eval(loss_low, loss_high)
        assert loss_high.item() > loss_low.item()

    def test_nonnegative(self):
        spk = mx.ones((10, 4, 5)) * 0.5
        loss = activity_reg_loss(spk, target_rate=0.1)
        mx.eval(loss)
        assert loss.item() >= 0.0

    def test_differentiable(self):
        """Gradient should flow through activity_reg_loss."""
        def loss_fn(w):
            spk = mx.sigmoid(w)
            spk = mx.broadcast_to(spk, (10,) + w.shape)
            return activity_reg_loss(spk, target_rate=0.1)

        w = mx.ones((4, 5)) * 0.5
        grad_fn = mx.grad(loss_fn)
        g = grad_fn(w)
        mx.eval(g)
        assert g.shape == w.shape
        assert not mx.allclose(g, mx.zeros_like(g)).item()


class TestL1SpikeLoss:
    def test_scalar_output(self):
        spk = mx.ones((10, 4, 5))
        loss = l1_spike_loss(spk)
        mx.eval(loss)
        assert loss.ndim == 0

    def test_zero_spikes(self):
        spk = mx.zeros((10, 4, 5))
        loss = l1_spike_loss(spk)
        mx.eval(loss)
        assert loss.item() == 0.0

    def test_value(self):
        """All ones: count=10 per neuron, mean(abs(10))=10."""
        spk = mx.ones((10, 4, 5))
        loss = l1_spike_loss(spk)
        mx.eval(loss)
        assert abs(loss.item() - 10.0) < 1e-5

    def test_nonnegative(self):
        spk = mx.ones((10, 4, 5)) * 0.5
        loss = l1_spike_loss(spk)
        mx.eval(loss)
        assert loss.item() >= 0.0


class TestL2SpikeLoss:
    def test_scalar_output(self):
        spk = mx.ones((10, 4, 5))
        loss = l2_spike_loss(spk)
        mx.eval(loss)
        assert loss.ndim == 0

    def test_zero_spikes(self):
        spk = mx.zeros((10, 4, 5))
        loss = l2_spike_loss(spk)
        mx.eval(loss)
        assert loss.item() == 0.0

    def test_value(self):
        """All ones: count=10 per neuron, mean(10^2)=100."""
        spk = mx.ones((10, 4, 5))
        loss = l2_spike_loss(spk)
        mx.eval(loss)
        assert abs(loss.item() - 100.0) < 1e-4

    def test_l2_greater_than_l1(self):
        """For counts > 1, L2 should be larger than L1."""
        spk = mx.ones((10, 4, 5))
        l1 = l1_spike_loss(spk)
        l2 = l2_spike_loss(spk)
        mx.eval(l1, l2)
        assert l2.item() > l1.item()
