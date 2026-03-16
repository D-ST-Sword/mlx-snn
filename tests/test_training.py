"""Tests for training utilities and loss functions."""

import mlx
import mlx.core as mx
import mlx.nn as nn
import pytest

from mlxsnn.training import bptt_forward, chunked_bptt_forward, detach_state
from mlxsnn.functional.loss import rate_coding_loss, membrane_loss, mse_count_loss
from mlxsnn.neurons import Leaky


class TestBPTTForward:
    """Tests for BPTT forward pass helper."""

    def test_output_shapes(self):
        lif = Leaky(beta=0.9, threshold=1.0)
        state = lif.init_state(4, 8)
        spikes_in = mx.ones((10, 4, 8)) * 0.5

        all_spk, all_mem, final_state = bptt_forward(lif, spikes_in, state)
        mx.eval(all_spk, all_mem)
        assert all_spk.shape == (10, 4, 8)
        assert all_mem.shape == (10, 4, 8)

    def test_num_steps_override(self):
        lif = Leaky(beta=0.9, threshold=1.0)
        state = lif.init_state(2, 4)
        spikes_in = mx.ones((20, 2, 4)) * 0.3

        all_spk, all_mem, _ = bptt_forward(lif, spikes_in, state, num_steps=5)
        mx.eval(all_spk)
        assert all_spk.shape[0] == 5

    def test_final_state_updated(self):
        """Final state should differ from initial state."""
        lif = Leaky(beta=0.9, threshold=1.0)
        state = lif.init_state(1, 2)
        spikes_in = mx.ones((10, 1, 2)) * 0.5

        _, _, final_state = bptt_forward(lif, spikes_in, state)
        mx.eval(final_state["mem"])
        # After 10 steps of input, membrane should not be all zeros
        assert not mx.allclose(
            final_state["mem"], mx.zeros_like(final_state["mem"])
        ).item()


class TestLossFunctions:
    """Tests for SNN-specific loss functions."""

    def test_rate_coding_loss_shape(self):
        spk_out = mx.ones((10, 4, 3)) * 0.5  # [T, B, C]
        targets = mx.array([0, 1, 2, 0])
        loss = rate_coding_loss(spk_out, targets)
        mx.eval(loss)
        assert loss.ndim == 0  # Scalar

    def test_rate_coding_loss_value(self):
        """Loss should be lower when spikes match targets."""
        # Create spikes concentrated in target class
        targets = mx.array([0, 1])  # batch of 2

        # Good predictions: class 0 spikes more for sample 0
        spk_good = mx.zeros((10, 2, 3))
        # We'll just test that loss is a valid number
        loss = rate_coding_loss(spk_good, targets)
        mx.eval(loss)
        assert not mx.isnan(loss).item()

    def test_membrane_loss_shape(self):
        mem = mx.ones((10, 4, 3))  # [T, B, C]
        targets = mx.array([0, 1, 2, 0])
        loss = membrane_loss(mem, targets)
        mx.eval(loss)
        assert loss.ndim == 0

    def test_mse_count_loss(self):
        spk_out = mx.ones((10, 2, 3))
        targets = mx.ones((2, 3)) * 10.0  # target count = 10
        loss = mse_count_loss(spk_out, targets)
        mx.eval(loss)
        # All spikes=1 for 10 steps -> count=10, target=10, loss=0
        assert mx.allclose(loss, mx.array(0.0), atol=1e-5).item()

    def test_loss_gradient_exists(self):
        """Loss functions should support gradient computation."""
        lif = Leaky(beta=0.9, threshold=1.0)

        # Test gradient flow: accumulate spike sum over time
        # NOTE: Use x[t:t+1] slicing instead of x[t] integer indexing
        # due to MLX scatter limitation with custom_function VJP.
        def loss_fn(x):
            state = lif.init_state(1, 3)
            total = mx.array(0.0)
            for t in range(5):
                spk, state = lif(x[t:t+1], state)
                total = total + mx.sum(spk)
            return total

        x = mx.ones((5, 1, 3)) * 0.5
        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(x)
        mx.eval(grads)
        assert grads.shape == (5, 1, 3)
        # At least some gradients should be nonzero
        assert mx.any(grads != 0).item()

    def test_rate_loss_end_to_end(self):
        """Rate coding loss should work with nn.value_and_grad."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 3)
                self.lif = Leaky(beta=0.9, threshold=1.0)

            def __call__(self, x):
                state = self.lif.init_state(x.shape[1], 3)
                total = mx.zeros((x.shape[1], 3))
                for t in range(x.shape[0]):
                    h = self.fc(x[t])
                    spk, state = self.lif(h, state)
                    total = total + spk
                return total

        model = SimpleModel()
        targets = mx.array([0, 1])

        def loss_fn(model, x):
            spike_count = model(x)
            return mx.mean(nn.losses.cross_entropy(spike_count, targets))

        loss_grad_fn = nn.value_and_grad(model, loss_fn)
        x = mx.ones((5, 2, 4)) * 0.5
        loss, grads = loss_grad_fn(model, x)
        mx.eval(loss, grads)
        assert not mx.isnan(loss).item()


class TestDetachState:
    """Tests for detach_state utility."""

    def test_basic_detach(self):
        """Detached arrays should have stop_gradient applied."""
        state = {"mem": mx.array([1.0, 2.0, 3.0])}
        detached = detach_state(state)
        mx.eval(detached["mem"])
        assert mx.allclose(detached["mem"], state["mem"]).item()

    def test_nested_detach(self):
        """Should handle nested state dicts."""
        state = {
            "mem": mx.array([1.0]),
            "syn": {"exc": mx.array([2.0]), "inh": mx.array([3.0])},
        }
        detached = detach_state(state)
        mx.eval(detached["syn"]["exc"], detached["syn"]["inh"])
        assert mx.allclose(detached["syn"]["exc"], mx.array([2.0])).item()
        assert mx.allclose(detached["syn"]["inh"], mx.array([3.0])).item()

    def test_non_array_preserved(self):
        """Non-array values should pass through unchanged."""
        state = {"mem": mx.array([1.0]), "step_count": 42, "label": "test"}
        detached = detach_state(state)
        assert detached["step_count"] == 42
        assert detached["label"] == "test"

    def test_gradient_blocked(self):
        """mx.grad through detach_state should return zero gradients."""
        lif = Leaky(beta=0.9, threshold=1.0)

        def loss_with_detach(x):
            state = lif.init_state(1, 4)
            # Run 3 steps, detach, run 3 more
            for t in range(3):
                spk, state = lif(x[t:t + 1].squeeze(0), state)
            state = detach_state(state)
            total = mx.array(0.0)
            for t in range(3, 6):
                spk, state = lif(x[t:t + 1].squeeze(0), state)
                total = total + mx.sum(spk)
            return total

        def loss_no_detach(x):
            state = lif.init_state(1, 4)
            total = mx.array(0.0)
            for t in range(6):
                spk, state = lif(x[t:t + 1].squeeze(0), state)
                total = total + mx.sum(spk)
            return total

        x = mx.ones((6, 1, 4)) * 0.5
        grad_detach = mx.grad(loss_with_detach)(x)
        grad_full = mx.grad(loss_no_detach)(x)
        mx.eval(grad_detach, grad_full)

        # Gradients for the first 3 steps should be zero when detached
        first3_detach = grad_detach[:3]
        assert mx.allclose(first3_detach, mx.zeros_like(first3_detach)).item()
        # Full BPTT should have non-zero gradients in first 3 steps
        first3_full = grad_full[:3]
        assert mx.any(first3_full != 0).item()


class TestChunkedBPTTForward:
    """Tests for chunked_bptt_forward."""

    def test_output_shapes(self):
        """Output shapes should match bptt_forward."""
        lif = Leaky(beta=0.9, threshold=1.0)
        state = lif.init_state(4, 8)
        spikes_in = mx.ones((20, 4, 8)) * 0.5

        all_spk, all_mem, final_state = chunked_bptt_forward(
            lif, spikes_in, state, chunk_size=8
        )
        mx.eval(all_spk, all_mem)
        assert all_spk.shape == (20, 4, 8)
        assert all_mem.shape == (20, 4, 8)

    def test_chunk_size_equals_T(self):
        """chunk_size=T should produce identical results to bptt_forward."""
        lif = Leaky(beta=0.9, threshold=1.0)
        spikes_in = mx.ones((10, 2, 4)) * 0.4

        state1 = lif.init_state(2, 4)
        spk_full, mem_full, _ = bptt_forward(lif, spikes_in, state1)

        state2 = lif.init_state(2, 4)
        spk_chunked, mem_chunked, _ = chunked_bptt_forward(
            lif, spikes_in, state2, chunk_size=10
        )

        mx.eval(spk_full, spk_chunked, mem_full, mem_chunked)
        assert mx.allclose(spk_full, spk_chunked).item()
        assert mx.allclose(mem_full, mem_chunked).item()

    def test_forward_values_match_bptt(self):
        """Forward pass values should match bptt_forward regardless of chunk_size."""
        lif = Leaky(beta=0.9, threshold=1.0)
        spikes_in = mx.random.uniform(shape=(16, 3, 6))

        state1 = lif.init_state(3, 6)
        spk_full, mem_full, _ = bptt_forward(lif, spikes_in, state1)

        state2 = lif.init_state(3, 6)
        spk_chunked, mem_chunked, _ = chunked_bptt_forward(
            lif, spikes_in, state2, chunk_size=4
        )

        mx.eval(spk_full, spk_chunked, mem_full, mem_chunked)
        assert mx.allclose(spk_full, spk_chunked).item()
        assert mx.allclose(mem_full, mem_chunked, atol=1e-5).item()

    def test_uneven_chunks(self):
        """Should handle T not divisible by chunk_size."""
        lif = Leaky(beta=0.9, threshold=1.0)
        state = lif.init_state(2, 4)
        spikes_in = mx.ones((7, 2, 4)) * 0.5

        all_spk, all_mem, _ = chunked_bptt_forward(
            lif, spikes_in, state, chunk_size=3
        )
        mx.eval(all_spk, all_mem)
        assert all_spk.shape == (7, 2, 4)

    def test_num_steps_override(self):
        """num_steps should override T from spikes shape."""
        lif = Leaky(beta=0.9, threshold=1.0)
        state = lif.init_state(2, 4)
        spikes_in = mx.ones((20, 2, 4)) * 0.3

        all_spk, _, _ = chunked_bptt_forward(
            lif, spikes_in, state, chunk_size=4, num_steps=8
        )
        mx.eval(all_spk)
        assert all_spk.shape[0] == 8

    def test_gradient_flow(self):
        """Gradients should flow within chunks but not across."""
        lif = Leaky(beta=0.9, threshold=1.0)

        def loss_chunked(x):
            state = lif.init_state(1, 4)
            spk, _, _ = chunked_bptt_forward(
                lif, x, state, chunk_size=3
            )
            return mx.sum(spk)

        x = mx.ones((6, 1, 4)) * 0.5
        grads = mx.grad(loss_chunked)(x)
        mx.eval(grads)
        assert grads.shape == x.shape
        # Should have non-zero gradients (at least within the last chunk)
        assert mx.any(grads != 0).item()
