"""Tests for v0.5 spike-aware dropout."""

import mlx.core as mx
import pytest

from mlxsnn.layers.dropout import SpikeDropout


class TestSpikeDropout:
    def test_eval_passthrough(self):
        """In eval mode, all spikes should pass through."""
        drop = SpikeDropout(p=0.5)
        drop.eval()
        x = mx.ones((4, 10))
        out = drop(x)
        mx.eval(out)
        assert mx.array_equal(out, x).item()

    def test_train_drops_spikes(self):
        """In train mode with high p, some spikes should be dropped."""
        drop = SpikeDropout(p=0.9)
        drop.train()
        x = mx.ones((100, 100))
        out = drop(x)
        mx.eval(out)
        total = mx.sum(out).item()
        # With p=0.9, expect ~10% to survive (1000 of 10000)
        assert total < 5000  # Very loose upper bound

    def test_binary_output(self):
        """Output should remain binary {0, 1}."""
        drop = SpikeDropout(p=0.5)
        drop.train()
        x = mx.ones((100, 100))
        out = drop(x)
        mx.eval(out)
        unique_vals = set(out.reshape(-1).tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_zero_p_passthrough(self):
        """p=0 should never drop anything."""
        drop = SpikeDropout(p=0.0)
        drop.train()
        x = mx.ones((4, 10))
        out = drop(x)
        mx.eval(out)
        assert mx.array_equal(out, x).item()

    def test_shape_preserved(self):
        drop = SpikeDropout(p=0.5)
        drop.train()
        x = mx.ones((2, 3, 4))
        out = drop(x)
        assert list(out.shape) == [2, 3, 4]

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            SpikeDropout(p=1.0)
        with pytest.raises(ValueError):
            SpikeDropout(p=-0.1)
