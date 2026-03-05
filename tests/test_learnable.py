"""Tests for learnable parameters (learn_beta, learn_threshold) across neurons."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlxsnn.neurons.lif import Leaky
from mlxsnn.neurons.if_neuron import IF
from mlxsnn.neurons.synaptic import Synaptic
from mlxsnn.neurons.alpha import Alpha
from mlxsnn.neurons.adaptive_lif import ALIF
from mlxsnn.neurons.rleaky import RLeaky
from mlxsnn.neurons.rsynaptic import RSynaptic


class TestLearnThreshold:
    """Tests for learn_threshold across all neuron types."""

    def test_leaky_learn_threshold(self):
        neuron = Leaky(beta=0.9, learn_threshold=True)
        mx.eval(neuron.parameters())
        assert isinstance(neuron.threshold, mx.array)

    def test_leaky_fixed_threshold(self):
        neuron = Leaky(beta=0.9, learn_threshold=False)
        assert hasattr(neuron, "_threshold_const")
        assert neuron._get_threshold() == 1.0

    def test_if_learn_threshold(self):
        neuron = IF(learn_threshold=True)
        mx.eval(neuron.parameters())
        assert isinstance(neuron.threshold, mx.array)

    def test_synaptic_learn_threshold(self):
        neuron = Synaptic(alpha=0.8, beta=0.9, learn_threshold=True)
        mx.eval(neuron.parameters())
        assert isinstance(neuron.threshold, mx.array)

    def test_alpha_learn_threshold(self):
        neuron = Alpha(alpha=0.85, beta=0.9, learn_threshold=True)
        mx.eval(neuron.parameters())
        assert isinstance(neuron.threshold, mx.array)

    def test_alif_learn_threshold(self):
        neuron = ALIF(beta=0.9, learn_threshold=True)
        mx.eval(neuron.parameters())
        assert isinstance(neuron.threshold, mx.array)

    def test_rleaky_learn_threshold(self):
        neuron = RLeaky(beta=0.9, learn_threshold=True)
        mx.eval(neuron.parameters())
        assert isinstance(neuron.threshold, mx.array)

    def test_rsynaptic_learn_threshold(self):
        neuron = RSynaptic(alpha=0.8, beta=0.9, learn_threshold=True)
        mx.eval(neuron.parameters())
        assert isinstance(neuron.threshold, mx.array)


class TestLearnThresholdGradient:
    """Tests that gradient flows through learnable threshold."""

    def test_leaky_threshold_gradient(self):
        neuron = Leaky(beta=0.9, threshold=1.0, learn_threshold=True)
        fc = nn.Linear(4, 3)
        mx.eval(fc.parameters(), neuron.parameters())

        def loss_fn(params, x):
            neuron.load_weights(list(params.items()))
            state = neuron.init_state(batch_size=2, features=3)
            spk, state = neuron(fc(x), state)
            return mx.mean(state["mem"])

        x = mx.random.normal((2, 4))
        mx.eval(x)
        grads = mx.grad(loss_fn)(dict(neuron.parameters()), x)
        mx.eval(grads)
        assert "threshold" in grads
        assert mx.any(mx.abs(grads["threshold"]) > 0).item()

    def test_if_threshold_gradient(self):
        neuron = IF(threshold=1.0, learn_threshold=True)
        mx.eval(neuron.parameters())

        def loss_fn(params, x):
            neuron.load_weights(list(params.items()))
            state = neuron.init_state(batch_size=2, features=3)
            spk, state = neuron(x, state)
            return mx.mean(state["mem"])

        x = mx.random.normal((2, 3))
        mx.eval(x)
        grads = mx.grad(loss_fn)(dict(neuron.parameters()), x)
        mx.eval(grads)
        assert "threshold" in grads

    def test_synaptic_threshold_gradient(self):
        neuron = Synaptic(alpha=0.8, beta=0.9, learn_threshold=True)
        mx.eval(neuron.parameters())

        def loss_fn(params, x):
            neuron.load_weights(list(params.items()))
            state = neuron.init_state(batch_size=2, features=3)
            spk, state = neuron(x, state)
            return mx.mean(state["mem"])

        x = mx.random.normal((2, 3))
        mx.eval(x)
        grads = mx.grad(loss_fn)(dict(neuron.parameters()), x)
        mx.eval(grads)
        assert "threshold" in grads


class TestCombinedLearnable:
    """Tests for learn_beta + learn_threshold combined."""

    def test_leaky_both_learnable(self):
        neuron = Leaky(beta=0.9, learn_beta=True, learn_threshold=True)
        mx.eval(neuron.parameters())
        params = dict(neuron.parameters())
        assert "beta" in params
        assert "threshold" in params

    def test_synaptic_all_learnable(self):
        neuron = Synaptic(alpha=0.8, beta=0.9,
                          learn_alpha=True, learn_beta=True,
                          learn_threshold=True)
        mx.eval(neuron.parameters())
        params = dict(neuron.parameters())
        assert "alpha" in params
        assert "beta" in params
        assert "threshold" in params

    def test_rleaky_all_learnable(self):
        neuron = RLeaky(beta=0.9, V=1.0,
                        learn_beta=True, learn_V=True,
                        learn_threshold=True)
        mx.eval(neuron.parameters())
        params = dict(neuron.parameters())
        assert "beta" in params
        assert "V" in params
        assert "threshold" in params

    def test_forward_still_works(self):
        """Forward pass should work with all params learnable."""
        neuron = Leaky(beta=0.9, learn_beta=True, learn_threshold=True)
        mx.eval(neuron.parameters())
        state = neuron.init_state(batch_size=2, features=4)
        x = mx.ones((2, 4))
        spk, state = neuron(x, state)
        mx.eval(spk)
        assert spk.shape == (2, 4)
