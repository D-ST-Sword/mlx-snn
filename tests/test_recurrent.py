"""Tests for recurrent spiking neuron models (RLeaky, RSynaptic)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlxsnn.neurons.rleaky import RLeaky
from mlxsnn.neurons.rsynaptic import RSynaptic


class TestRLeaky:
    """Tests for the RLeaky (Recurrent LIF) neuron."""

    def test_forward_shape(self):
        neuron = RLeaky(beta=0.9)
        state = neuron.init_state(batch_size=4, features=128)
        x = mx.ones((4, 128))
        spk, new_state = neuron(x, state)
        mx.eval(spk)
        assert spk.shape == (4, 128)
        assert new_state["mem"].shape == (4, 128)
        assert new_state["spk"].shape == (4, 128)

    def test_init_state_zeros(self):
        neuron = RLeaky(beta=0.9)
        state = neuron.init_state(batch_size=2, features=64)
        mx.eval(state["mem"], state["spk"])
        assert mx.allclose(state["mem"], mx.zeros((2, 64))).item()
        assert mx.allclose(state["spk"], mx.zeros((2, 64))).item()

    def test_recurrence_effect(self):
        """Output at t should depend on spike at t-1 (not just input)."""
        neuron = RLeaky(beta=0.0, V=5.0)  # No decay, strong recurrence
        state = neuron.init_state(batch_size=1, features=1)
        x = mx.zeros((1, 1))

        # Step 1: no input, no previous spike -> no spike
        spk1, state = neuron(x, state)
        mx.eval(spk1, state["mem"])

        # Manually inject a spike into state
        state["spk"] = mx.ones((1, 1))

        # Step 2: no input, but recurrent spike -> mem gets V*1=5.0 -> spike
        spk2, state2 = neuron(x, state)
        mx.eval(spk2, state2["mem"])
        assert spk2.item() == 1.0  # Should spike due to recurrence

    def test_fixed_V(self):
        neuron = RLeaky(beta=0.9, V=2.0, learn_V=False)
        assert hasattr(neuron, "_V_const")
        assert neuron._get_V() == 2.0

    def test_learnable_V(self):
        neuron = RLeaky(beta=0.9, V=2.0, learn_V=True)
        mx.eval(neuron.parameters())
        assert isinstance(neuron.V, mx.array)

    def test_learn_beta(self):
        neuron = RLeaky(beta=0.9, learn_beta=True)
        mx.eval(neuron.parameters())
        assert isinstance(neuron.beta, mx.array)

    def test_subtract_reset(self):
        """Delayed reset: reset is applied based on previous membrane."""
        neuron = RLeaky(beta=0.0, V=0.0, threshold=1.0,
                        reset_mechanism="subtract")
        state = neuron.init_state(1, 1)
        x = mx.array([[1.5]])
        # Step 1: old_mem=0 < thr, no reset. mem=1.5, spk=1
        spk, state = neuron(x, state)
        mx.eval(spk, state["mem"])
        assert spk.item() == 1.0
        assert np.isclose(state["mem"].item(), 1.5, atol=1e-5)

        # Step 2: old_mem=1.5 >= thr, reset applied. base=0+1.5+0=1.5, mem=1.5-1.0=0.5
        spk2, state2 = neuron(x, state)
        mx.eval(spk2, state2["mem"])
        assert np.isclose(state2["mem"].item(), 0.5, atol=1e-5)

    def test_zero_reset(self):
        """Delayed reset: zero reset based on previous membrane."""
        neuron = RLeaky(beta=0.0, V=0.0, threshold=1.0,
                        reset_mechanism="zero")
        state = neuron.init_state(1, 1)
        x = mx.array([[1.5]])
        # Step 1: old_mem=0 < thr, no reset. mem=1.5, spk=1
        spk, state = neuron(x, state)
        mx.eval(spk, state["mem"])
        assert spk.item() == 1.0
        assert np.isclose(state["mem"].item(), 1.5, atol=1e-5)

        # Step 2: old_mem=1.5 >= thr, reset applied. base=0+1.5+0=1.5, mem=1.5*(1-1)=0
        spk2, state2 = neuron(x, state)
        mx.eval(spk2, state2["mem"])
        assert np.isclose(state2["mem"].item(), 0.0, atol=1e-5)

    def test_none_reset(self):
        neuron = RLeaky(beta=0.0, V=0.0, threshold=1.0,
                        reset_mechanism="none")
        state = neuron.init_state(1, 1)
        x = mx.array([[1.5]])
        spk, state = neuron(x, state)
        mx.eval(spk, state["mem"])
        assert spk.item() == 1.0
        assert np.isclose(state["mem"].item(), 1.5, atol=1e-5)

    def test_batch_independence(self):
        """Different batch elements should evolve independently."""
        neuron = RLeaky(beta=0.5, V=0.5)
        state = neuron.init_state(batch_size=2, features=1)
        x = mx.array([[2.0], [0.1]])
        spk, state = neuron(x, state)
        mx.eval(spk)
        # First element should spike (2.0 > 1.0), second should not
        assert spk[0, 0].item() == 1.0
        assert spk[1, 0].item() == 0.0

    def test_gradient_flow(self):
        """Verify gradients flow through the recurrent neuron."""
        fc = nn.Linear(4, 3)
        neuron = RLeaky(beta=0.9, learn_V=True)
        mx.eval(fc.parameters(), neuron.parameters())

        def loss_fn(params, x):
            fc.load_weights(list(params["fc"].items()))
            neuron.load_weights(list(params["neuron"].items()))
            state = neuron.init_state(batch_size=2, features=3)
            h = fc(x)
            spk, state = neuron(h, state)
            spk, state = neuron(h, state)
            return mx.mean(state["mem"])

        params = {"fc": dict(fc.parameters()), "neuron": dict(neuron.parameters())}
        x = mx.random.normal((2, 4))
        mx.eval(x)
        grads = mx.grad(loss_fn)(params, x)
        mx.eval(grads)
        assert any(
            mx.any(mx.abs(v) > 0).item()
            for v in grads["fc"].values()
            if isinstance(v, mx.array)
        )


class TestRSynaptic:
    """Tests for the RSynaptic (Recurrent Synaptic LIF) neuron."""

    def test_forward_shape(self):
        neuron = RSynaptic(alpha=0.8, beta=0.9)
        state = neuron.init_state(batch_size=4, features=128)
        x = mx.ones((4, 128))
        spk, new_state = neuron(x, state)
        mx.eval(spk)
        assert spk.shape == (4, 128)
        assert new_state["syn"].shape == (4, 128)
        assert new_state["mem"].shape == (4, 128)
        assert new_state["spk"].shape == (4, 128)

    def test_init_state_zeros(self):
        neuron = RSynaptic(alpha=0.8, beta=0.9)
        state = neuron.init_state(batch_size=2, features=64)
        mx.eval(state["syn"], state["mem"], state["spk"])
        for key in ["syn", "mem", "spk"]:
            assert mx.allclose(state[key], mx.zeros((2, 64))).item()

    def test_recurrence_effect(self):
        neuron = RSynaptic(alpha=0.0, beta=0.0, V=5.0)
        state = neuron.init_state(batch_size=1, features=1)
        x = mx.zeros((1, 1))

        # Inject a spike
        state["spk"] = mx.ones((1, 1))
        spk, state2 = neuron(x, state)
        mx.eval(spk, state2["syn"])
        # syn = 0*0 + 0 + 5*1 = 5, mem = 0*0 + 5 = 5 -> spike
        assert spk.item() == 1.0

    def test_learn_alpha_beta_V(self):
        neuron = RSynaptic(alpha=0.8, beta=0.9, V=1.0,
                           learn_alpha=True, learn_beta=True, learn_V=True)
        mx.eval(neuron.parameters())
        assert isinstance(neuron.alpha, mx.array)
        assert isinstance(neuron.beta, mx.array)
        assert isinstance(neuron.V, mx.array)

    def test_fixed_params(self):
        neuron = RSynaptic(alpha=0.8, beta=0.9, V=2.0)
        assert neuron._get_alpha() == 0.8
        assert neuron._get_beta() == 0.9
        assert neuron._get_V() == 2.0

    def test_gradient_flow(self):
        fc = nn.Linear(4, 3)
        neuron = RSynaptic(alpha=0.8, beta=0.9, learn_V=True)
        mx.eval(fc.parameters(), neuron.parameters())

        def loss_fn(params, x):
            fc.load_weights(list(params["fc"].items()))
            neuron.load_weights(list(params["neuron"].items()))
            state = neuron.init_state(batch_size=2, features=3)
            h = fc(x)
            spk, state = neuron(h, state)
            spk, state = neuron(h, state)
            return mx.mean(state["mem"])

        params = {"fc": dict(fc.parameters()), "neuron": dict(neuron.parameters())}
        x = mx.random.normal((2, 4))
        mx.eval(x)
        grads = mx.grad(loss_fn)(params, x)
        mx.eval(grads)
        assert any(
            mx.any(mx.abs(v) > 0).item()
            for v in grads["fc"].values()
            if isinstance(v, mx.array)
        )

    def test_multi_step(self):
        """Run multiple timesteps and verify state accumulation."""
        neuron = RSynaptic(alpha=0.8, beta=0.9)
        state = neuron.init_state(batch_size=2, features=8)
        x = mx.random.normal((2, 8)) * 0.3
        mx.eval(x)

        for _ in range(5):
            spk, state = neuron(x, state)
        mx.eval(spk, state["mem"])
        assert spk.shape == (2, 8)
