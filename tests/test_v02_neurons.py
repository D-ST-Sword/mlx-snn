"""Tests for v0.2 neuron models: Izhikevich, ALIF, Synaptic, Alpha."""

import mlx.core as mx
import pytest

from mlxsnn.neurons import Izhikevich, ALIF, Synaptic, Alpha


class TestIzhikevich:

    def test_forward_shape(self):
        n = Izhikevich(preset="RS")
        state = n.init_state(batch_size=4, features=16)
        spk, state = n(mx.ones((4, 16)) * 15.0, state)
        mx.eval(spk, state["v"], state["u"])
        assert spk.shape == (4, 16)
        assert state["v"].shape == (4, 16)
        assert state["u"].shape == (4, 16)

    def test_init_state_resting(self):
        n = Izhikevich(preset="RS")
        state = n.init_state(batch_size=2, features=8)
        mx.eval(state["v"])
        assert mx.allclose(state["v"], mx.full((2, 8), -65.0)).item()

    def test_presets_exist(self):
        for preset in ["RS", "IB", "CH", "FS"]:
            n = Izhikevich(preset=preset)
            state = n.init_state(1, 4)
            spk, _ = n(mx.ones((1, 4)) * 10.0, state)
            mx.eval(spk)

    def test_spike_with_strong_input(self):
        n = Izhikevich(preset="RS")
        state = n.init_state(1, 1)
        # Drive it hard for many steps
        total = 0
        for _ in range(100):
            spk, state = n(mx.array([[40.0]]), state)
            mx.eval(spk, state["v"], state["u"])
            total += spk.item()
        assert total > 0, "Should spike with strong input"


class TestALIF:

    def test_forward_shape(self):
        n = ALIF(beta=0.9, rho=0.95, b=0.1)
        state = n.init_state(batch_size=4, features=16)
        spk, state = n(mx.ones((4, 16)), state)
        mx.eval(spk, state["mem"], state["adapt"])
        assert spk.shape == (4, 16)
        assert state["mem"].shape == (4, 16)
        assert state["adapt"].shape == (4, 16)

    def test_adaptation_increases_threshold(self):
        """After repeated spiking, adaptation should suppress further spikes."""
        n = ALIF(beta=0.0, rho=0.99, b=0.5, threshold=1.0)
        state = n.init_state(1, 1)
        # First spike should happen easily
        spk1, state = n(mx.array([[1.5]]), state)
        mx.eval(spk1, state["mem"], state["adapt"])
        assert spk1.item() == 1.0
        # After many spikes, adaptation should build up
        for _ in range(20):
            _, state = n(mx.array([[1.5]]), state)
            mx.eval(state["mem"], state["adapt"])
        # Adaptation should be positive
        assert state["adapt"].item() > 0

    def test_init_state_zeros(self):
        n = ALIF()
        state = n.init_state(2, 4)
        mx.eval(state["mem"], state["adapt"])
        assert mx.allclose(state["mem"], mx.zeros((2, 4))).item()
        assert mx.allclose(state["adapt"], mx.zeros((2, 4))).item()


class TestSynaptic:

    def test_forward_shape(self):
        n = Synaptic(alpha=0.9, beta=0.8)
        state = n.init_state(batch_size=4, features=16)
        spk, state = n(mx.ones((4, 16)), state)
        mx.eval(spk, state["syn"], state["mem"])
        assert spk.shape == (4, 16)
        assert state["syn"].shape == (4, 16)
        assert state["mem"].shape == (4, 16)

    def test_dual_decay(self):
        """Both syn and mem should decay with no input."""
        n = Synaptic(alpha=0.5, beta=0.5, threshold=100.0)
        state = {"syn": mx.ones((1, 1)), "mem": mx.ones((1, 1))}
        spk, state = n(mx.zeros((1, 1)), state)
        mx.eval(state["syn"], state["mem"])
        # syn = 0.5 * 1.0 + 0.0 = 0.5
        assert mx.allclose(state["syn"], mx.array([[0.5]]), atol=1e-5).item()

    def test_init_state_zeros(self):
        n = Synaptic()
        state = n.init_state(2, 4)
        mx.eval(state["syn"], state["mem"])
        assert mx.allclose(state["syn"], mx.zeros((2, 4))).item()
        assert mx.allclose(state["mem"], mx.zeros((2, 4))).item()


class TestAlpha:

    def test_forward_shape(self):
        n = Alpha(alpha=0.9, beta=0.8)
        state = n.init_state(batch_size=4, features=16)
        spk, state = n(mx.ones((4, 16)), state)
        mx.eval(spk)
        assert spk.shape == (4, 16)
        assert "syn_exc" in state
        assert "syn_inh" in state
        assert "mem" in state

    def test_three_state_vars(self):
        n = Alpha(alpha=0.5, beta=0.5, threshold=100.0)
        state = n.init_state(1, 2)
        spk, state = n(mx.ones((1, 2)), state)
        mx.eval(state["syn_exc"], state["syn_inh"], state["mem"])
        # syn_exc = 0.5 * 0 + 1 = 1
        assert mx.allclose(state["syn_exc"], mx.ones((1, 2)), atol=1e-5).item()

    def test_init_state_zeros(self):
        n = Alpha()
        state = n.init_state(2, 4)
        mx.eval(state["syn_exc"], state["syn_inh"], state["mem"])
        assert mx.allclose(state["syn_exc"], mx.zeros((2, 4))).item()
        assert mx.allclose(state["syn_inh"], mx.zeros((2, 4))).item()
        assert mx.allclose(state["mem"], mx.zeros((2, 4))).item()


class TestGradientFlowV02:
    """Ensure gradients flow through all new neuron models."""

    def test_izhikevich_gradient(self):
        n = Izhikevich(preset="RS")
        def loss(x):
            state = n.init_state(1, 2)
            total = mx.array(0.0)
            for t in range(10):
                spk, state = n(x, state)
                total = total + mx.sum(spk)
            return total
        g = mx.grad(loss)(mx.ones((1, 2)) * 20.0)
        mx.eval(g)
        assert mx.any(g != 0).item()

    def test_alif_gradient(self):
        n = ALIF(beta=0.9, rho=0.95, b=0.1)
        def loss(x):
            state = n.init_state(1, 2)
            spk, _ = n(x, state)
            return mx.sum(spk)
        g = mx.grad(loss)(mx.ones((1, 2)) * 1.5)
        mx.eval(g)
        assert mx.any(g != 0).item()

    def test_synaptic_gradient(self):
        n = Synaptic(alpha=0.9, beta=0.8)
        def loss(x):
            state = n.init_state(1, 2)
            spk, _ = n(x, state)
            return mx.sum(spk)
        g = mx.grad(loss)(mx.ones((1, 2)) * 1.5)
        mx.eval(g)
        assert mx.any(g != 0).item()

    def test_alpha_gradient(self):
        n = Alpha(alpha=0.9, beta=0.8)
        def loss(x):
            state = n.init_state(1, 2)
            spk, _ = n(x, state)
            return mx.sum(spk)
        g = mx.grad(loss)(mx.ones((1, 2)) * 1.5)
        mx.eval(g)
        assert mx.any(g != 0).item()
