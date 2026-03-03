"""Tests for spiking neuron models."""

import mlx
import mlx.core as mx
import pytest

from mlxsnn.neurons import Leaky, IF, SpikingNeuron


class TestLeakyNeuron:
    """Tests for the LIF (Leaky) neuron."""

    def test_forward_shape(self):
        lif = Leaky(beta=0.9)
        state = lif.init_state(batch_size=8, features=32)
        x = mx.ones((8, 32)) * 0.5
        spk, new_state = lif(x, state)
        mx.eval(spk, new_state["mem"])
        assert spk.shape == (8, 32)
        assert new_state["mem"].shape == (8, 32)

    def test_init_state_zeros(self):
        lif = Leaky(beta=0.9)
        state = lif.init_state(batch_size=4, features=16)
        mx.eval(state["mem"])
        assert mx.allclose(state["mem"], mx.zeros((4, 16))).item()

    def test_membrane_decay(self):
        """Membrane should decay by beta each step with no input."""
        lif = Leaky(beta=0.5, threshold=100.0)  # High threshold to prevent spikes
        state = {"mem": mx.ones((1, 1))}
        spk, state = lif(mx.zeros((1, 1)), state)
        mx.eval(state["mem"])
        assert mx.allclose(state["mem"], mx.array([[0.5]])).item()

    def test_spike_generation(self):
        """Neuron should spike when membrane exceeds threshold."""
        lif = Leaky(beta=0.9, threshold=1.0)
        state = {"mem": mx.zeros((1, 1))}
        # Input larger than threshold should cause spike
        spk, state = lif(mx.array([[2.0]]), state)
        mx.eval(spk)
        assert spk.item() == 1.0

    def test_no_spike_below_threshold(self):
        """Neuron should not spike when membrane is below threshold."""
        lif = Leaky(beta=0.0, threshold=1.0)
        state = {"mem": mx.zeros((1, 1))}
        spk, state = lif(mx.array([[0.5]]), state)
        mx.eval(spk)
        assert spk.item() == 0.0

    def test_subtract_reset(self):
        """After spike, membrane should be reduced by threshold."""
        lif = Leaky(beta=0.0, threshold=1.0, reset_mechanism="subtract")
        state = {"mem": mx.zeros((1, 1))}
        spk, state = lif(mx.array([[1.5]]), state)
        mx.eval(state["mem"])
        # mem = 1.5, spike = 1, reset: 1.5 - 1.0 = 0.5
        assert mx.allclose(state["mem"], mx.array([[0.5]])).item()

    def test_zero_reset(self):
        """After spike, membrane should be zeroed."""
        lif = Leaky(beta=0.0, threshold=1.0, reset_mechanism="zero")
        state = {"mem": mx.zeros((1, 1))}
        spk, state = lif(mx.array([[1.5]]), state)
        mx.eval(state["mem"])
        assert mx.allclose(state["mem"], mx.array([[0.0]])).item()

    def test_no_reset(self):
        """With 'none' reset, membrane should not be altered after spike."""
        lif = Leaky(beta=0.0, threshold=1.0, reset_mechanism="none")
        state = {"mem": mx.zeros((1, 1))}
        spk, state = lif(mx.array([[1.5]]), state)
        mx.eval(state["mem"])
        assert mx.allclose(state["mem"], mx.array([[1.5]])).item()

    def test_learnable_beta(self):
        """Learnable beta should be an MLX array (parameter)."""
        lif = Leaky(beta=0.9, learn_beta=True)
        # Just verify it's an mx.array, not a float
        assert isinstance(lif.beta, mx.array)

    def test_fixed_beta(self):
        """Fixed beta should be a Python float, not a parameter."""
        lif = Leaky(beta=0.9, learn_beta=False)
        assert lif._get_beta() == 0.9
        # Should have no trainable parameters
        params = lif.trainable_parameters()
        flat = mlx.utils.tree_flatten(params)
        assert len(flat) == 0

    def test_batch_independence(self):
        """Different batch elements should be processed independently."""
        lif = Leaky(beta=0.0, threshold=1.0)
        state = lif.init_state(batch_size=2, features=1)
        x = mx.array([[0.5], [1.5]])  # Only second element should spike
        spk, state = lif(x, state)
        mx.eval(spk)
        assert spk[0, 0].item() == 0.0
        assert spk[1, 0].item() == 1.0


class TestIFNeuron:
    """Tests for the IF (non-leaky) neuron."""

    def test_forward_shape(self):
        neuron = IF(threshold=1.0)
        state = neuron.init_state(batch_size=8, features=32)
        x = mx.ones((8, 32)) * 0.5
        spk, new_state = neuron(x, state)
        mx.eval(spk, new_state["mem"])
        assert spk.shape == (8, 32)
        assert new_state["mem"].shape == (8, 32)

    def test_no_decay(self):
        """IF neuron should accumulate without decay."""
        neuron = IF(threshold=100.0)  # High threshold
        state = {"mem": mx.ones((1, 1))}
        spk, state = neuron(mx.ones((1, 1)), state)
        mx.eval(state["mem"])
        # mem should be 1 + 1 = 2 (no decay)
        assert mx.allclose(state["mem"], mx.array([[2.0]])).item()

    def test_spike_and_reset(self):
        """IF should spike at threshold and reset correctly."""
        neuron = IF(threshold=1.0, reset_mechanism="subtract")
        state = {"mem": mx.zeros((1, 1))}
        # Step 1: mem = 0.5, no spike
        spk, state = neuron(mx.array([[0.5]]), state)
        mx.eval(spk, state["mem"])
        assert spk.item() == 0.0
        # Step 2: mem = 0.5 + 0.5 = 1.0, spike, reset to 0.0
        spk, state = neuron(mx.array([[0.5]]), state)
        mx.eval(spk, state["mem"])
        assert spk.item() == 1.0
        assert mx.allclose(state["mem"], mx.array([[0.0]])).item()

    def test_init_state_zeros(self):
        neuron = IF()
        state = neuron.init_state(batch_size=4, features=16)
        mx.eval(state["mem"])
        assert mx.allclose(state["mem"], mx.zeros((4, 16))).item()


class TestNumericalConsistency:
    """Test that neuron dynamics match the reference equations."""

    def test_lif_dynamics_equation(self):
        """Verify: U[t+1] = beta * U[t] + X[t+1] - S[t] * threshold."""
        beta = 0.8
        threshold = 1.0
        lif = Leaky(beta=beta, threshold=threshold)

        mem_prev = mx.array([[0.6]])
        x = mx.array([[0.7]])
        state = {"mem": mem_prev}
        spk, new_state = lif(x, state)
        mx.eval(spk, new_state["mem"])

        # Manual calculation
        mem_expected = beta * 0.6 + 0.7  # = 1.18
        spk_expected = 1.0  # 1.18 >= 1.0
        mem_after_reset = mem_expected - spk_expected * threshold  # 0.18

        assert spk.item() == spk_expected
        assert mx.allclose(
            new_state["mem"], mx.array([[mem_after_reset]]), atol=1e-5
        ).item()

    def test_if_dynamics_equation(self):
        """Verify: U[t+1] = U[t] + X[t+1] - S[t] * threshold."""
        threshold = 1.0
        neuron = IF(threshold=threshold)

        mem_prev = mx.array([[0.3]])
        x = mx.array([[0.5]])
        state = {"mem": mem_prev}
        spk, new_state = neuron(x, state)
        mx.eval(spk, new_state["mem"])

        # Manual: mem = 0.3 + 0.5 = 0.8, no spike
        assert spk.item() == 0.0
        assert mx.allclose(
            new_state["mem"], mx.array([[0.8]]), atol=1e-5
        ).item()

    def test_multi_step_lif(self):
        """Run LIF for multiple steps and verify accumulated behavior."""
        lif = Leaky(beta=0.9, threshold=1.0)
        state = lif.init_state(batch_size=1, features=1)

        total_spikes = 0
        for _ in range(20):
            x = mx.array([[0.3]])
            spk, state = lif(x, state)
            mx.eval(spk, state["mem"])
            total_spikes += spk.item()

        # With beta=0.9, constant input 0.3:
        # Membrane should reach ~0.3/(1-0.9) = 3.0 in steady state
        # but threshold=1.0 causes spikes, so we expect some spikes
        assert total_spikes > 0
