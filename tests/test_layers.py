"""Tests for composite spiking layers (conv, pooling, flatten)."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlxsnn.layers import (
    SpikingConv2d,
    SpikingMaxPool2d,
    SpikingAvgPool2d,
    SpikingFlatten,
    create_neuron,
)
from mlxsnn.neurons.base import SpikingNeuron
from mlxsnn.neurons.lif import Leaky
from mlxsnn.neurons.if_neuron import IF


# ---------------------------------------------------------------------------
# SpikingConv2d
# ---------------------------------------------------------------------------

class TestSpikingConv2d:
    """Tests for SpikingConv2d layer."""

    def test_forward_shape_nhwc(self):
        """Output should match NHWC layout (B, H', W', C_out)."""
        layer = SpikingConv2d(3, 16, kernel_size=3, padding=1)
        state = layer.init_state(batch_size=4, spatial_shape=(8, 8))
        x = mx.ones((4, 8, 8, 3))
        spk, new_state = layer(x, state)
        mx.eval(spk, new_state["mem"])
        assert spk.shape == (4, 8, 8, 16)

    def test_forward_shape_stride(self):
        """Stride should reduce spatial dimensions."""
        layer = SpikingConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        state = layer.init_state(batch_size=2, spatial_shape=(4, 4))
        x = mx.ones((2, 8, 8, 3))
        spk, new_state = layer(x, state)
        mx.eval(spk, new_state["mem"])
        assert spk.shape == (2, 4, 4, 32)

    def test_forward_shape_no_padding(self):
        """Without padding, spatial dims shrink by kernel_size - 1."""
        layer = SpikingConv2d(3, 8, kernel_size=3, padding=0)
        # 8x8 input with 3x3 kernel, no padding -> 6x6 output
        state = layer.init_state(batch_size=2, spatial_shape=(6, 6))
        x = mx.ones((2, 8, 8, 3))
        spk, new_state = layer(x, state)
        mx.eval(spk, new_state["mem"])
        assert spk.shape == (2, 6, 6, 8)

    def test_init_state_zeros(self):
        """Neuron state should be initialised to zeros."""
        layer = SpikingConv2d(3, 16, kernel_size=3, padding=1)
        state = layer.init_state(batch_size=4, spatial_shape=(8, 8))
        mx.eval(state["mem"])
        assert state["mem"].shape == (4, 8, 8, 16)
        assert mx.allclose(state["mem"], mx.zeros_like(state["mem"])).item()

    def test_temporal_loop(self):
        """Running multiple timesteps should update state correctly."""
        layer = SpikingConv2d(
            1, 4, kernel_size=3, padding=1,
            neuron_params={"beta": 0.9, "threshold": 0.5},
        )
        state = layer.init_state(batch_size=2, spatial_shape=(4, 4))
        total_spikes = 0

        for t in range(10):
            x = mx.ones((2, 4, 4, 1)) * 0.3
            spk, state = layer(x, state)
            mx.eval(spk, state["mem"])
            total_spikes = total_spikes + mx.sum(spk).item()

        # With constant input and beta=0.9, membrane should accumulate
        # and eventually spike.
        assert total_spikes > 0

    def test_no_bn(self):
        """Layer without batch norm should still work."""
        layer = SpikingConv2d(3, 16, kernel_size=3, padding=1, bn=False)
        assert layer.bn is None
        state = layer.init_state(batch_size=2, spatial_shape=(8, 8))
        x = mx.ones((2, 8, 8, 3))
        spk, new_state = layer(x, state)
        mx.eval(spk)
        assert spk.shape == (2, 8, 8, 16)

    def test_no_bias(self):
        """Layer without bias should still work."""
        layer = SpikingConv2d(3, 16, kernel_size=3, padding=1, bias=False)
        state = layer.init_state(batch_size=2, spatial_shape=(8, 8))
        x = mx.ones((2, 8, 8, 3))
        spk, new_state = layer(x, state)
        mx.eval(spk)
        assert spk.shape == (2, 8, 8, 16)

    def test_neuron_instance(self):
        """Passing a pre-built SpikingNeuron instance should work."""
        neuron = Leaky(beta=0.8, threshold=0.5)
        layer = SpikingConv2d(3, 16, kernel_size=3, padding=1, neuron=neuron)
        assert layer.neuron is neuron
        state = layer.init_state(batch_size=2, spatial_shape=(8, 8))
        x = mx.ones((2, 8, 8, 3))
        spk, _ = layer(x, state)
        mx.eval(spk)
        assert spk.shape == (2, 8, 8, 16)

    def test_if_neuron_type(self):
        """String 'if' should create an IF neuron."""
        layer = SpikingConv2d(3, 8, kernel_size=3, padding=1, neuron="if")
        assert isinstance(layer.neuron, IF)

    def test_spikes_are_binary(self):
        """Output spikes should only contain 0s and 1s."""
        layer = SpikingConv2d(1, 4, kernel_size=3, padding=1)
        state = layer.init_state(batch_size=2, spatial_shape=(4, 4))
        x = mx.random.normal((2, 4, 4, 1)) * 2.0
        spk, _ = layer(x, state)
        mx.eval(spk)
        unique_ok = mx.allclose(spk, mx.clip(spk, 0.0, 1.0)).item()
        is_binary = mx.allclose(spk, mx.round(spk)).item()
        assert unique_ok and is_binary

    def test_gradient_flow(self):
        """Gradients should flow through SpikingConv2d via surrogates."""
        layer = SpikingConv2d(
            1, 4, kernel_size=3, padding=1,
            neuron_params={"beta": 0.9},
        )

        def loss_fn(model, x, state):
            spk, _ = model(x, state)
            return mx.sum(spk)

        state = layer.init_state(batch_size=2, spatial_shape=(4, 4))
        x = mx.ones((2, 4, 4, 1))

        grad_fn = nn.value_and_grad(layer, loss_fn)
        loss, grads = grad_fn(layer, x, state)
        mx.eval(loss, grads)

        # At least some conv weight gradients should be non-zero.
        conv_grad = grads["conv"]["weight"]
        assert not mx.allclose(conv_grad, mx.zeros_like(conv_grad)).item()


# ---------------------------------------------------------------------------
# SpikingMaxPool2d
# ---------------------------------------------------------------------------

class TestSpikingMaxPool2d:
    """Tests for SpikingMaxPool2d."""

    def test_forward_shape(self):
        pool = SpikingMaxPool2d(kernel_size=2, stride=2)
        x = mx.ones((4, 8, 8, 16))
        out = pool(x)
        mx.eval(out)
        assert out.shape == (4, 4, 4, 16)

    def test_preserves_spikes(self):
        """Max pool on binary spikes: any spike in window is preserved."""
        pool = SpikingMaxPool2d(kernel_size=2, stride=2)
        x = mx.zeros((1, 4, 4, 1))
        # Set one spike in the first 2x2 window
        x = x.at[0, 0, 0, 0].add(1.0)
        out = pool(x)
        mx.eval(out)
        # The first pooled cell should be 1
        assert out[0, 0, 0, 0].item() == 1.0

    def test_default_stride(self):
        """When stride is None, it should default to kernel_size."""
        pool = SpikingMaxPool2d(kernel_size=2)
        x = mx.ones((2, 6, 6, 3))
        out = pool(x)
        mx.eval(out)
        assert out.shape == (2, 3, 3, 3)

    def test_with_padding(self):
        pool = SpikingMaxPool2d(kernel_size=3, stride=1, padding=1)
        x = mx.ones((2, 4, 4, 8))
        out = pool(x)
        mx.eval(out)
        assert out.shape == (2, 4, 4, 8)


# ---------------------------------------------------------------------------
# SpikingAvgPool2d
# ---------------------------------------------------------------------------

class TestSpikingAvgPool2d:
    """Tests for SpikingAvgPool2d."""

    def test_forward_shape(self):
        pool = SpikingAvgPool2d(kernel_size=2, stride=2)
        x = mx.ones((4, 8, 8, 16))
        out = pool(x)
        mx.eval(out)
        assert out.shape == (4, 4, 4, 16)

    def test_average_value(self):
        """Avg pool on all-ones should give 1.0."""
        pool = SpikingAvgPool2d(kernel_size=2, stride=2)
        x = mx.ones((1, 4, 4, 1))
        out = pool(x)
        mx.eval(out)
        assert mx.allclose(out, mx.ones((1, 2, 2, 1))).item()


# ---------------------------------------------------------------------------
# SpikingFlatten
# ---------------------------------------------------------------------------

class TestSpikingFlatten:
    """Tests for SpikingFlatten."""

    def test_basic_flatten(self):
        flat = SpikingFlatten()
        x = mx.ones((4, 8, 8, 16))
        out = flat(x)
        mx.eval(out)
        assert out.shape == (4, 8 * 8 * 16)

    def test_preserves_batch_dim(self):
        flat = SpikingFlatten(start_dim=1)
        x = mx.ones((2, 3, 4, 5))
        out = flat(x)
        mx.eval(out)
        assert out.shape == (2, 3 * 4 * 5)

    def test_start_dim_zero(self):
        flat = SpikingFlatten(start_dim=0)
        x = mx.ones((2, 3, 4))
        out = flat(x)
        mx.eval(out)
        assert out.shape == (2 * 3 * 4,)

    def test_start_dim_two(self):
        flat = SpikingFlatten(start_dim=2)
        x = mx.ones((2, 3, 4, 5))
        out = flat(x)
        mx.eval(out)
        assert out.shape == (2, 3, 4 * 5)


# ---------------------------------------------------------------------------
# create_neuron factory
# ---------------------------------------------------------------------------

class TestCreateNeuron:
    """Tests for the create_neuron factory function."""

    def test_leaky(self):
        n = create_neuron("leaky", beta=0.8)
        assert isinstance(n, Leaky)

    def test_lif_alias(self):
        n = create_neuron("lif")
        assert isinstance(n, Leaky)

    def test_if(self):
        n = create_neuron("if")
        assert isinstance(n, IF)

    def test_passthrough_instance(self):
        neuron = Leaky(beta=0.5)
        result = create_neuron(neuron)
        assert result is neuron

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown neuron type"):
            create_neuron("nonexistent")

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            create_neuron(42)


# ---------------------------------------------------------------------------
# Integration: multi-layer convolutional SNN
# ---------------------------------------------------------------------------

class TestConvSNNIntegration:
    """End-to-end test with a small conv SNN pipeline."""

    def test_conv_pool_flatten_pipeline(self):
        """Conv -> Pool -> Flatten should produce correct shapes."""
        conv = SpikingConv2d(1, 8, kernel_size=3, padding=1)
        pool = SpikingMaxPool2d(kernel_size=2, stride=2)
        flat = SpikingFlatten()

        # Input: (B=2, H=8, W=8, C=1)
        x = mx.random.normal((2, 8, 8, 1))
        # After conv: (2, 8, 8, 8)
        # After pool: (2, 4, 4, 8)
        # After flatten: (2, 128)

        state = conv.init_state(batch_size=2, spatial_shape=(8, 8))
        spk, state = conv(x, state)
        mx.eval(spk)
        assert spk.shape == (2, 8, 8, 8)

        pooled = pool(spk)
        mx.eval(pooled)
        assert pooled.shape == (2, 4, 4, 8)

        out = flat(pooled)
        mx.eval(out)
        assert out.shape == (2, 4 * 4 * 8)

    def test_two_conv_layers_temporal(self):
        """Two conv layers run over multiple timesteps."""
        conv1 = SpikingConv2d(1, 4, kernel_size=3, padding=1)
        pool1 = SpikingMaxPool2d(kernel_size=2, stride=2)
        conv2 = SpikingConv2d(4, 8, kernel_size=3, padding=1)

        state1 = conv1.init_state(batch_size=2, spatial_shape=(8, 8))
        state2 = conv2.init_state(batch_size=2, spatial_shape=(4, 4))

        for t in range(5):
            x = mx.random.normal((2, 8, 8, 1)) * 0.5
            spk1, state1 = conv1(x, state1)
            pooled = pool1(spk1)
            spk2, state2 = conv2(pooled, state2)
            mx.eval(spk1, spk2, state1["mem"], state2["mem"])

        assert spk2.shape == (2, 4, 4, 8)
