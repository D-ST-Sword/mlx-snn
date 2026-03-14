"""Convolutional spiking layers.

Combines ``mx.nn.Conv2d`` with optional batch normalization and a spiking
neuron for temporal processing of spatial feature maps.  MLX uses NHWC
format throughout: input ``(B, H, W, C_in)`` and output ``(B, H', W', C_out)``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlxsnn.layers._factory import create_neuron
from mlxsnn.neurons.base import SpikingNeuron


class SpikingConv2d(nn.Module):
    """Conv2d + optional BatchNorm + spiking neuron.

    Wraps ``mx.nn.Conv2d`` with a spiking neuron so that each forward
    call processes a single timestep.  The neuron state must be carried
    across timesteps by the caller.

    MLX Conv2d uses NHWC layout:
        input  ``(B, H, W, C_in)``  ->  output ``(B, H', W', C_out)``

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Convolution kernel size (int or tuple).
        stride: Convolution stride (default 1).
        padding: Convolution padding (default 0).
        bias: If True, add a learnable bias to the convolution.
        bn: If True, apply batch normalization before the neuron.
        neuron: Neuron type string (``'leaky'``, ``'if'``, ``'synaptic'``)
            or a pre-constructed ``SpikingNeuron`` instance.
        neuron_params: Dict of keyword arguments forwarded to the neuron
            constructor (e.g. ``{'beta': 0.95, 'threshold': 0.5}``).

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.layers import SpikingConv2d
        >>> layer = SpikingConv2d(3, 16, kernel_size=3, padding=1)
        >>> state = layer.init_state(batch_size=4, spatial_shape=(8, 8))
        >>> x = mx.ones((4, 8, 8, 3))
        >>> spk, state = layer(x, state)
        >>> spk.shape
        [4, 8, 8, 16]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        bias: bool = True,
        bn: bool = True,
        neuron: str | SpikingNeuron = "leaky",
        neuron_params: dict | None = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, bias=bias,
        )
        self.bn = nn.BatchNorm(out_channels) if bn else None
        self.neuron = create_neuron(neuron, **(neuron_params or {}))
        self._out_channels = out_channels

    def init_state(self, batch_size: int, spatial_shape: tuple) -> dict:
        """Initialize neuron state for spatial feature maps.

        Args:
            batch_size: Number of samples in the batch.
            spatial_shape: ``(H', W')`` spatial dimensions **after**
                the convolution (accounting for stride and padding).

        Returns:
            State dict whose tensors have shape
            ``(B, H', W', C_out)`` in NHWC format.
        """
        h, w = spatial_shape
        return self.neuron.init_state(batch_size, h, w, self._out_channels)

    def __call__(
        self, x: mx.array, state: dict,
    ) -> tuple[mx.array, dict]:
        """Forward one timestep.

        Args:
            x: Input tensor ``(B, H, W, C_in)`` in NHWC format.
            state: Neuron state dict from the previous timestep.

        Returns:
            Tuple of ``(spikes, new_state)`` where spikes has shape
            ``(B, H', W', C_out)``.
        """
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)

        # Neuron ops are elementwise — no need to flatten spatial dims.
        spk, new_state = self.neuron(out, state)
        return spk, new_state
