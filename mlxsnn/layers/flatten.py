"""Flatten layer for connecting convolutional to fully-connected layers.

Converts spatial feature maps ``(B, H, W, C)`` to flat vectors
``(B, H*W*C)`` for use with ``nn.Linear`` and spiking neuron layers.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SpikingFlatten(nn.Module):
    """Flatten spatial dimensions of a spiking feature map.

    Reshapes ``(B, H, W, C)`` to ``(B, H*W*C)``.  This is a stateless
    layer that carries no learnable parameters.

    Args:
        start_dim: First dimension to flatten (default 1, preserving
            the batch dimension).

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.layers import SpikingFlatten
        >>> flat = SpikingFlatten()
        >>> x = mx.ones((4, 8, 8, 16))
        >>> out = flat(x)
        >>> out.shape
        [4, 1024]
    """

    def __init__(self, start_dim: int = 1):
        super().__init__()
        self._start_dim = start_dim

    def __call__(self, x: mx.array) -> mx.array:
        """Flatten the tensor from ``start_dim`` onward.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Flattened tensor with all dims from ``start_dim`` merged.
        """
        if self._start_dim == 0:
            return mx.reshape(x, (-1,))
        leading = x.shape[: self._start_dim]
        return mx.reshape(x, (*leading, -1))
