"""Pooling layers for spiking networks.

Provides wrappers around MLX pooling operations that follow the mlx-snn
layer interface.  All layers expect NHWC format ``(B, H, W, C)``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SpikingMaxPool2d(nn.Module):
    """Max pooling that operates on spike tensors.

    Wraps ``mx.nn.MaxPool2d``.  Because spike tensors are binary
    (0 or 1), max pooling preserves a spike if *any* input in the
    pooling window fired.

    Input and output use NHWC layout: ``(B, H, W, C)``.

    Args:
        kernel_size: Size of the pooling window (int or tuple).
        stride: Stride of the pooling window.  Defaults to
            ``kernel_size`` when ``None``.
        padding: Zero-padding added to both sides (default 0).

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.layers import SpikingMaxPool2d
        >>> pool = SpikingMaxPool2d(kernel_size=2, stride=2)
        >>> x = mx.ones((4, 8, 8, 16))
        >>> out = pool(x)
        >>> out.shape
        [4, 4, 4, 16]
    """

    def __init__(
        self,
        kernel_size: int | tuple,
        stride: int | tuple | None = None,
        padding: int | tuple = 0,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply max pooling.

        Args:
            x: Input tensor ``(B, H, W, C)`` in NHWC format.

        Returns:
            Pooled tensor ``(B, H', W', C)``.
        """
        return self.pool(x)


class SpikingAvgPool2d(nn.Module):
    """Average pooling for spiking networks.

    Wraps ``mx.nn.AvgPool2d``.  When applied to binary spike tensors
    the output represents the local firing density within each pooling
    window.

    Input and output use NHWC layout: ``(B, H, W, C)``.

    Args:
        kernel_size: Size of the pooling window (int or tuple).
        stride: Stride of the pooling window.  Defaults to
            ``kernel_size`` when ``None``.
        padding: Zero-padding added to both sides (default 0).

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.layers import SpikingAvgPool2d
        >>> pool = SpikingAvgPool2d(kernel_size=2, stride=2)
        >>> x = mx.ones((4, 8, 8, 16))
        >>> out = pool(x)
        >>> out.shape
        [4, 4, 4, 16]
    """

    def __init__(
        self,
        kernel_size: int | tuple,
        stride: int | tuple | None = None,
        padding: int | tuple = 0,
    ):
        super().__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply average pooling.

        Args:
            x: Input tensor ``(B, H, W, C)`` in NHWC format.

        Returns:
            Pooled tensor ``(B, H', W', C)``.
        """
        return self.pool(x)
