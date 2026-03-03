"""User-definable surrogate gradient functions."""

import mlx.core as mx
from typing import Callable


def custom_surrogate(approx_fn: Callable[[mx.array], mx.array]):
    """Create a surrogate gradient from a user-defined smooth approximation.

    The forward pass always uses the Heaviside step function.
    The backward pass uses gradients from the provided smooth approximation.

    The approximation function should map real numbers to roughly [0, 1],
    approximating the Heaviside step function.

    Args:
        approx_fn: A smooth, differentiable function approximating
            the Heaviside step. Must be compatible with MLX autodiff.

    Returns:
        A callable with Heaviside forward and custom backward.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.surrogate.custom import custom_surrogate
        >>> # Sigmoid surrogate with custom scale
        >>> my_sg = custom_surrogate(lambda x: mx.sigmoid(50.0 * x))
    """

    def forward(x):
        heaviside = mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        approx = approx_fn(x)
        return mx.stop_gradient(heaviside - approx) + approx

    return forward
