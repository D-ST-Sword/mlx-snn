"""Arctan surrogate gradient.

Forward: Heaviside step function  H(x) = 1 if x >= 0 else 0
Backward: d/dx (1/pi * arctan(alpha * x) + 0.5)

Uses the STE pattern with an arctan-based smooth approximation:
    approx(x) = 1/pi * arctan(alpha * x) + 0.5
"""

import mlx.core as mx


def arctan_surrogate(alpha: float = 2.0):
    """Create an arctan surrogate gradient function.

    Args:
        alpha: Controls the sharpness of the surrogate gradient.

    Returns:
        A callable with Heaviside forward and arctan backward.
    """

    def forward(x):
        heaviside = mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        approx = (1.0 / mx.pi) * mx.arctan(alpha * x) + 0.5
        return mx.stop_gradient(heaviside - approx) + approx

    return forward
