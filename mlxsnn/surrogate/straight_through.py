"""Straight-through estimator (STE) surrogate gradient.

Forward: Heaviside step function  H(x) = 1 if x >= 0 else 0
Backward: Passes gradient through unchanged where |x| <= 0.5/scale

Uses the STE pattern with a hard-tanh style clipped linear approximation.
"""

import mlx.core as mx


def straight_through_surrogate(scale: float = 1.0):
    """Create a straight-through estimator surrogate gradient function.

    Args:
        scale: Width of the pass-through window (gradient is scale where
            |x| <= 0.5/scale, zero otherwise).

    Returns:
        A callable with Heaviside forward and STE backward.
    """

    def forward(x):
        heaviside = mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        # Clamped linear: maps [-0.5/scale, 0.5/scale] -> [0, 1]
        approx = mx.clip(scale * x + 0.5, 0.0, 1.0)
        return mx.stop_gradient(heaviside - approx) + approx

    return forward
