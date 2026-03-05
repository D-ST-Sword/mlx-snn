"""Straight-through estimator (STE) surrogate gradient.

Forward: Heaviside step function  H(x) = 1 if x >= 0 else 0
Backward: Passes gradient through unchanged (gradient = 1 everywhere)

This matches snnTorch's ``surrogate.straight_through_estimator()`` which
simply passes the incoming gradient through without modification.

The smooth approximation used is ``approx(x) = x + 0.5``, whose
derivative is 1 everywhere — the simplest possible surrogate.
"""

import mlx.core as mx


def straight_through_surrogate(scale: float = 1.0):
    """Create a straight-through estimator surrogate gradient function.

    The gradient is passed through unchanged regardless of the input value.
    This matches snnTorch's ``StraightThroughEstimator``.

    Args:
        scale: Unused. Kept for API consistency with other surrogates.

    Returns:
        A callable with Heaviside forward and identity backward.
    """

    def forward(x):
        heaviside = mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        # approx = x + 0.5 gives d(approx)/dx = 1 everywhere
        approx = x + 0.5
        return mx.stop_gradient(heaviside - approx) + approx

    return forward
