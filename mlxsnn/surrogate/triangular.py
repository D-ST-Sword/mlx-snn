"""Triangular (tent) surrogate gradient.

Forward: Heaviside step function  H(x) = 1 if x >= 0 else 0
Backward: max(0, 1 - |x|)  — a triangle/tent centered at threshold

The gradient peaks at x=0 (the threshold boundary) and linearly decays
to zero at |x|=1.  This provides useful gradient signal near threshold
while ignoring neurons far from firing.

Note: snnTorch's ``triangular`` uses a sign-flipping gradient (+1 below,
-1 above threshold) which is numerically unstable.  Our implementation
uses a standard tent function which is the natural triangular pulse
commonly used in signal processing and spline interpolation.
"""

import mlx.core as mx


def triangular_surrogate(scale: float = 1.0):
    """Create a triangular (tent) surrogate gradient function.

    The backward pass uses a tent function centered at the threshold:
        grad = max(0, 1 - |x|)

    This provides a positive, localized gradient near threshold that
    linearly decays to zero, giving a good trade-off between signal
    quality and stability.

    Args:
        scale: Unused. Kept for API consistency with other surrogates.

    Returns:
        A callable with Heaviside forward and triangular backward.
    """

    def forward(x):
        heaviside = mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        # Tent-shaped smooth step: integral of max(0, 1-|x|)
        # For |x| > 1: approx = H(x) (0 or 1)
        # For -1 <= x < 0: approx = 0.5*(1+x)^2
        # For 0 <= x <= 1: approx = 1 - 0.5*(1-x)^2
        abs_x = mx.abs(x)
        quad_below = 0.5 * (1.0 + x) * (1.0 + x)  # 0.5*(1+x)^2
        quad_above = 1.0 - 0.5 * (1.0 - x) * (1.0 - x)  # 1 - 0.5*(1-x)^2
        approx = mx.where(
            abs_x >= 1.0,
            mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x)),
            mx.where(x >= 0, quad_above, quad_below),
        )
        return mx.stop_gradient(heaviside - approx) + approx

    return forward
