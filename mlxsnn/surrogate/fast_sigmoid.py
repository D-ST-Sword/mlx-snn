"""Fast sigmoid surrogate gradient.

Forward: Heaviside step function  H(x) = 1 if x >= 0 else 0
Backward: d/dx sigmoid(scale * x) = scale * sigmoid(scale*x) * (1 - sigmoid(scale*x))

Uses the straight-through estimator (STE) pattern:
    output = stop_gradient(heaviside - sigmoid) + sigmoid
This gives exact Heaviside in the forward pass while routing gradients
through the smooth sigmoid approximation.
"""

import mlx.core as mx


def fast_sigmoid_surrogate(scale: float = 25.0):
    """Create a fast sigmoid surrogate gradient function.

    Args:
        scale: Controls the sharpness of the surrogate gradient.
            Larger values produce a sharper (more step-like) gradient.

    Returns:
        A callable with Heaviside forward and sigmoid backward.
    """

    def forward(x):
        heaviside = mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        approx = mx.sigmoid(scale * x)
        return mx.stop_gradient(heaviside - approx) + approx

    return forward
