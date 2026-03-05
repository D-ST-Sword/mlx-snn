"""Sigmoid surrogate gradient.

Forward: Heaviside step function  H(x) = 1 if x >= 0 else 0
Backward: sigmoid(k*x) * (1 - sigmoid(k*x))

Uses the STE pattern with a sigmoid-based smooth approximation:
    approx(x) = sigmoid(k * x)
"""

import mlx.core as mx


def sigmoid_surrogate(slope: float = 25.0):
    """Create a sigmoid surrogate gradient function.

    The backward pass uses the derivative of the sigmoid function,
    which has a bell-shaped curve centered at the threshold.

    Args:
        slope: Controls the steepness of the sigmoid. Larger values
            produce a sharper transition.

    Returns:
        A callable with Heaviside forward and sigmoid backward.
    """

    def forward(x):
        heaviside = mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        approx = mx.sigmoid(slope * x)
        return mx.stop_gradient(heaviside - approx) + approx

    return forward
