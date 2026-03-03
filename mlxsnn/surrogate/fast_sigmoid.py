"""Fast sigmoid surrogate gradient.

Forward: Heaviside step function  H(x) = 1 if x >= 0 else 0
Backward: d/dx (0.5 * scale * x / (1 + scale * |x|) + 0.5)
        = scale / (2 * (1 + scale * |x|)^2)

This implements the *rational* fast sigmoid used by snnTorch, NOT the
standard sigmoid.  The rational form has heavier tails (polynomial vs
exponential decay) which provides more useful gradient signal to neurons
that are far from threshold.

Uses the straight-through estimator (STE) pattern:
    output = stop_gradient(heaviside - approx) + approx
This gives exact Heaviside in the forward pass while routing gradients
through the smooth approximation.

References:
    Zenke, F. & Vogels, T. P. (2021). The Remarkable Robustness of
    Surrogate Gradient Learning in Spiking Neural Networks. Neural
    Computation, 33(4), 899-925.

    Neftci, E. O., Mostafa, H. & Zenke, F. (2019). Surrogate Gradient
    Learning in Spiking Neural Networks. IEEE Signal Processing Magazine.
"""

import mlx.core as mx


def fast_sigmoid_surrogate(scale: float = 25.0):
    """Create a fast sigmoid surrogate gradient function.

    The smooth approximation is the rational fast sigmoid:

        approx(x) = 0.5 * scale * x / (1 + scale * |x|) + 0.5

    Its derivative (which becomes the surrogate gradient) is:

        d/dx approx = scale / (2 * (1 + scale * |x|)^2)

    This matches snnTorch's ``surrogate.fast_sigmoid(slope=scale)``.

    Args:
        scale: Controls the sharpness of the surrogate gradient.
            Larger values produce a sharper (more step-like) gradient.
            Default 25.0 matches snnTorch's default slope.

    Returns:
        A callable with Heaviside forward and fast-sigmoid backward.
    """

    def forward(x):
        heaviside = mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
        # Rational fast sigmoid: maps (-inf, inf) -> (0, 1)
        approx = 0.5 * scale * x / (1.0 + scale * mx.abs(x)) + 0.5
        return mx.stop_gradient(heaviside - approx) + approx

    return forward
