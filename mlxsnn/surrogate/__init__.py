"""Surrogate gradient functions for spiking neural networks.

Surrogate gradients replace the non-differentiable Heaviside step function
used in spike generation with smooth approximations during the backward pass.

References:
    Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate Gradient
    Learning in Spiking Neural Networks. IEEE Signal Processing Magazine.
"""

from mlxsnn.surrogate.fast_sigmoid import fast_sigmoid_surrogate
from mlxsnn.surrogate.arctan import arctan_surrogate
from mlxsnn.surrogate.straight_through import straight_through_surrogate
from mlxsnn.surrogate.sigmoid import sigmoid_surrogate
from mlxsnn.surrogate.triangular import triangular_surrogate

_SURROGATE_REGISTRY = {
    "fast_sigmoid": fast_sigmoid_surrogate,
    "arctan": arctan_surrogate,
    "straight_through": straight_through_surrogate,
    "sigmoid": sigmoid_surrogate,
    "triangular": triangular_surrogate,
}


def get_surrogate(name: str, scale: float = 25.0):
    """Get a surrogate gradient function by name.

    Args:
        name: Name of the surrogate function. One of
            'fast_sigmoid', 'arctan', 'straight_through'.
        scale: Scaling parameter controlling gradient sharpness.

    Returns:
        A callable that computes the Heaviside step in the forward pass
        and a smooth surrogate gradient in the backward pass.

    Raises:
        ValueError: If name is not in the registry.

    Available surrogates:
        'fast_sigmoid', 'arctan', 'straight_through', 'sigmoid',
        'triangular'.
    """
    if callable(name):
        return name
    if name not in _SURROGATE_REGISTRY:
        raise ValueError(
            f"Unknown surrogate '{name}'. "
            f"Available: {list(_SURROGATE_REGISTRY.keys())}"
        )
    return _SURROGATE_REGISTRY[name](scale)
