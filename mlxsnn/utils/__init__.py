"""Utility functions for mlx-snn."""

from __future__ import annotations

import mlx.core as mx

from mlxsnn.neurons.base import SpikingNeuron


def reset_states(model) -> None:
    """Reset all stateful neuron layers in a model.

    Walks through the model's attributes and calls ``init_state``
    on any ``SpikingNeuron`` instance, but this function is provided
    primarily as a convenience reminder — in mlx-snn, state is
    always explicit (passed in and returned), so resetting simply
    means re-creating initial states via ``init_states``.

    Args:
        model: An ``nn.Module`` whose spiking neuron layers should
            be identified.

    Note:
        Since mlx-snn uses explicit state dicts rather than hidden
        instance variables, this function is a no-op convenience.
        Use ``init_states`` to create fresh state dicts instead.
    """
    pass  # State is explicit in mlx-snn; nothing to reset on the model.


def init_states(model, batch_size: int) -> dict:
    """Initialize hidden states for all spiking neuron layers.

    Walks through the model's named attributes and calls
    ``init_state`` on every ``SpikingNeuron`` found, inferring
    the feature count from a preceding ``Linear`` layer when
    possible.

    Args:
        model: An ``nn.Module`` containing spiking neuron layers.
        batch_size: Batch size for state tensors.

    Returns:
        A dictionary mapping attribute names to their initial
        state dicts (e.g., ``{"lif1": {"mem": ...}, ...}``).

    Examples:
        >>> import mlx.nn as nn
        >>> import mlxsnn
        >>> class SNN(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc1 = nn.Linear(784, 128)
        ...         self.lif1 = mlxsnn.Leaky(beta=0.9)
        ...     def __call__(self, x, states):
        ...         x = self.fc1(x)
        ...         spk, states["lif1"] = self.lif1(x, states["lif1"])
        ...         return spk, states
        >>> model = SNN()
        >>> states = mlxsnn.init_states(model, batch_size=32)
        >>> print(states["lif1"]["mem"].shape)
        [32, 128]
    """
    import mlx.nn as nn

    states = {}
    last_linear_out = None

    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(model, name)
        except (AttributeError, Exception):
            continue

        if isinstance(attr, nn.Linear):
            last_linear_out = attr.weight.shape[0]
        elif isinstance(attr, SpikingNeuron) and last_linear_out is not None:
            states[name] = attr.init_state(batch_size, last_linear_out)

    return states
