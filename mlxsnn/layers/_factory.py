"""Neuron factory helper for layer constructors."""

from __future__ import annotations

from mlxsnn.neurons.base import SpikingNeuron


def create_neuron(neuron_type: str | SpikingNeuron, **kwargs) -> SpikingNeuron:
    """Create a spiking neuron from a type string or return an existing instance.

    Args:
        neuron_type: A string identifier (``'leaky'``, ``'if'``,
            ``'synaptic'``, ``'alpha'``, ``'alif'``, ``'izhikevich'``)
            or an already-constructed ``SpikingNeuron``.
        **kwargs: Keyword arguments forwarded to the neuron constructor
            when ``neuron_type`` is a string.

    Returns:
        A ``SpikingNeuron`` instance.

    Raises:
        ValueError: If ``neuron_type`` is an unrecognised string.
        TypeError: If ``neuron_type`` is neither a string nor a
            ``SpikingNeuron``.
    """
    if isinstance(neuron_type, SpikingNeuron):
        return neuron_type

    if not isinstance(neuron_type, str):
        raise TypeError(
            f"neuron_type must be a string or SpikingNeuron, got {type(neuron_type)}"
        )

    name = neuron_type.lower()

    # Lazy imports to avoid circular dependencies.
    if name == "leaky" or name == "lif":
        from mlxsnn.neurons.lif import Leaky
        return Leaky(**kwargs)
    elif name == "if":
        from mlxsnn.neurons.if_neuron import IF
        return IF(**kwargs)
    elif name == "synaptic":
        from mlxsnn.neurons.synaptic import Synaptic
        return Synaptic(**kwargs)
    elif name == "alpha":
        from mlxsnn.neurons.alpha import Alpha
        return Alpha(**kwargs)
    elif name == "alif" or name == "adaptive_lif":
        from mlxsnn.neurons.adaptive_lif import ALIF
        return ALIF(**kwargs)
    elif name == "izhikevich":
        from mlxsnn.neurons.izhikevich import Izhikevich
        return Izhikevich(**kwargs)
    else:
        raise ValueError(
            f"Unknown neuron type: '{neuron_type}'. "
            f"Choose from: 'leaky', 'if', 'synaptic', 'alpha', 'alif', 'izhikevich'."
        )
