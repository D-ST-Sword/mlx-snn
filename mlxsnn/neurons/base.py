"""Abstract base class for all spiking neurons.

All neuron models in mlx-snn inherit from SpikingNeuron, which provides
the common interface for spike generation, reset mechanisms, and state
management.
"""

import mlx.core as mx
import mlx.nn as nn

from mlxsnn.surrogate import get_surrogate


class SpikingNeuron(nn.Module):
    """Abstract base class for spiking neuron models.

    Args:
        threshold: Membrane potential threshold for spike generation.
        learn_threshold: If True, threshold becomes a learnable parameter.
        reset_mechanism: How to reset membrane after a spike.
            'subtract' — subtract threshold from membrane potential.
            'zero' — reset membrane to zero.
            'none' — no reset (useful for output layers).
        surrogate_fn: Surrogate gradient function name or callable.
        surrogate_scale: Scale parameter for the surrogate gradient.

    Examples:
        Subclasses must implement ``init_state`` and ``__call__``::

            class MyNeuron(SpikingNeuron):
                def init_state(self, batch_size, features):
                    return {"mem": mx.zeros((batch_size, features))}

                def __call__(self, x, state):
                    mem = state["mem"] + x
                    spk = self.fire(mem)
                    mem = self.reset(mem, spk)
                    return spk, {"mem": mem}
    """

    def __init__(
        self,
        threshold: float = 1.0,
        learn_threshold: bool = False,
        reset_mechanism: str = "subtract",
        surrogate_fn: str = "fast_sigmoid",
        surrogate_scale: float = 25.0,
    ):
        super().__init__()
        if learn_threshold:
            self.threshold = mx.array(threshold)
        else:
            self._threshold_const = threshold
        self.reset_mechanism = reset_mechanism
        self._surrogate_fn = get_surrogate(surrogate_fn, surrogate_scale)

    def _get_threshold(self):
        """Return threshold as a float or tracked array."""
        if hasattr(self, "_threshold_const"):
            return self._threshold_const
        return self.threshold

    def init_state(self, batch_size: int, *args) -> dict:
        """Initialize neuron hidden state.

        Args:
            batch_size: Number of samples in the batch.
            *args: Additional shape dimensions (e.g., feature size).

        Returns:
            A dictionary of state tensors initialized to zeros.
        """
        raise NotImplementedError

    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        """Forward pass: compute spikes from input and previous state.

        Args:
            x: Input current of shape ``[batch, features]``.
            state: Dictionary of state tensors from the previous timestep.

        Returns:
            A tuple of ``(spikes, new_state)`` where spikes is a binary
            array of the same shape as x.
        """
        raise NotImplementedError

    def fire(self, mem: mx.array) -> mx.array:
        """Generate spikes using surrogate gradient.

        In the forward pass this applies the Heaviside step function.
        In the backward pass the surrogate gradient is used.

        Args:
            mem: Membrane potential array.

        Returns:
            Binary spike array (1 where mem >= threshold, 0 otherwise).
        """
        return self._surrogate_fn(mem - self._get_threshold())

    def reset(self, mem: mx.array, spk: mx.array) -> mx.array:
        """Apply reset mechanism after spike generation.

        Args:
            mem: Membrane potential before reset.
            spk: Binary spike array from ``fire()``.

        Returns:
            Membrane potential after reset.
        """
        if self.reset_mechanism == "subtract":
            return mem - spk * self._get_threshold()
        elif self.reset_mechanism == "zero":
            return mem * (1.0 - spk)
        return mem  # "none"
