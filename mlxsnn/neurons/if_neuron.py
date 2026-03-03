"""Integrate-and-Fire (IF) neuron model (non-leaky).

Membrane dynamics (discrete-time):
    U[t+1] = U[t] + X[t+1] - S[t] * threshold

This is equivalent to LIF with beta=1 (no decay).

where:
    U — membrane potential
    X — input current
    S — output spike (0 or 1)
"""

import mlx.core as mx

from mlxsnn.neurons.base import SpikingNeuron


class IF(SpikingNeuron):
    """Integrate-and-Fire neuron (no leak).

    This is the simplest spiking neuron model. The membrane potential
    integrates input current without decay and fires when threshold
    is reached.

    Args:
        threshold: Spike threshold voltage.
        reset_mechanism: Reset method after spike ('subtract', 'zero', 'none').
        surrogate_fn: Surrogate gradient function name or callable.
        surrogate_scale: Scale parameter for surrogate gradient.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.neurons import IF
        >>> neuron = IF(threshold=1.0)
        >>> state = neuron.init_state(batch_size=4, features=128)
        >>> x = mx.ones((4, 128)) * 0.5
        >>> spk, state = neuron(x, state)  # No spike yet (mem=0.5)
        >>> spk, state = neuron(x, state)  # Spike! (mem=1.0 >= threshold)
    """

    def __init__(
        self,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        surrogate_fn: str = "fast_sigmoid",
        surrogate_scale: float = 25.0,
    ):
        super().__init__(
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            surrogate_fn=surrogate_fn,
            surrogate_scale=surrogate_scale,
        )

    def init_state(self, batch_size: int, features: int) -> dict:
        """Initialize IF neuron state.

        Args:
            batch_size: Number of samples in the batch.
            features: Number of neuron features.

        Returns:
            State dict with 'mem' initialized to zeros.
        """
        return {"mem": mx.zeros((batch_size, features))}

    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        """Forward one timestep.

        Args:
            x: Input current ``[batch, features]``.
            state: Dict with 'mem' from previous timestep.

        Returns:
            Tuple of (spikes, new_state).
        """
        mem = state["mem"] + x
        spk = self.fire(mem)
        mem = self.reset(mem, spk)
        return spk, {"mem": mem}
