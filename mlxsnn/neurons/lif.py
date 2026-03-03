"""Leaky Integrate-and-Fire (LIF) neuron model.

Membrane dynamics (discrete-time):
    U[t+1] = beta * U[t] + X[t+1] - S[t] * threshold

where:
    U — membrane potential
    X — input current
    S — output spike (0 or 1)
    beta — membrane decay factor (0 < beta < 1)

Compatible with snnTorch.Leaky API.

References:
    Eshraghian, J. K., et al. (2023). Training Spiking Neural Networks
    Using Lessons From Deep Learning. Proceedings of the IEEE.
"""

import mlx.core as mx

from mlxsnn.neurons.base import SpikingNeuron


class Leaky(SpikingNeuron):
    """Leaky Integrate-and-Fire neuron.

    Args:
        beta: Membrane potential decay rate. Values closer to 1 give
            longer memory; closer to 0 gives faster decay.
        learn_beta: If True, beta becomes a learnable parameter.
        threshold: Spike threshold voltage.
        reset_mechanism: Reset method after spike ('subtract', 'zero', 'none').
        surrogate_fn: Surrogate gradient function name or callable.
        surrogate_scale: Scale parameter for surrogate gradient.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.neurons import Leaky
        >>> lif = Leaky(beta=0.9)
        >>> state = lif.init_state(batch_size=4, features=128)
        >>> x = mx.ones((4, 128))
        >>> spk, state = lif(x, state)
    """

    def __init__(
        self,
        beta: float = 0.9,
        learn_beta: bool = False,
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
        if learn_beta:
            self.beta = mx.array(beta)
        else:
            self._beta_const = beta

    def _get_beta(self):
        if hasattr(self, "_beta_const"):
            return self._beta_const
        return self.beta

    def init_state(self, batch_size: int, features: int) -> dict:
        """Initialize LIF neuron state.

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
        beta = self._get_beta()
        mem = beta * state["mem"] + x
        spk = self.fire(mem)
        mem = self.reset(mem, spk)
        return spk, {"mem": mem}
