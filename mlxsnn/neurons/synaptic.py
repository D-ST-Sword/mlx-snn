"""Synaptic (conductance-based) Leaky Integrate-and-Fire neuron model.

Membrane dynamics (discrete-time):
    I[t+1] = alpha * I[t] + X[t+1]
    U[t+1] = beta * U[t] + I[t+1] - S[t] * threshold

where:
    I — synaptic current
    U — membrane potential
    X — input current
    S — output spike (0 or 1)
    alpha — synaptic current decay factor (0 < alpha < 1)
    beta — membrane potential decay factor (0 < beta < 1)

The two-state formulation models a first-order synaptic current that
is filtered before reaching the membrane, producing smoother temporal
dynamics than a single-state LIF neuron.

Compatible with snnTorch.Synaptic API.

References:
    Eshraghian, J. K., et al. (2023). Training Spiking Neural Networks
    Using Lessons From Deep Learning. Proceedings of the IEEE.
"""

import mlx.core as mx

from mlxsnn.neurons.base import SpikingNeuron


class Synaptic(SpikingNeuron):
    """Synaptic (conductance-based) LIF neuron.

    Extends the standard LIF model with an explicit synaptic current
    state variable. Input is first integrated into the synaptic current,
    which then drives the membrane potential. This two-state model
    captures more realistic post-synaptic dynamics.

    Args:
        alpha: Synaptic current decay rate. Values closer to 1 give
            slower synaptic dynamics; closer to 0 gives faster decay.
        beta: Membrane potential decay rate. Values closer to 1 give
            longer memory; closer to 0 gives faster decay.
        learn_alpha: If True, alpha becomes a learnable parameter.
        learn_beta: If True, beta becomes a learnable parameter.
        threshold: Spike threshold voltage.
        reset_mechanism: Reset method after spike ('subtract', 'zero', 'none').
        surrogate_fn: Surrogate gradient function name or callable.
        surrogate_scale: Scale parameter for surrogate gradient.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.neurons import Synaptic
        >>> neuron = Synaptic(alpha=0.8, beta=0.9)
        >>> state = neuron.init_state(batch_size=4, features=128)
        >>> x = mx.ones((4, 128))
        >>> spk, state = neuron(x, state)
    """

    def __init__(
        self,
        alpha: float = 0.8,
        beta: float = 0.9,
        learn_alpha: bool = False,
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

        if learn_alpha:
            self.alpha = mx.array(alpha)
        else:
            self._alpha_const = alpha

        if learn_beta:
            self.beta = mx.array(beta)
        else:
            self._beta_const = beta

    def _get_alpha(self):
        """Return alpha as a float or tracked array."""
        if hasattr(self, "_alpha_const"):
            return self._alpha_const
        return self.alpha

    def _get_beta(self):
        """Return beta as a float or tracked array."""
        if hasattr(self, "_beta_const"):
            return self._beta_const
        return self.beta

    def init_state(self, batch_size: int, features: int) -> dict:
        """Initialize Synaptic neuron state.

        Args:
            batch_size: Number of samples in the batch.
            features: Number of neuron features.

        Returns:
            State dict with 'syn' and 'mem' initialized to zeros.
        """
        shape = (batch_size, features)
        return {
            "syn": mx.zeros(shape),
            "mem": mx.zeros(shape),
        }

    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        """Forward one timestep.

        Args:
            x: Input current ``[batch, features]``.
            state: Dict with 'syn' and 'mem' from previous timestep.

        Returns:
            Tuple of (spikes, new_state).
        """
        alpha = self._get_alpha()
        beta = self._get_beta()

        syn = alpha * state["syn"] + x
        mem = beta * state["mem"] + syn

        spk = self.fire(mem)
        mem = self.reset(mem, spk)

        return spk, {"syn": syn, "mem": mem}
