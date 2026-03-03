"""Adaptive Leaky Integrate-and-Fire (ALIF) neuron model.

Membrane dynamics (discrete-time):
    U[t+1] = beta * U[t] + X[t+1] - S[t] * threshold
    A[t+1] = rho * A[t] + S[t]
    thresh[t] = threshold + b * A[t]

where:
    U — membrane potential
    X — input current
    S — output spike (0 or 1)
    A — adaptation variable
    beta — membrane decay factor (0 < beta < 1)
    rho — adaptation decay factor (0 < rho < 1)
    b — adaptation strength
    threshold — base firing threshold

The adaptive threshold increases after each spike and decays back
to the base threshold over time, implementing spike-frequency
adaptation.

References:
    Bellec, G., et al. (2018). Long short-term memory and
    learning-to-learn in networks of spiking neurons. NeurIPS.
"""

import mlx.core as mx

from mlxsnn.neurons.base import SpikingNeuron


class ALIF(SpikingNeuron):
    """Adaptive Leaky Integrate-and-Fire neuron.

    Extends the standard LIF neuron with an adaptive threshold that
    increases after each spike and decays exponentially between spikes.
    This mechanism implements spike-frequency adaptation.

    Args:
        beta: Membrane potential decay rate. Values closer to 1 give
            longer memory; closer to 0 gives faster decay.
        rho: Adaptation variable decay rate. Controls how quickly the
            threshold relaxes back to its base value after a spike.
        b: Adaptation strength. Scales the contribution of the
            adaptation variable to the effective threshold.
        learn_beta: If True, beta becomes a learnable parameter.
        learn_rho: If True, rho becomes a learnable parameter.
        threshold: Base spike threshold voltage.
        reset_mechanism: Reset method after spike ('subtract', 'zero', 'none').
        surrogate_fn: Surrogate gradient function name or callable.
        surrogate_scale: Scale parameter for surrogate gradient.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.neurons import ALIF
        >>> neuron = ALIF(beta=0.9, rho=0.95, b=0.1)
        >>> state = neuron.init_state(batch_size=4, features=128)
        >>> x = mx.ones((4, 128))
        >>> spk, state = neuron(x, state)
    """

    def __init__(
        self,
        beta: float = 0.9,
        rho: float = 0.95,
        b: float = 0.1,
        learn_beta: bool = False,
        learn_rho: bool = False,
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
        self.b = b

        if learn_beta:
            self.beta = mx.array(beta)
        else:
            self._beta_const = beta

        if learn_rho:
            self.rho = mx.array(rho)
        else:
            self._rho_const = rho

    def _get_beta(self):
        """Return beta as a float or tracked array."""
        if hasattr(self, "_beta_const"):
            return self._beta_const
        return self.beta

    def _get_rho(self):
        """Return rho as a float or tracked array."""
        if hasattr(self, "_rho_const"):
            return self._rho_const
        return self.rho

    def init_state(self, batch_size: int, features: int) -> dict:
        """Initialize ALIF neuron state.

        Args:
            batch_size: Number of samples in the batch.
            features: Number of neuron features.

        Returns:
            State dict with 'mem' and 'adapt' initialized to zeros.
        """
        shape = (batch_size, features)
        return {
            "mem": mx.zeros(shape),
            "adapt": mx.zeros(shape),
        }

    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        """Forward one timestep.

        Args:
            x: Input current ``[batch, features]``.
            state: Dict with 'mem' and 'adapt' from previous timestep.

        Returns:
            Tuple of (spikes, new_state).
        """
        beta = self._get_beta()
        rho = self._get_rho()

        mem = beta * state["mem"] + x
        adapt = rho * state["adapt"]

        # Spike with adaptive threshold.
        effective_threshold = self.threshold + self.b * adapt
        spk = self._surrogate_fn(mem - effective_threshold)

        # Update adaptation variable with new spike.
        adapt = adapt + spk

        # Reset membrane potential only.
        mem = self.reset(mem, spk)

        return spk, {"mem": mem, "adapt": adapt}
