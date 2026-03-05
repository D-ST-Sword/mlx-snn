"""Recurrent Leaky Integrate-and-Fire (RLeaky) neuron model.

Membrane dynamics (discrete-time, snnTorch-compatible):
    R[t]   = H(U[t] - threshold)       (detached reset from previous mem)
    U[t+1] = beta * U[t] + X[t+1] + V * S[t] - R[t] * threshold
    S[t+1] = H(U[t+1] - threshold)     (with surrogate gradient)

where:
    U — membrane potential
    X — input current
    S — output spike (0 or 1)
    R — reset signal (detached, based on previous membrane)
    V — recurrent weight (scalar or learnable)
    beta — membrane decay factor (0 < beta < 1)

The recurrent connection feeds the previous output spike back into
the neuron via a learnable weight V, enabling memory beyond what
the membrane decay alone provides.

Uses delayed reset (reset_delay=True) matching snnTorch's default:
the reset is computed from the previous membrane potential and
detached from the computation graph, preventing gradient flow
through the reset path.

Compatible with snnTorch.RLeaky API (all_to_all=False mode).

References:
    Eshraghian, J. K., et al. (2023). Training Spiking Neural Networks
    Using Lessons From Deep Learning. Proceedings of the IEEE.
"""

from __future__ import annotations

import mlx.core as mx

from mlxsnn.neurons.base import SpikingNeuron


class RLeaky(SpikingNeuron):
    """Recurrent Leaky Integrate-and-Fire neuron.

    Output spikes are fed back as additional input via a learnable
    recurrent weight V, giving the neuron an explicit recurrent
    connection in addition to the implicit memory from membrane
    decay.

    Uses snnTorch-compatible delayed reset: the reset signal is
    computed from the previous membrane potential and detached from
    the computation graph.

    Args:
        beta: Membrane potential decay rate.
        V: Recurrent weight. Scales the feedback spike signal.
        learn_beta: If True, beta becomes a learnable parameter.
        learn_V: If True, V becomes a learnable parameter.
        threshold: Spike threshold voltage.
        learn_threshold: If True, threshold becomes a learnable parameter.
        reset_mechanism: Reset method after spike ('subtract', 'zero', 'none').
        surrogate_fn: Surrogate gradient function name or callable.
        surrogate_scale: Scale parameter for surrogate gradient.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.neurons import RLeaky
        >>> neuron = RLeaky(beta=0.9, learn_V=True)
        >>> state = neuron.init_state(batch_size=4, features=128)
        >>> x = mx.ones((4, 128))
        >>> spk, state = neuron(x, state)
    """

    def __init__(
        self,
        beta: float = 0.9,
        V: float = 1.0,
        learn_beta: bool = False,
        learn_V: bool = False,
        threshold: float = 1.0,
        learn_threshold: bool = False,
        reset_mechanism: str = "subtract",
        surrogate_fn: str = "fast_sigmoid",
        surrogate_scale: float = 25.0,
    ):
        super().__init__(
            threshold=threshold,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            surrogate_fn=surrogate_fn,
            surrogate_scale=surrogate_scale,
        )

        if learn_beta:
            self.beta = mx.array(beta)
        else:
            self._beta_const = beta

        if learn_V:
            self.V = mx.array(V)
        else:
            self._V_const = V

    def _get_beta(self):
        """Return beta as a float or tracked array."""
        if hasattr(self, "_beta_const"):
            return self._beta_const
        return self.beta

    def _get_V(self):
        """Return V as a float or tracked array."""
        if hasattr(self, "_V_const"):
            return self._V_const
        return self.V

    def init_state(self, batch_size: int, features: int) -> dict:
        """Initialize RLeaky neuron state.

        Args:
            batch_size: Number of samples in the batch.
            features: Number of neuron features.

        Returns:
            State dict with 'mem' and 'spk' initialized to zeros.
        """
        shape = (batch_size, features)
        return {
            "mem": mx.zeros(shape),
            "spk": mx.zeros(shape),
        }

    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        """Forward one timestep.

        Uses delayed reset matching snnTorch: reset is computed from the
        previous membrane potential (detached from graph) before the
        membrane update.

        Args:
            x: Input current ``[batch, features]``.
            state: Dict with 'mem' and 'spk' from previous timestep.

        Returns:
            Tuple of (spikes, new_state).
        """
        beta = self._get_beta()
        V = self._get_V()
        threshold = self._get_threshold()

        # Delayed reset: compute from previous membrane, detach from graph
        reset = mx.stop_gradient(
            mx.where(state["mem"] >= threshold,
                     mx.ones_like(state["mem"]),
                     mx.zeros_like(state["mem"]))
        )

        # Membrane update with reset built in
        base = beta * state["mem"] + x + V * state["spk"]
        if self.reset_mechanism == "subtract":
            mem = base - reset * threshold
        elif self.reset_mechanism == "zero":
            mem = base * (1.0 - reset)
        else:  # "none"
            mem = base

        # Generate spikes with surrogate gradient
        spk = self.fire(mem)

        return spk, {"mem": mem, "spk": spk}
