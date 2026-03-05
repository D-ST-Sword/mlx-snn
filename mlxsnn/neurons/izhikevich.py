"""Izhikevich spiking neuron model.

Two-dimensional dynamics (continuous-time):
    v' = 0.04*v^2 + 5*v + 140 - u + I
    u' = a*(b*v - u)

Spike condition and reset:
    if v >= 30 mV:
        v <- c
        u <- u + d

The model reproduces a wide range of cortical firing patterns depending
on the choice of parameters (a, b, c, d). Several standard presets are
provided via the ``PRESETS`` dictionary.

References:
    Izhikevich, E. M. (2003). Simple Model of Spiking Neurons. IEEE
    Transactions on Neural Networks, 14(6), 1569-1572.
"""

from __future__ import annotations

import mlx.core as mx

from mlxsnn.neurons.base import SpikingNeuron


PRESETS: dict[str, dict[str, float]] = {
    "RS": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
    "IB": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
    "CH": {"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},
    "FS": {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},
}


class Izhikevich(SpikingNeuron):
    """Izhikevich spiking neuron with two-dimensional dynamics.

    This model uses two coupled variables (v, u) to reproduce a rich
    repertoire of neural spiking behaviours while remaining
    computationally efficient.

    Parameters
    ----------
    a : float
        Time scale of the recovery variable *u*.  Smaller values result
        in slower recovery.
    b : float
        Sensitivity of the recovery variable *u* to sub-threshold
        fluctuations of the membrane potential *v*.
    c : float
        After-spike reset value of *v* (in mV).
    d : float
        After-spike increment applied to *u*.
    dt : float
        Integration timestep.  The default of 0.5 ms provides a good
        balance between accuracy and speed; use smaller values for
        higher fidelity.
    preset : str or None
        If given, must be a key in ``PRESETS`` (one of 'RS', 'IB',
        'CH', 'FS').  When a preset is specified the (a, b, c, d)
        values are loaded from the preset and any explicitly passed
        values for those parameters are ignored.
    surrogate_fn : str
        Surrogate gradient function name.
    surrogate_scale : float
        Scale parameter for the surrogate gradient.

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlxsnn.neurons.izhikevich import Izhikevich
    >>> neuron = Izhikevich(preset="RS")
    >>> state = neuron.init_state(batch_size=4, features=128)
    >>> x = mx.ones((4, 128)) * 10.0
    >>> spk, state = neuron(x, state)
    """

    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        dt: float = 0.5,
        preset: str | None = None,
        surrogate_fn: str = "fast_sigmoid",
        surrogate_scale: float = 0.1,
    ):
        # Threshold is fixed at 30 mV for the Izhikevich model, and the
        # reset is handled manually (not via the base-class ``reset``).
        super().__init__(
            threshold=30.0,
            learn_threshold=False,
            reset_mechanism="none",
            surrogate_fn=surrogate_fn,
            surrogate_scale=surrogate_scale,
        )

        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(
                    f"Unknown preset '{preset}'. "
                    f"Available: {list(PRESETS.keys())}"
                )
            params = PRESETS[preset]
            a = params["a"]
            b = params["b"]
            c = params["c"]
            d = params["d"]

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt

    def init_state(self, batch_size: int, features: int) -> dict:
        """Initialize Izhikevich neuron state.

        Parameters
        ----------
        batch_size : int
            Number of samples in the batch.
        features : int
            Number of neuron features.

        Returns
        -------
        dict
            State dictionary with keys ``'v'`` (membrane potential,
            initialised to -65.0 mV) and ``'u'`` (recovery variable,
            initialised to ``b * v``).
        """
        v = mx.full((batch_size, features), -65.0)
        u = mx.full((batch_size, features), self.b * -65.0)
        return {"v": v, "u": u}

    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        """Forward one timestep of the Izhikevich model.

        The membrane potential is integrated using a forward-Euler step
        with step size ``dt``.  When ``v >= 30`` a spike is emitted, *v*
        is reset to *c*, and *u* is incremented by *d*.

        Parameters
        ----------
        x : mx.array
            Input current of shape ``[batch, features]``.
        state : dict
            Dictionary with ``'v'`` and ``'u'`` from the previous
            timestep.

        Returns
        -------
        tuple[mx.array, dict]
            A tuple ``(spk, new_state)`` where *spk* is a binary spike
            array and *new_state* contains the updated ``'v'`` and
            ``'u'``.
        """
        v = state["v"]
        u = state["u"]

        # --- Euler integration of membrane and recovery dynamics ---
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + x
        du = self.a * (self.b * v - u)

        v = v + self.dt * dv
        u = u + self.dt * du

        # --- Spike detection via surrogate gradient ---
        spk = self.fire(v)

        # --- Custom reset: v -> c, u -> u + d where spiked ---
        v = mx.where(spk, self.c, v)
        u = mx.where(spk, u + self.d, u)

        return spk, {"v": v, "u": u}
