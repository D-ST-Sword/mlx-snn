"""Pure functions for spiking neuron membrane potential updates.

These functions implement single-timestep neuron dynamics without
any state management — suitable for use with mx.compile and vmap.
"""

import mlx.core as mx

from mlxsnn.surrogate import get_surrogate
from mlxsnn.functional.spike_ops import fire, reset_subtract, reset_zero


def lif_step(
    x: mx.array,
    mem: mx.array,
    beta: float = 0.9,
    threshold: float = 1.0,
    reset_mechanism: str = "subtract",
    surrogate_fn: str = "fast_sigmoid",
    surrogate_scale: float = 25.0,
) -> tuple[mx.array, mx.array]:
    """Single timestep of Leaky Integrate-and-Fire dynamics.

    Membrane dynamics:
        mem[t+1] = beta * mem[t] + x[t+1] - spk[t] * threshold

    Args:
        x: Input current ``[batch, features]``.
        mem: Previous membrane potential ``[batch, features]``.
        beta: Membrane decay factor.
        threshold: Spike threshold.
        reset_mechanism: 'subtract', 'zero', or 'none'.
        surrogate_fn: Surrogate gradient function name or callable.
        surrogate_scale: Scale for surrogate gradient.

    Returns:
        Tuple of (spikes, new_membrane_potential).
    """
    sg = get_surrogate(surrogate_fn, surrogate_scale)
    mem = beta * mem + x
    spk = fire(mem, threshold, sg)
    if reset_mechanism == "subtract":
        mem = reset_subtract(mem, spk, threshold)
    elif reset_mechanism == "zero":
        mem = reset_zero(mem, spk)
    return spk, mem


def if_step(
    x: mx.array,
    mem: mx.array,
    threshold: float = 1.0,
    reset_mechanism: str = "subtract",
    surrogate_fn: str = "fast_sigmoid",
    surrogate_scale: float = 25.0,
) -> tuple[mx.array, mx.array]:
    """Single timestep of Integrate-and-Fire dynamics (no leak).

    Membrane dynamics:
        mem[t+1] = mem[t] + x[t+1] - spk[t] * threshold

    Args:
        x: Input current ``[batch, features]``.
        mem: Previous membrane potential ``[batch, features]``.
        threshold: Spike threshold.
        reset_mechanism: 'subtract', 'zero', or 'none'.
        surrogate_fn: Surrogate gradient function name or callable.
        surrogate_scale: Scale for surrogate gradient.

    Returns:
        Tuple of (spikes, new_membrane_potential).
    """
    sg = get_surrogate(surrogate_fn, surrogate_scale)
    mem = mem + x
    spk = fire(mem, threshold, sg)
    if reset_mechanism == "subtract":
        mem = reset_subtract(mem, spk, threshold)
    elif reset_mechanism == "zero":
        mem = reset_zero(mem, spk)
    return spk, mem
