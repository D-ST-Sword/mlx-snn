"""Spike generation and reset operations.

Low-level operations for spike generation (with surrogate gradients)
and membrane reset. These are building blocks for neuron models.
"""

import mlx.core as mx
from typing import Callable


def fire(
    mem: mx.array,
    threshold: float,
    surrogate_fn: Callable[[mx.array], mx.array],
) -> mx.array:
    """Generate spikes using surrogate gradient function.

    Args:
        mem: Membrane potential.
        threshold: Spike threshold.
        surrogate_fn: Surrogate gradient function (Heaviside forward,
            smooth backward).

    Returns:
        Binary spike array.
    """
    return surrogate_fn(mem - threshold)


def reset_subtract(
    mem: mx.array,
    spk: mx.array,
    threshold: float,
) -> mx.array:
    """Subtract-reset: reduce membrane by threshold where spike occurred.

    Args:
        mem: Membrane potential before reset.
        spk: Binary spike array.
        threshold: Value to subtract.

    Returns:
        Membrane potential after reset.
    """
    return mem - spk * threshold


def reset_zero(mem: mx.array, spk: mx.array) -> mx.array:
    """Zero-reset: set membrane to zero where spike occurred.

    Args:
        mem: Membrane potential before reset.
        spk: Binary spike array.

    Returns:
        Membrane potential after reset.
    """
    return mem * (1.0 - spk)
