from __future__ import annotations

"""Direct and repeat spike encoding.

Simple encoding strategies that convert static data or repeat existing
spike patterns over time.
"""

import mlx.core as mx


def direct_encode(data: mx.array, num_steps: int) -> mx.array:
    """Repeat static data across multiple timesteps (direct encoding).

    Each timestep receives an identical copy of the input data.
    This is the simplest encoding — useful when the network itself
    should learn temporal dynamics from constant input.

    Args:
        data: Input data of shape ``[batch, ...]`` or ``[batch, C, H, W]``.
        num_steps: Number of timesteps ``T`` to repeat.

    Returns:
        Tensor of shape ``[T, batch, ...]`` (time-first).

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.encoding import direct_encode
        >>> data = mx.ones((4, 10))
        >>> out = direct_encode(data, num_steps=25)
        >>> out.shape
        [25, 4, 10]
    """
    # Use broadcast_to for zero-copy expansion
    return mx.broadcast_to(data[None], (num_steps,) + data.shape)


def repeat_encode(spikes: mx.array, num_repeats: int) -> mx.array:
    """Tile a spike pattern multiple times along the time axis.

    Concatenates ``num_repeats`` copies of the input along axis 0,
    useful for extending short spike sequences to fill a longer
    simulation window.

    Args:
        spikes: Spike tensor of shape ``[T, batch, ...]`` (time-first).
        num_repeats: Number of times to repeat the pattern.

    Returns:
        Tensor of shape ``[T * num_repeats, batch, ...]``.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.encoding import repeat_encode
        >>> spikes = mx.ones((5, 4, 10))
        >>> out = repeat_encode(spikes, num_repeats=3)
        >>> out.shape
        [15, 4, 10]
    """
    copies = [spikes] * num_repeats
    return mx.concatenate(copies, axis=0)
