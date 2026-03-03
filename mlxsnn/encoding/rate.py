from __future__ import annotations

"""Rate coding (Poisson) spike encoding.

Maps continuous values in [0, 1] to spike trains where the probability
of spiking at each timestep equals the input value.

Higher input values produce higher firing rates.
"""

import mlx.core as mx


def rate_encode(
    data: mx.array,
    num_steps: int,
    gain: float = 1.0,
    offset: float = 0.0,
    key: mx.array | None = None,
) -> mx.array:
    """Encode continuous data into Poisson spike trains.

    Each input value is interpreted as a firing probability. At each
    timestep, a spike is generated with that probability.

    Args:
        data: Input data with values in [0, 1], shape ``[batch, ...]``.
        num_steps: Number of time steps to generate.
        gain: Multiplicative scaling applied to data before encoding.
        offset: Additive offset applied after gain.
        key: MLX random key. If None, uses default random state.

    Returns:
        Spike trains of shape ``[num_steps, batch, ...]`` (time-first).

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.encoding import rate_encode
        >>> data = mx.array([[0.8, 0.2], [0.5, 0.5]])
        >>> spikes = rate_encode(data, num_steps=100)
        >>> spikes.shape
        (100, 2, 2)
    """
    rates = mx.clip(data * gain + offset, 0.0, 1.0)

    # Generate random thresholds for each timestep
    shape = (num_steps,) + data.shape
    if key is not None:
        rand = mx.random.uniform(shape=shape, key=key)
    else:
        rand = mx.random.uniform(shape=shape)

    # Spike where random value < firing rate
    spikes = mx.where(rand < rates, mx.ones_like(rand), mx.zeros_like(rand))
    return spikes
