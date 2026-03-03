"""Delta modulation spike encoding.

Maps continuous signals to spike trains based on temporal changes.
A positive spike is emitted when the signal increases beyond a threshold,
and an optional negative (off) spike when it decreases beyond the threshold.

Reference:
    Akolkar, H., et al. (2015). What can neuromorphic event-driven precise
    timing add to spike rate coding? Neural Computation.
"""

import mlx.core as mx


def delta_encode(
    data: mx.array,
    threshold: float = 0.1,
    off_spike: bool = True,
    padding: bool = True,
) -> mx.array:
    """Encode continuous data using delta modulation.

    Computes temporal differences between consecutive timesteps and
    generates spikes where the absolute change exceeds a threshold.

    For single-step input (no time dimension), the data is returned
    as a single-timestep tensor with zero spikes (since no temporal
    difference can be computed).

    Args:
        data: Input data. Either single-step ``[batch, ...]`` or
            temporal ``[time, batch, ...]`` format.
        threshold: Minimum absolute change required to emit a spike.
            Larger values produce sparser spike trains.
        off_spike: If True, emit ``-1`` spikes when the signal
            decreases beyond ``-threshold``. If False, only positive
            spikes (``+1``) are generated.
        padding: If True, pad the first timestep with zeros so the
            output shape matches the input. If False, the output has
            one fewer timestep than the input.

    Returns:
        Spike array. For temporal input with ``padding=True``, shape
        matches ``data`` shape ``[time, batch, ...]``. With
        ``padding=False``, shape is ``[time - 1, batch, ...]``.
        For single-step input, returns ``[1, batch, ...]``.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.encoding import delta_encode
        >>> signal = mx.array([
        ...     [[0.0, 0.5]],
        ...     [[0.2, 0.3]],
        ...     [[0.5, 0.1]],
        ... ])  # shape [3, 1, 2]
        >>> spikes = delta_encode(signal, threshold=0.15)
        >>> spikes.shape
        [3, 1, 2]
    """
    # Handle single-step input: no temporal difference possible
    if data.ndim < 2 or (data.ndim >= 2 and _is_single_step(data)):
        return mx.expand_dims(mx.zeros_like(data), axis=0)

    # Temporal input: compute differences along time axis (axis 0)
    diff = data[1:] - data[:-1]

    # Positive spikes where change exceeds threshold
    on_spikes = mx.where(diff > threshold, mx.ones_like(diff), mx.zeros_like(diff))

    if off_spike:
        # Negative spikes where change is below -threshold
        off_spikes = mx.where(
            diff < -threshold,
            -mx.ones_like(diff),
            mx.zeros_like(diff),
        )
        spikes = on_spikes + off_spikes
    else:
        spikes = on_spikes

    if padding:
        # Pad first timestep with zeros to preserve shape
        pad_shape = (1,) + tuple(data.shape[1:])
        zeros = mx.zeros(pad_shape)
        spikes = mx.concatenate([zeros, spikes], axis=0)

    return spikes


def _is_single_step(data: mx.array) -> bool:
    """Check whether data is single-step (no time dimension).

    Heuristic: if the first dimension is 1, treat it as temporal with
    ``time=1``. Otherwise, assume data with ``ndim >= 2`` where the
    first axis has size > 1 is temporal. Single-step data should be
    explicitly expanded before calling ``delta_encode`` if it is
    genuinely ``[batch, ...]`` with ``batch > 1``.

    This is a private helper and not part of the public API.

    Args:
        data: Input array with at least 2 dimensions.

    Returns:
        True if the data appears to be single-step.
    """
    return data.shape[0] == 1
