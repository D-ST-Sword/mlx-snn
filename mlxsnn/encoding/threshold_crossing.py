"""Threshold crossing spike encoding for continuous signals.

Emits a spike whenever the signal crosses a threshold level. Supports
multiple thresholds for multi-level encoding, and both upward and
downward crossings.

This is particularly suited for EEG signals where threshold crossings
at different amplitude levels capture different neural activity patterns.

Reference:
    Petro, B., et al. (2020). Selection and Optimization of Temporal Spike
    Encoding Methods for Spiking Neural Networks. IEEE TNNLS.
"""

from __future__ import annotations

import mlx.core as mx


def threshold_crossing_encode(
    data: mx.array,
    n_thresholds: int = 3,
    symmetric: bool = True,
) -> mx.array:
    """Encode continuous temporal signal via multi-level threshold crossings.

    Places ``n_thresholds`` evenly-spaced thresholds across the signal's
    amplitude range. A spike is emitted at each timestep where the signal
    crosses a threshold (upward or downward).

    Args:
        data: Temporal signal ``[T, channels]`` or ``[T, batch, channels]``.
            Should be z-score normalized (mean ~0, std ~1).
        n_thresholds: Number of positive threshold levels. With
            ``symmetric=True``, the same number of negative thresholds are
            added, giving ``2 * n_thresholds`` output channels per input
            channel.
        symmetric: If True, use both positive and negative thresholds.

    Returns:
        Spike trains ``[T, ..., channels * n_levels]`` where ``n_levels``
        is ``2 * n_thresholds`` if symmetric, else ``n_thresholds``.

    Examples:
        >>> import mlx.core as mx
        >>> signal = mx.random.normal((100, 18))  # T=100, 18 EEG channels
        >>> spikes = threshold_crossing_encode(signal, n_thresholds=3)
        >>> spikes.shape  # [100, 18*6] = [100, 108]
    """
    # Define threshold levels in units of standard deviation
    pos_thresholds = mx.linspace(0.5, 2.5, n_thresholds)
    if symmetric:
        thresholds = mx.concatenate([-pos_thresholds[::-1], pos_thresholds])
    else:
        thresholds = pos_thresholds

    n_levels = thresholds.shape[0]
    orig_shape = data.shape  # [T, ...channels]
    T = orig_shape[0]
    flat = data.reshape(T, -1)  # [T, C]
    C = flat.shape[1]

    # For each threshold, detect crossings
    all_crossings = []
    for i in range(n_levels):
        thr = thresholds[i]
        # Above/below threshold at each timestep
        above = flat >= thr  # [T, C]
        # Crossing = change from below to above or above to below
        crossing = above[1:] != above[:-1]  # [T-1, C]
        # Pad first timestep
        pad = mx.zeros((1, C), dtype=crossing.dtype)
        crossing = mx.concatenate([pad, crossing], axis=0)  # [T, C]
        all_crossings.append(crossing.astype(mx.float32))

    # Stack: [T, C, n_levels] → [T, C * n_levels]
    result = mx.concatenate(all_crossings, axis=-1)  # [T, C * n_levels]

    # Reshape to match original batch dims
    if data.ndim == 3:
        B = orig_shape[1]
        Ch = orig_shape[2]
        result = result.reshape(T, B, Ch * n_levels)
    elif data.ndim == 2:
        Ch = orig_shape[1]
        result = result.reshape(T, Ch * n_levels)

    return result
