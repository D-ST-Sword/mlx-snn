"""Latency (time-to-first-spike) encoding.

Maps continuous values in [0, 1] to spike times — higher values spike
earlier. Each neuron fires exactly once. This is a deterministic encoding.

Reference:
    Park, S., et al. (2020). T2FSNN: Deep Spiking Neural Networks with
    Time-to-First-Spike Coding.
"""

import mlx.core as mx


def latency_encode(
    data: mx.array,
    num_steps: int,
    tau: float = 5.0,
    normalize: bool = True,
    linear: bool = False,
) -> mx.array:
    """Encode continuous data using time-to-first-spike latency coding.

    Higher input values produce earlier spikes. Each neuron fires
    exactly once across the time window.

    Args:
        data: Input data with values in [0, 1], shape ``[batch, ...]``.
        num_steps: Number of time steps in the encoding window.
        tau: Time constant controlling the mapping from value to spike
            time (only used when ``linear=False``).
        normalize: If True, normalize data to [0, 1] range.
        linear: If True, use linear mapping instead of exponential.

    Returns:
        Spike trains of shape ``[num_steps, batch, ...]`` (time-first).
        Each spatial position has exactly one spike.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.encoding import latency_encode
        >>> data = mx.array([[0.9, 0.1], [0.5, 0.5]])
        >>> spikes = latency_encode(data, num_steps=10)
        >>> spikes.shape
        (10, 2, 2)
    """
    if normalize:
        data = mx.clip(data, 0.0, 1.0)

    if linear:
        # Linear: high values -> early spike (low index)
        spike_times = mx.round((1.0 - data) * (num_steps - 1)).astype(mx.int32)
    else:
        # Exponential: spike_time = tau * log(data / (data - threshold))
        # Simplified: map through exponential decay
        eps = 1e-7
        data_clipped = mx.clip(data, eps, 1.0)
        spike_times = mx.round(
            tau * mx.log(1.0 / data_clipped)
        ).astype(mx.int32)
        spike_times = mx.clip(spike_times, 0, num_steps - 1)

    # Build one-hot spike trains along time axis
    time_indices = mx.arange(num_steps).reshape(
        (num_steps,) + (1,) * data.ndim
    )
    spike_times_expanded = mx.expand_dims(spike_times, axis=0)
    mask = (time_indices == spike_times_expanded)
    spikes = mx.where(mask, 1.0, 0.0)
    return spikes
