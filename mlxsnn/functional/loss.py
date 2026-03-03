"""SNN-specific loss functions.

Loss functions designed for different spike decoding strategies:
rate coding, time-to-first-spike, and membrane potential readout.
"""

import mlx.core as mx
import mlx.nn as nn


def rate_coding_loss(spk_out: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss on spike counts (rate coding).

    Sums spikes across time to get a firing rate, then applies
    softmax cross-entropy against target labels.

    Args:
        spk_out: Output spikes ``[T, batch, num_classes]``.
        targets: Integer class labels ``[batch]``.

    Returns:
        Scalar loss value.
    """
    spike_count = mx.sum(spk_out, axis=0)  # [batch, num_classes]
    return mx.mean(nn.losses.cross_entropy(spike_count, targets))


def membrane_loss(mem: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss on final-timestep membrane potential.

    Uses the membrane potential at the last timestep as logits
    for classification.

    Args:
        mem: Membrane potentials ``[T, batch, num_classes]``.
        targets: Integer class labels ``[batch]``.

    Returns:
        Scalar loss value.
    """
    return mx.mean(nn.losses.cross_entropy(mem[-1], targets))


def mse_count_loss(spk_out: mx.array, targets: mx.array) -> mx.array:
    """MSE loss between spike counts and target counts.

    Useful for regression or when target firing rates are known.

    Args:
        spk_out: Output spikes ``[T, batch, num_classes]``.
        targets: Target spike counts ``[batch, num_classes]``.

    Returns:
        Scalar loss value.
    """
    spike_count = mx.sum(spk_out, axis=0)
    return mx.mean((spike_count - targets) ** 2)
