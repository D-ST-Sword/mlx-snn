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


def ce_rate_loss(spk_out: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss on mean spike rate.

    Averages spikes over time to get a firing rate, then applies
    softmax cross-entropy against integer class labels.

    Args:
        spk_out: Output spikes ``[T, batch, num_classes]``.
        targets: Integer class labels ``[batch]``.

    Returns:
        Scalar loss value.
    """
    spike_rate = mx.mean(spk_out, axis=0)  # [batch, num_classes]
    return mx.mean(nn.losses.cross_entropy(spike_rate, targets))


def ce_count_loss(spk_out: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy loss on total spike count.

    Sums spikes over time, then applies softmax cross-entropy
    against integer class labels.

    Args:
        spk_out: Output spikes ``[T, batch, num_classes]``.
        targets: Integer class labels ``[batch]``.

    Returns:
        Scalar loss value.
    """
    spike_count = mx.sum(spk_out, axis=0)  # [batch, num_classes]
    return mx.mean(nn.losses.cross_entropy(spike_count, targets))


def mse_membrane_loss(
    mem: mx.array,
    targets: mx.array,
    on_target: float = 1.0,
    off_target: float = 0.0,
) -> mx.array:
    """MSE loss on membrane potential with one-hot target encoding.

    Creates a target tensor where the correct class has value
    ``on_target`` and all other classes have ``off_target``, then
    computes the mean squared error against the membrane potential.

    Args:
        mem: Membrane potential ``[batch, num_classes]`` at the last
            timestep.
        targets: Integer class labels ``[batch]``.
        on_target: Target value for the correct class.
        off_target: Target value for incorrect classes.

    Returns:
        Scalar loss value.
    """
    num_classes = mem.shape[-1]
    one_hot = mx.eye(num_classes)[targets]
    target_vals = one_hot * on_target + (1.0 - one_hot) * off_target
    return mx.mean((mem - target_vals) ** 2)


def spike_rate(spk_out: mx.array) -> mx.array:
    """Compute mean firing rate over time.

    Args:
        spk_out: Output spikes ``[T, batch, ...]``.

    Returns:
        Mean firing rate ``[batch, ...]``.
    """
    return mx.mean(spk_out, axis=0)


def spike_count(spk_out: mx.array) -> mx.array:
    """Compute total spike count over time.

    Args:
        spk_out: Output spikes ``[T, batch, ...]``.

    Returns:
        Total spike count ``[batch, ...]``.
    """
    return mx.sum(spk_out, axis=0)
