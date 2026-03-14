"""Spike-aware dropout layer.

Drops spikes (sets to zero) with a given probability during training.
Unlike standard dropout, no rescaling is applied because spikes are
binary — rescaling would violate the {0, 1} constraint.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SpikeDropout(nn.Module):
    """Drops spikes with probability ``p`` during training.

    During evaluation (``model.eval()``), all spikes pass through
    unchanged.  Unlike ``nn.Dropout``, no rescaling is applied:
    spikes are binary, so rescaling would produce non-binary values.

    Args:
        p: Probability of dropping each spike.  Must be in [0, 1).

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.layers import SpikeDropout
        >>> drop = SpikeDropout(p=0.3)
        >>> drop.train()
        >>> x = mx.ones((4, 10))
        >>> out = drop(x)
        >>> out.shape
        [4, 10]
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def __call__(self, x: mx.array) -> mx.array:
        """Apply spike dropout.

        Args:
            x: Spike tensor of any shape (typically binary {0, 1}).

        Returns:
            Tensor with spikes randomly zeroed during training.
        """
        if self.p == 0.0 or not self.training:
            return x
        mask = mx.random.bernoulli(1.0 - self.p, shape=x.shape)
        return x * mask
