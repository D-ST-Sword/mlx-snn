"""Learnable Temporal Aggregated Convolution (L-TAC) for SNNs.

L-TAC replaces the fixed exponential weights beta^{K-1-k} in standard TAC
with learnable per-layer weights, initialized to exponential decay.

Standard TAC aggregation:
    X_agg = sum_{k=0}^{K-1} beta^{K-1-k} * X_{c*K+k}   (fixed weights)

L-TAC aggregation:
    w = softmax(alpha)                                      (learnable)
    X_agg = sum_{k=0}^{K-1} w_k * X_{c*K+k}

The softmax ensures weights are positive and normalized. Initialization
matches exponential decay: alpha_k = (K-1-k) * log(beta), so that
softmax(alpha) ≈ normalized(beta^{K-1-k}).

This allows each layer to learn its own optimal temporal weighting strategy
during training, potentially discovering that recent frames matter more
or less depending on the layer's role in the hierarchy.

Supports both standard TAC (output T/K) and TAC-TP (output T) modes.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlxsnn.layers._factory import create_neuron


class LearnableTAC(nn.Module):
    """Learnable Temporal Aggregated Convolution.

    Like standard TAC but with learnable aggregation weights per layer,
    initialized to exponential decay beta^{K-1-k}.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial convolution kernel size.
        beta: Membrane decay factor (0 < beta < 1).
        chunk_size: Number of timesteps K to aggregate per conv call.
        preserve_temporal: If True, use TAC-TP mode (output T timesteps).
            If False, use standard TAC mode (output T/K timesteps).
        stride: Convolution stride.
        padding: Convolution padding.
        bias: If True, add learnable bias to convolution.
        bn: If True, apply batch normalization.
        threshold: Spike threshold.
        reset_mechanism: 'subtract', 'zero', or 'none'.
        surrogate_fn: Surrogate gradient function name.
        surrogate_scale: Scale for surrogate gradient.

    Examples:
        >>> import mlx.core as mx
        >>> layer = LearnableTAC(2, 16, 3, beta=0.9, chunk_size=4,
        ...                      preserve_temporal=True, padding=1)
        >>> state = layer.init_state(batch_size=2, spatial_shape=(8, 8))
        >>> x_seq = mx.ones((16, 2, 8, 8, 2))  # (T, B, H, W, C)
        >>> spk_seq, state = layer(x_seq, state)
        >>> spk_seq.shape  # preserve_temporal=True → T preserved
        [16, 2, 8, 8, 16]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        beta: float = 0.9,
        chunk_size: int = 1,
        preserve_temporal: bool = True,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        bn: bool = False,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        surrogate_fn: str = "fast_sigmoid",
        surrogate_scale: float = 25.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.bn = nn.BatchNorm(out_channels) if bn else None
        self.neuron = create_neuron(
            "leaky",
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            surrogate_fn=surrogate_fn,
            surrogate_scale=surrogate_scale,
        )
        self._out_channels = out_channels
        self._beta = beta
        self._chunk_size = chunk_size
        self._preserve_temporal = preserve_temporal

        # Learnable aggregation weights (logits before softmax).
        # Initialize to match exponential decay: alpha_k = (K-1-k) * log(beta).
        K = chunk_size
        if beta > 0 and beta < 1:
            log_beta = math.log(beta)
            init_alpha = mx.array([float((K - 1 - k) * log_beta) for k in range(K)])
        else:
            init_alpha = mx.zeros((K,))
        self.agg_logits = init_alpha

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def agg_weights(self) -> mx.array:
        """Current aggregation weights (softmax of logits)."""
        return mx.softmax(self.agg_logits)

    def init_state(self, batch_size: int, spatial_shape: tuple) -> dict:
        """Initialize neuron state."""
        h, w = spatial_shape
        return self.neuron.init_state(batch_size, h, w, self._out_channels)

    def _aggregate_chunk(self, x_chunk: mx.array) -> mx.array:
        """Learnable weighted temporal aggregation of a chunk.

        Uses softmax(agg_logits) as weights. For partial chunks (last chunk
        when T is not divisible by K), uses the last actual_k weights
        renormalized via softmax.
        """
        actual_k = x_chunk.shape[0]
        K = self._chunk_size

        if actual_k == K:
            weights = mx.softmax(self.agg_logits)
        else:
            # Partial chunk: use last actual_k logits, renormalize
            weights = mx.softmax(self.agg_logits[:actual_k])

        # Weighted sum: weights shape (actual_k,), x_chunk shape (actual_k, B, H, W, C)
        x_agg = mx.zeros_like(x_chunk[0])
        for k in range(actual_k):
            x_agg = x_agg + weights[k] * x_chunk[k]
        return x_agg

    def __call__(
        self, x_seq: mx.array, state: dict,
    ) -> tuple[mx.array, dict]:
        """Forward pass.

        If preserve_temporal=True (TAC-TP mode): output has same T as input.
        If preserve_temporal=False (standard TAC mode): output has T/K timesteps.
        """
        T = x_seq.shape[0]
        K = self._chunk_size
        beta = self._beta

        spk_list = []
        mem = state["mem"]

        c = 0
        while c < T:
            end = min(c + K, T)
            chunk = x_seq[c:end]
            actual_k = end - c

            # Learnable aggregation
            x_agg = self._aggregate_chunk(chunk)

            # Single conv call
            conv_out = self.conv(x_agg)
            if self.bn is not None:
                conv_out = self.bn(conv_out)

            if self._preserve_temporal:
                # TAC-TP mode: run LIF K times with shared conv output
                for j in range(actual_k):
                    mem = beta * mem + conv_out
                    spk = self.neuron.fire(mem)
                    mem = self.neuron.reset(mem, spk)
                    spk_list.append(spk)
            else:
                # Standard TAC mode: single LIF step with beta^K decay
                mem = (beta ** actual_k) * mem + conv_out
                spk = self.neuron.fire(mem)
                mem = self.neuron.reset(mem, spk)
                spk_list.append(spk)

            c = end

        spk_seq = mx.stack(spk_list, axis=0)
        return spk_seq, {"mem": mem}

    def conv_calls(self, T: int) -> int:
        """Number of conv calls for a sequence of length T."""
        return math.ceil(T / self._chunk_size)
