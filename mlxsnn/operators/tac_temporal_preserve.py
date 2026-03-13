"""TAC with Temporal Preservation (TAC-TP) for Spiking Neural Networks.

TAC-TP aggregates K consecutive input frames for convolution (reducing
conv calls by K×) but runs LIF membrane dynamics for each of the K
timesteps using the shared conv output. This preserves the full temporal
dimension T in the output, unlike standard TAC which reduces T to T/K.

Standard TAC:
    For each chunk c of K timesteps:
        X_agg = sum_{k=0}^{K-1} beta^{K-1-k} * X_{c*K+k}
        I = W * X_agg                  # 1 conv call
        U = beta^K * U + I             # 1 membrane update
        S = Theta(U - V_th)            # 1 spike
    Output: T/K timesteps

TAC-TP:
    For each chunk c of K timesteps:
        X_agg = sum_{k=0}^{K-1} beta^{K-1-k} * X_{c*K+k}
        I = W * X_agg                  # 1 conv call (shared)
        For j = 0, ..., K-1:           # K membrane updates
            U = beta * U + I
            S_j = Theta(U - V_th)
            U = reset(U, S_j)
    Output: T timesteps (preserved!)

Speedup: K× reduction in conv calls (same as TAC).
Temporal resolution: Full T preserved (same as baseline).
Error: Each of the K LIF steps uses the same conv output (averaged
    over K input frames), losing per-frame conv detail.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlxsnn.neurons.base import SpikingNeuron
from mlxsnn.layers._factory import create_neuron


class TACTemporalPreserve(nn.Module):
    """TAC with Temporal Preservation.

    Aggregates K input frames for a single conv call but preserves all T
    output timesteps by running LIF dynamics K times per chunk with the
    shared conv output.

    When K=1, this is exactly standard Conv+LIF.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial convolution kernel size.
        beta: Membrane decay factor (0 < beta < 1).
        chunk_size: Number of timesteps K to aggregate per conv call.
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
        >>> layer = TACTemporalPreserve(2, 16, 3, beta=0.9, chunk_size=4, padding=1)
        >>> state = layer.init_state(batch_size=2, spatial_shape=(8, 8))
        >>> x_seq = mx.ones((16, 2, 8, 8, 2))  # (T, B, H, W, C)
        >>> spk_seq, state = layer(x_seq, state)
        >>> spk_seq.shape  # (T, B, H', W', C_out) — T preserved!
        [16, 2, 8, 8, 16]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        beta: float = 0.9,
        chunk_size: int = 1,
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

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def init_state(self, batch_size: int, spatial_shape: tuple) -> dict:
        """Initialize neuron state."""
        h, w = spatial_shape
        return self.neuron.init_state(batch_size, h, w, self._out_channels)

    def _aggregate_chunk(self, x_chunk: mx.array) -> mx.array:
        """Exponentially-weighted temporal aggregation of a chunk."""
        K = x_chunk.shape[0]
        beta = self._beta
        x_agg = mx.zeros_like(x_chunk[0])
        for k in range(K):
            weight = beta ** (K - 1 - k)
            x_agg = x_agg + weight * x_chunk[k]
        return x_agg

    def __call__(
        self, x_seq: mx.array, state: dict,
    ) -> tuple[mx.array, dict]:
        """Forward pass preserving full temporal dimension.

        Args:
            x_seq: Input sequence (T, B, H, W, C_in).
            state: Neuron state dict.

        Returns:
            Tuple of (spk_seq, new_state) where spk_seq has shape
            (T, B, H', W', C_out) — same T as input.
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

            # Aggregate chunk inputs with exponential weighting.
            x_agg = self._aggregate_chunk(chunk)

            # Single conv call for the entire chunk.
            conv_out = self.conv(x_agg)
            if self.bn is not None:
                conv_out = self.bn(conv_out)

            # Run LIF dynamics K times with shared conv output.
            for j in range(actual_k):
                mem = beta * mem + conv_out
                spk = self.neuron.fire(mem)
                mem = self.neuron.reset(mem, spk)
                spk_list.append(spk)

            c = end

        spk_seq = mx.stack(spk_list, axis=0)
        return spk_seq, {"mem": mem}

    def conv_calls(self, T: int) -> int:
        """Number of conv calls for a sequence of length T."""
        import math
        return math.ceil(T / self._chunk_size)
