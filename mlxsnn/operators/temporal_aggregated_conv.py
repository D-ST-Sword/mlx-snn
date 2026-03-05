"""Temporal Aggregated Convolution (TAC) for Spiking Neural Networks.

Exploits the linearity of convolution to reduce T conv calls to T/K by
aggregating K consecutive timesteps with exponential weighting before
applying a single spatial convolution per chunk.

Standard Conv+LIF (per layer, T timesteps):
    For t = 1, ..., T:
        I_t = W * X_t              # T conv calls
        U_t = beta * U_{t-1} + I_t
        S_t = Theta(U_t - V_th)
        U_t = U_t - S_t * V_th

TAC with chunk size K:
    For c = 0, ..., T/K - 1:
        X_agg = sum_{k=0}^{K-1} beta^{K-1-k} * X_{c*K+k}   # weighted sum
        I_c   = W * X_agg                                     # 1 conv per chunk
        U     = beta^K * U + I_c
        S     = Theta(U - V_th)
        U     = U - S * V_th

Speedup = K (conv calls reduced from T to T/K).

Error bound (subtract reset, firing rate rho):
    |U_exact - U_TAC| <= rho * V_th * beta / (1 - beta)

References:
    Linearity of convolution is a standard property. The temporal
    aggregation trick for SNNs is, to our knowledge, novel.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlxsnn.neurons.base import SpikingNeuron
from mlxsnn.layers._factory import create_neuron


class TemporalAggregatedConv(nn.Module):
    """Temporal Aggregated Convolution layer.

    Reduces the number of convolution calls by aggregating K consecutive
    input timesteps with exponential weighting, then applying a single
    spatial conv per chunk. Output has T/K timesteps.

    When K=1, this is equivalent to standard Conv+LIF.
    When K=T, this collapses the entire sequence into a single conv call
    (exact for output layers with reset_mechanism='none').

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial convolution kernel size.
        beta: Membrane decay factor (0 < beta < 1).
        chunk_size: Number of timesteps K to aggregate per conv call.
            If None, defaults to 1 (standard per-timestep conv).
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
        >>> layer = TemporalAggregatedConv(2, 16, 3, beta=0.9, chunk_size=4, padding=1)
        >>> state = layer.init_state(batch_size=2, spatial_shape=(8, 8))
        >>> x_seq = mx.ones((16, 2, 8, 8, 2))  # (T, B, H, W, C)
        >>> spk_seq, state = layer(x_seq, state)
        >>> spk_seq.shape  # (T/K, B, H', W', C_out)
        [4, 2, 8, 8, 16]
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
        """Initialize neuron state.

        Args:
            batch_size: Batch size.
            spatial_shape: (H', W') output spatial dims after conv.

        Returns:
            State dict with 'mem' of shape (B, H', W', C_out).
        """
        h, w = spatial_shape
        return self.neuron.init_state(batch_size, h, w, self._out_channels)

    def _aggregate_chunk(self, x_chunk: mx.array) -> mx.array:
        """Exponentially-weighted temporal aggregation of a chunk.

        Args:
            x_chunk: (K, B, H, W, C) input chunk.

        Returns:
            Aggregated input (B, H, W, C).
        """
        K = x_chunk.shape[0]
        beta = self._beta

        # Compute weights: beta^{K-1}, beta^{K-2}, ..., beta^0 = 1
        # x_agg = sum_{k=0}^{K-1} beta^{K-1-k} * x_chunk[k]
        x_agg = mx.zeros_like(x_chunk[0])
        for k in range(K):
            weight = beta ** (K - 1 - k)
            x_agg = x_agg + weight * x_chunk[k]
        return x_agg

    def __call__(
        self, x_seq: mx.array, state: dict,
    ) -> tuple[mx.array, dict]:
        """Forward pass over full temporal sequence.

        Aggregates every K timesteps and applies a single conv per chunk.

        Args:
            x_seq: Input sequence (T, B, H, W, C_in).
            state: Neuron state dict from previous call.

        Returns:
            Tuple of (spk_seq, new_state) where spk_seq has shape
            (T/K, B, H', W', C_out). If T is not divisible by K,
            the remaining timesteps form a smaller final chunk.
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
            out = self.conv(x_agg)
            if self.bn is not None:
                out = self.bn(out)

            # Membrane update: decay by beta^K, then add conv output.
            mem = (beta ** actual_k) * mem + out

            # Spike and reset.
            spk = self.neuron.fire(mem)
            mem = self.neuron.reset(mem, spk)
            spk_list.append(spk)

            c = end

        spk_seq = mx.stack(spk_list, axis=0)
        return spk_seq, {"mem": mem}

    def theoretical_error_bound(self, firing_rate: float) -> float:
        """Compute theoretical per-neuron error bound.

        Args:
            firing_rate: Average firing rate rho in [0, 1].

        Returns:
            Upper bound on |U_exact - U_TAC| per neuron.
        """
        beta = self._beta
        threshold = self.neuron._get_threshold()
        if beta >= 1.0:
            return float("inf")
        return firing_rate * threshold * beta / (1.0 - beta)
