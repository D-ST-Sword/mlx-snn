"""Information-Maximizing Spike Convolution (IMC) for Spiking Neural Networks.

Applies information-theoretic principles to optimize channel allocation
in convolutional SNN layers. Each output channel has a learnable gate
alpha_c in [0, 1] that modulates its contribution. An information
bottleneck regularization loss penalizes layers whose output information
capacity is below the input information content.

Information capacity of a spiking layer:
    C_layer = C_out * H_out * W_out * T * H(rho)

where H(rho) = -rho * log2(rho) - (1-rho) * log2(1-rho) is the
binary entropy of firing rate rho.

Channel-width sufficiency theorem:
    C_out_min = ceil(C_in * H(rho_in) / H(rho_out) * r_spatial)

where r_spatial = (H_in * W_in) / (H_out * W_out).

The InfoMax loss encourages the network to maintain sufficient
information capacity while pruning redundant channels via L1
regularization on the channel gates.

References:
    - Tishby, N. & Zaslavsky, N. (2015). Deep Learning and the
      Information Bottleneck Principle.
    - Kundu, S., et al. (2021). Spike-Thrift: Towards Energy-Efficient
      Deep Spiking Neural Networks.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlxsnn.neurons.base import SpikingNeuron
from mlxsnn.layers._factory import create_neuron


def binary_entropy(rho: mx.array) -> mx.array:
    """Binary entropy H(rho) in bits.

    Args:
        rho: Firing rate(s) in (0, 1). Clamped to avoid log(0).

    Returns:
        H(rho) = -rho*log2(rho) - (1-rho)*log2(1-rho).
    """
    eps = 1e-7
    p = mx.clip(rho, eps, 1.0 - eps)
    return -(p * mx.log2(p) + (1.0 - p) * mx.log2(1.0 - p))


class InfoMaxSpikeConv(nn.Module):
    """Information-Maximizing Spike Convolution layer.

    Standard Conv2d + spiking neuron with per-channel learnable gates
    and information-theoretic regularization. The gates alpha_c
    modulate output spikes per channel, enabling differentiable
    channel pruning during training.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial convolution kernel size.
        beta: Membrane decay factor for the LIF neuron.
        info_reg_weight: Lambda for information bottleneck regularization.
        gate_reg_weight: L1 regularization weight on channel gates.
        ema_decay: Decay factor for running firing rate estimation.
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
        >>> layer = InfoMaxSpikeConv(2, 16, 3, info_reg_weight=0.01, padding=1)
        >>> state = layer.init_state(batch_size=2, spatial_shape=(8, 8))
        >>> x = mx.ones((2, 8, 8, 2))
        >>> spk, state = layer(x, state)
        >>> loss = layer.info_loss()
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        beta: float = 0.9,
        info_reg_weight: float = 0.01,
        gate_reg_weight: float = 0.001,
        ema_decay: float = 0.99,
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
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._info_reg_weight = info_reg_weight
        self._gate_reg_weight = gate_reg_weight
        self._ema_decay = ema_decay

        # Per-channel gates, initialized to 1 (all channels active).
        # Parameterized as raw logits, passed through sigmoid.
        self.gate_logits = mx.full((out_channels,), 3.0)  # sigmoid(3) ~ 0.95

        # Running estimates of firing rates (not learnable).
        self._firing_rate_out = mx.full((out_channels,), 0.1)
        self._firing_rate_in = mx.array(0.1)

    def _get_gates(self) -> mx.array:
        """Per-channel gate values in [0, 1]."""
        return mx.sigmoid(self.gate_logits)

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

    def __call__(
        self, x: mx.array, state: dict,
    ) -> tuple[mx.array, dict]:
        """Forward one timestep with channel gating.

        Args:
            x: Input tensor (B, H, W, C_in).
            state: Neuron state dict.

        Returns:
            Tuple of (gated_spikes, new_state).
        """
        # Update input firing rate estimate (detached from grad graph).
        input_rate = mx.stop_gradient(mx.mean(mx.abs(x) > 0.5))
        self._firing_rate_in = mx.stop_gradient(
            self._ema_decay * self._firing_rate_in
            + (1.0 - self._ema_decay) * input_rate
        )

        # Spatial convolution.
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)

        # Spiking neuron.
        spk, new_state = self.neuron(out, state)

        # Apply per-channel gates.
        gates = self._get_gates()  # (C_out,)
        gated_spk = spk * gates

        # Update per-channel firing rate estimates (detached from grad graph).
        # Mean over batch and spatial dims: (C_out,)
        per_channel_rate = mx.stop_gradient(mx.mean(spk, axis=(0, 1, 2)))
        self._firing_rate_out = mx.stop_gradient(
            self._ema_decay * self._firing_rate_out
            + (1.0 - self._ema_decay) * per_channel_rate
        )

        return gated_spk, new_state

    def info_loss(self) -> mx.array:
        """Information bottleneck regularization loss.

        Penalizes the layer if output information capacity is below
        the input information content. Also applies L1 regularization
        on channel gates to encourage sparsity.

        Returns:
            Scalar loss value.
        """
        gates = self._get_gates()

        # Effective number of output channels.
        effective_c_out = mx.sum(gates)

        # Per-channel output entropy (averaged across channels).
        rho_out = mx.clip(self._firing_rate_out, 1e-6, 1.0 - 1e-6)
        h_out_per_channel = binary_entropy(rho_out)  # (C_out,)
        # Weighted by gates.
        total_output_capacity = mx.sum(gates * h_out_per_channel)

        # Input information estimate.
        rho_in = mx.clip(self._firing_rate_in, 1e-6, 1.0 - 1e-6)
        h_in = binary_entropy(rho_in)
        total_input_info = self._in_channels * h_in

        # Information bottleneck penalty: penalize when capacity < demand.
        info_penalty = mx.maximum(
            mx.array(0.0),
            total_input_info - total_output_capacity,
        )

        # L1 gate sparsity.
        gate_penalty = mx.mean(gates)

        loss = (
            self._info_reg_weight * info_penalty
            + self._gate_reg_weight * gate_penalty
        )
        return loss

    def effective_channels(self, threshold: float = 0.5) -> int:
        """Count channels with gate > threshold.

        Args:
            threshold: Gate threshold for considering a channel active.

        Returns:
            Number of active channels.
        """
        gates = self._get_gates()
        return int(mx.sum(gates > threshold).item())

    def channel_utilization(self) -> dict:
        """Get channel utilization statistics.

        Returns:
            Dict with gate stats, firing rates, and entropy values.
        """
        gates = self._get_gates()
        mx.eval(gates, self._firing_rate_out, self._firing_rate_in)
        return {
            "gate_mean": float(mx.mean(gates).item()),
            "gate_min": float(mx.min(gates).item()),
            "gate_max": float(mx.max(gates).item()),
            "effective_channels": self.effective_channels(),
            "total_channels": self._out_channels,
            "mean_firing_rate": float(mx.mean(self._firing_rate_out).item()),
            "input_firing_rate": float(self._firing_rate_in.item()),
        }

    @staticmethod
    def minimum_channels(
        c_in: int,
        rho_in: float,
        rho_out: float,
        spatial_reduction: float = 1.0,
    ) -> int:
        """Compute minimum channels for information preservation.

        Args:
            c_in: Input channels.
            rho_in: Input firing rate.
            rho_out: Target output firing rate.
            spatial_reduction: H_in*W_in / (H_out*W_out).

        Returns:
            Minimum number of output channels.
        """
        h_in = -rho_in * math.log2(rho_in) - (1 - rho_in) * math.log2(1 - rho_in)
        h_out = -rho_out * math.log2(rho_out) - (1 - rho_out) * math.log2(1 - rho_out)
        if h_out < 1e-10:
            return c_in * 10  # Degenerate case
        return math.ceil(c_in * h_in / h_out * spatial_reduction)
