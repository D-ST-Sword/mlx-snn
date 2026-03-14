"""Fourier Temporal Convolution (FTC) for Spiking Neural Networks.

Replaces the fixed first-order IIR temporal dynamics of standard LIF
neurons with learnable second-order IIR (biquad) filters per output
channel. Each channel can learn its own temporal transfer function,
enabling resonance, bandpass, and multi-timescale behavior.

Standard LIF temporal dynamics (first-order IIR):
    U_t = beta * U_{t-1} + I_t
    H(z) = 1 / (1 - beta * z^{-1})

FTC biquad temporal dynamics (second-order IIR, per channel c):
    U_t^c = a1_c * U_{t-1}^c + a2_c * U_{t-2}^c + b0_c * I_t^c + b1_c * I_{t-1}^c
    H_c(z) = (b0_c + b1_c * z^{-1}) / (1 - a1_c * z^{-1} - a2_c * z^{-2})

Stability is guaranteed by parameterizing the poles in polar form:
    pole = r * exp(j*theta), with r = sigmoid(r_raw) < 1
    a1 = 2 * r * cos(theta)
    a2 = -r^2

This ensures both poles lie strictly inside the unit circle.

References:
    - Fang, W., et al. (2021). Incorporating Learnable Membrane Time
      Constants to Enhance Learning of SNNs. ICCV.
    - Gu, A., et al. (2022). Efficiently Modeling Long Sequences with
      Structured State Spaces. ICLR.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlxsnn.surrogate import get_surrogate


class FourierTemporalConv(nn.Module):
    """Fourier Temporal Convolution with learnable biquad IIR filters.

    Applies spatial Conv2d followed by per-channel second-order IIR
    temporal filtering and spiking nonlinearity. Each output channel
    has its own learnable temporal transfer function (4 parameters:
    r, theta, b0, b1).

    When filter_order=1, reduces to standard Conv+LIF with learnable beta.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Spatial convolution kernel size.
        filter_order: 1 for standard LIF, 2 for biquad IIR.
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
        >>> layer = FourierTemporalConv(2, 16, 3, filter_order=2, padding=1)
        >>> state = layer.init_state(batch_size=2, spatial_shape=(8, 8))
        >>> x_seq = mx.ones((10, 2, 8, 8, 2))  # (T, B, H, W, C)
        >>> spk_seq, state = layer(x_seq, state)
        >>> spk_seq.shape
        [10, 2, 8, 8, 16]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        filter_order: int = 2,
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
        assert filter_order in (1, 2), "filter_order must be 1 or 2"

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.bn = nn.BatchNorm(out_channels) if bn else None
        self._out_channels = out_channels
        self._filter_order = filter_order
        self._threshold = threshold
        self.reset_mechanism = reset_mechanism
        self._surrogate_fn = get_surrogate(surrogate_fn, surrogate_scale)

        # Learnable IIR parameters per channel.
        # Polar parameterization for stability: pole = r * exp(j*theta).
        # r_raw -> r = sigmoid(r_raw) ensures 0 < r < 1 (stable).
        # theta_raw -> theta = pi * sigmoid(theta_raw) ensures 0 < theta < pi.
        # Initialize r ~0.9 (long memory), theta ~pi/4 (moderate frequency).
        r_init = math.log(0.9 / (1.0 - 0.9))  # inverse sigmoid of 0.9
        theta_init = math.log(0.25 / 0.75)     # inverse sigmoid of 0.25 -> theta=pi/4

        self.r_raw = mx.full((out_channels,), r_init)
        self.theta_raw = mx.full((out_channels,), theta_init)

        # Feedforward coefficients (b0, b1).
        # Initialize b0=1, b1=0 so it starts as a pure autoregressive filter.
        self.b0 = mx.ones((out_channels,))
        if filter_order == 2:
            self.b1 = mx.zeros((out_channels,))

    def _get_coefficients(self):
        """Compute IIR coefficients from learnable parameters.

        Returns:
            (a1, a2, b0, b1) where a1/a2 are feedback and b0/b1 feedforward.
            For order=1: a2=0, b1=0 (reduces to first-order IIR).
        """
        r = mx.sigmoid(self.r_raw)       # (C,) in (0, 1)
        theta = math.pi * mx.sigmoid(self.theta_raw)  # (C,) in (0, pi)

        if self._filter_order == 2:
            a1 = 2.0 * r * mx.cos(theta)   # (C,)
            a2 = -(r * r)                    # (C,)
            return a1, a2, self.b0, self.b1
        else:
            # Order 1: single real pole at r.
            a1 = r                           # (C,) same as beta
            a2 = mx.zeros_like(r)
            b1 = mx.zeros_like(self.b0)
            return a1, a2, self.b0, b1

    def init_state(self, batch_size: int, spatial_shape: tuple) -> dict:
        """Initialize FTC state.

        Args:
            batch_size: Batch size.
            spatial_shape: (H', W') output spatial dims after conv.

        Returns:
            State dict with 'mem' and 'mem_prev' (for second-order),
            and 'input_prev' for feedforward delay.
        """
        h, w = spatial_shape
        shape = (batch_size, h, w, self._out_channels)
        state = {
            "mem": mx.zeros(shape),
            "input_prev": mx.zeros(shape),
        }
        if self._filter_order == 2:
            state["mem_prev"] = mx.zeros(shape)
        return state

    def __call__(
        self, x_seq: mx.array, state: dict,
    ) -> tuple[mx.array, dict]:
        """Forward pass over full temporal sequence.

        Args:
            x_seq: Input sequence (T, B, H, W, C_in).
            state: State dict from previous call.

        Returns:
            Tuple of (spk_seq, new_state) both with T timesteps.
        """
        T = x_seq.shape[0]
        a1, a2, b0, b1 = self._get_coefficients()

        mem = state["mem"]
        input_prev = state["input_prev"]
        mem_prev = state.get("mem_prev", mx.zeros_like(mem))

        spk_list = []

        for t in range(T):
            # Spatial convolution.
            cur_input = self.conv(x_seq[t])
            if self.bn is not None:
                cur_input = self.bn(cur_input)

            # Per-channel IIR temporal filtering.
            # U_t = a1 * U_{t-1} + a2 * U_{t-2} + b0 * I_t + b1 * I_{t-1}
            new_mem = (
                a1 * mem
                + a2 * mem_prev
                + b0 * cur_input
                + b1 * input_prev
            )

            # Spike generation and reset.
            spk = self._surrogate_fn(new_mem - self._threshold)

            if self.reset_mechanism == "subtract":
                new_mem = new_mem - spk * self._threshold
            elif self.reset_mechanism == "zero":
                new_mem = new_mem * (1.0 - spk)

            spk_list.append(spk)

            # Shift state.
            mem_prev = mem
            mem = new_mem
            input_prev = cur_input

        spk_seq = mx.stack(spk_list, axis=0)
        new_state = {"mem": mem, "input_prev": input_prev}
        if self._filter_order == 2:
            new_state["mem_prev"] = mem_prev
        return spk_seq, new_state

    def get_frequency_response(self, num_points: int = 256) -> tuple:
        """Compute the magnitude frequency response for each channel.

        Args:
            num_points: Number of frequency points.

        Returns:
            (freqs, magnitudes) where freqs is (num_points,) in [0, pi]
            and magnitudes is (C_out, num_points).
        """
        a1, a2, b0, b1 = self._get_coefficients()
        omega = mx.linspace(0.0, math.pi, num_points)  # (N,)

        # H(omega) = (b0 + b1 * exp(-j*omega)) / (1 - a1*exp(-j*omega) - a2*exp(-j*2*omega))
        # Compute numerator and denominator magnitudes.
        # MLX doesn't have complex numbers, compute magnitude directly.
        # |H|^2 = |numerator|^2 / |denominator|^2

        cos_w = mx.cos(omega)   # (N,)
        sin_w = mx.sin(omega)
        cos_2w = mx.cos(2.0 * omega)
        sin_2w = mx.sin(2.0 * omega)

        # Numerator: b0 + b1*cos(w) - j*b1*sin(w)
        # |num|^2 = (b0 + b1*cos(w))^2 + (b1*sin(w))^2
        # Expand: b0^2 + 2*b0*b1*cos(w) + b1^2
        b0_2d = mx.expand_dims(b0, axis=1)   # (C, 1)
        b1_2d = mx.expand_dims(b1, axis=1)
        cos_w_2d = mx.expand_dims(cos_w, axis=0)   # (1, N)
        sin_w_2d = mx.expand_dims(sin_w, axis=0)
        cos_2w_2d = mx.expand_dims(cos_2w, axis=0)
        sin_2w_2d = mx.expand_dims(sin_2w, axis=0)

        num_sq = b0_2d**2 + 2.0 * b0_2d * b1_2d * cos_w_2d + b1_2d**2

        # Denominator: 1 - a1*cos(w) - a2*cos(2w) - j*(-a1*sin(w) - a2*sin(2w))
        a1_2d = mx.expand_dims(a1, axis=1)
        a2_2d = mx.expand_dims(a2, axis=1)

        den_real = 1.0 - a1_2d * cos_w_2d - a2_2d * cos_2w_2d
        den_imag = a1_2d * sin_w_2d + a2_2d * sin_2w_2d
        den_sq = den_real**2 + den_imag**2

        magnitude = mx.sqrt(num_sq / (den_sq + 1e-10))  # (C, N)

        return omega, magnitude
