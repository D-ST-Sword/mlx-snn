"""EEG-to-spike encoding for spiking neural networks.

Provides methods for converting raw EEG signals (continuous voltage traces)
into discrete spike trains suitable for SNN processing. Supports rate coding,
delta modulation, and multi-level threshold crossing strategies.

Designed for standard EEG data shapes produced by libraries such as MNE-Python.

Reference:
    Kasabov, N. K. (2014). NeuCube: A spiking neural network architecture for
    mapping, learning and understanding of spatio-temporal brain data.
    Neural Networks, 52, 62-76.
"""

import mlx.core as mx


class EEGEncoder:
    """Encode EEG signals into spike trains.

    Supports multiple encoding strategies commonly used for neural signal
    processing in spiking neural networks:

    - ``"rate"``: Amplitude-to-firing-rate mapping via Poisson sampling.
    - ``"delta"``: Temporal difference coding; spikes when consecutive
      samples differ by more than a threshold.
    - ``"threshold_crossing"``: Multi-level threshold crossing using
      positive and negative thresholds.

    Attributes:
        method: Encoding strategy name.
        num_steps: Number of output time steps for rate coding.
        threshold: Threshold for delta and threshold-crossing methods.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.encoding.medical.eeg import EEGEncoder
        >>> encoder = EEGEncoder(method="rate", num_steps=50)
        >>> signal = mx.random.normal(shape=(4, 64, 256))
        >>> spikes = encoder(signal)
        >>> spikes.shape
        [50, 4, 64]
    """

    def __init__(
        self,
        method: str = "rate",
        num_steps: int = 100,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the EEG encoder.

        Args:
            method: Encoding strategy. One of ``"rate"``, ``"delta"``,
                or ``"threshold_crossing"``.
            num_steps: Number of output time steps. Used directly by
                rate coding; delta and threshold-crossing methods
                resample the input to this many steps.
            threshold: Spike threshold. For delta coding, a spike is
                emitted when the absolute temporal difference exceeds
                this value. For threshold crossing, this sets the
                magnitude of the positive and negative crossing levels.

        Raises:
            ValueError: If ``method`` is not one of the supported
                strategies.
        """
        supported = ("rate", "delta", "threshold_crossing")
        if method not in supported:
            raise ValueError(
                f"Unknown encoding method '{method}'. "
                f"Supported methods: {supported}"
            )

        self.method = method
        self.num_steps = num_steps
        self.threshold = threshold

    def __call__(self, signal: mx.array) -> mx.array:
        """Encode a continuous EEG signal into a spike train.

        Args:
            signal: Raw EEG signal with shape ``[channels, timepoints]``
                or ``[batch, channels, timepoints]``.

        Returns:
            Spike array in time-first format
            ``[num_steps, batch, channels]``.

        Raises:
            ValueError: If ``signal`` does not have 2 or 3 dimensions.
        """
        if signal.ndim == 2:
            # [channels, timepoints] -> [1, channels, timepoints]
            signal = mx.expand_dims(signal, axis=0)
        elif signal.ndim != 3:
            raise ValueError(
                f"Expected signal with 2 or 3 dimensions, got {signal.ndim}."
            )

        # signal is now [batch, channels, timepoints]
        if self.method == "rate":
            return self._rate_encode(signal)
        elif self.method == "delta":
            return self._delta_encode(signal)
        else:
            return self._threshold_crossing_encode(signal)

    # ------------------------------------------------------------------
    # Private encoding implementations
    # ------------------------------------------------------------------

    def _rate_encode(self, signal: mx.array) -> mx.array:
        """Rate coding via Poisson spike generation.

        Normalizes the signal to [0, 1] across each channel, averages
        over the temporal dimension to obtain a single firing-rate
        value per channel, then samples Poisson spikes for
        ``num_steps`` time steps.

        Args:
            signal: Input with shape ``[batch, channels, timepoints]``.

        Returns:
            Spikes of shape ``[num_steps, batch, channels]``.
        """
        # Normalize each channel independently to [0, 1]
        sig_min = mx.min(signal, axis=-1, keepdims=True)
        sig_max = mx.max(signal, axis=-1, keepdims=True)
        denom = sig_max - sig_min
        # Avoid division by zero for constant channels
        denom = mx.where(denom == 0.0, mx.ones_like(denom), denom)
        normalized = (signal - sig_min) / denom  # [batch, channels, timepoints]

        # Collapse timepoints into a single firing rate per channel
        rates = mx.mean(normalized, axis=-1)  # [batch, channels]

        # Poisson spike generation: spike where random < rate
        shape = (self.num_steps,) + rates.shape  # [num_steps, batch, channels]
        rand = mx.random.uniform(shape=shape)
        spikes = mx.where(rand < rates, mx.ones_like(rand), mx.zeros_like(rand))
        return spikes

    def _delta_encode(self, signal: mx.array) -> mx.array:
        """Delta modulation encoding.

        Computes temporal differences along the timepoints axis and
        emits a spike at each sample where the absolute change exceeds
        the threshold. The spike train is then resampled to
        ``num_steps`` via nearest-neighbour indexing.

        Args:
            signal: Input with shape ``[batch, channels, timepoints]``.

        Returns:
            Spikes of shape ``[num_steps, batch, channels]``.
        """
        # Temporal difference along the last axis
        diff = signal[:, :, 1:] - signal[:, :, :-1]  # [batch, channels, T-1]

        # Spike where |diff| > threshold
        spikes = mx.where(
            mx.abs(diff) > self.threshold,
            mx.ones_like(diff),
            mx.zeros_like(diff),
        )  # [batch, channels, T-1]

        # Resample to num_steps along the temporal axis
        spikes = self._resample_temporal(spikes)  # [batch, channels, num_steps]

        # Transpose to time-first: [num_steps, batch, channels]
        spikes = mx.transpose(spikes, axes=(2, 0, 1))
        return spikes

    def _threshold_crossing_encode(self, signal: mx.array) -> mx.array:
        """Multi-level threshold crossing encoding.

        Uses positive and negative thresholds. A spike (+1) is emitted
        at each timestep where the signal crosses above the positive
        threshold, and a spike (-1) where it crosses below the
        negative threshold. The result is then resampled to
        ``num_steps``.

        Args:
            signal: Input with shape ``[batch, channels, timepoints]``.

        Returns:
            Spikes of shape ``[num_steps, batch, channels]``.
        """
        # Normalize signal channel-wise to zero mean, unit variance
        mean = mx.mean(signal, axis=-1, keepdims=True)
        std = mx.sqrt(mx.var(signal, axis=-1, keepdims=True))
        std = mx.where(std == 0.0, mx.ones_like(std), std)
        normed = (signal - mean) / std  # [batch, channels, timepoints]

        # Detect crossings: compare consecutive samples against thresholds
        prev = normed[:, :, :-1]
        curr = normed[:, :, 1:]

        # Positive crossing: signal goes from below to above +threshold
        pos_cross = mx.where(
            (prev <= self.threshold) & (curr > self.threshold),
            mx.ones_like(curr),
            mx.zeros_like(curr),
        )

        # Negative crossing: signal goes from above to below -threshold
        neg_cross = mx.where(
            (prev >= -self.threshold) & (curr < -self.threshold),
            -mx.ones_like(curr),
            mx.zeros_like(curr),
        )

        spikes = pos_cross + neg_cross  # [batch, channels, T-1]

        # Resample to num_steps
        spikes = self._resample_temporal(spikes)  # [batch, channels, num_steps]

        # Transpose to time-first: [num_steps, batch, channels]
        spikes = mx.transpose(spikes, axes=(2, 0, 1))
        return spikes

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _resample_temporal(self, data: mx.array) -> mx.array:
        """Resample the last axis of ``data`` to ``self.num_steps``.

        Uses nearest-neighbour indexing to avoid interpolation artefacts
        on binary spike data.

        Args:
            data: Array of shape ``[batch, channels, T]``.

        Returns:
            Array of shape ``[batch, channels, num_steps]``.
        """
        t_in = data.shape[-1]
        if t_in == self.num_steps:
            return data

        # Nearest-neighbour indices into the source time axis
        indices = mx.round(
            mx.arange(self.num_steps).astype(mx.float32)
            * ((t_in - 1) / max(self.num_steps - 1, 1))
        ).astype(mx.int32)
        indices = mx.clip(indices, 0, t_in - 1)

        return data[:, :, indices]
