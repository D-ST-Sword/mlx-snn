"""Frequency-band decomposition spike encoding for EEG signals.

Decomposes EEG signals into canonical frequency bands (delta, theta, alpha,
beta, gamma) and applies rate coding to the instantaneous power of each band.
This preserves both spectral and temporal information in the spike domain.

This encoding is physiologically motivated: different EEG frequency bands
correlate with distinct brain states, and epileptic seizures typically show
characteristic changes across multiple bands simultaneously.

Reference:
    Nunez, P. L. & Srinivasan, R. (2006). Electric Fields of the Brain.
    Oxford University Press.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

# Canonical EEG frequency bands (Hz)
DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def frequency_band_encode(
    data: mx.array,
    sfreq: float = 256.0,
    num_steps: int | None = None,
    bands: dict[str, tuple[float, float]] | None = None,
    method: str = "power",
) -> mx.array:
    """Encode EEG signal into spike trains via frequency-band decomposition.

    Steps:
    1. Bandpass filter into canonical frequency bands (via FFT).
    2. Compute instantaneous power envelope per band.
    3. Normalize power to [0, 1] per band.
    4. Apply Poisson rate coding to generate spikes.

    Args:
        data: EEG signal ``[T_samples, channels]``, raw amplitude at
            ``sfreq`` Hz. Should be z-score normalized per channel.
        sfreq: Sampling frequency in Hz.
        num_steps: Number of output timesteps. If None, matches input
            length. If smaller, the power envelope is downsampled.
        bands: Dict mapping band name to (low_hz, high_hz). Defaults to
            the 5 canonical EEG bands.
        method: ``"power"`` for instantaneous power envelope,
            ``"amplitude"`` for rectified filtered signal.

    Returns:
        Spike trains ``[num_steps, channels * n_bands]`` where
        ``n_bands`` is the number of frequency bands.

    Examples:
        >>> import mlx.core as mx
        >>> eeg = mx.random.normal((1280, 18))  # 5s at 256Hz, 18 channels
        >>> spikes = frequency_band_encode(eeg, sfreq=256, num_steps=128)
        >>> spikes.shape  # [128, 18*5] = [128, 90]
    """
    if bands is None:
        bands = DEFAULT_BANDS

    data_np = np.array(data)  # [T_samples, C]
    T_samples, C = data_np.shape
    n_bands = len(bands)

    # FFT-based bandpass filtering
    freqs = np.fft.rfftfreq(T_samples, d=1.0 / sfreq)
    fft_data = np.fft.rfft(data_np, axis=0)  # [T_freq, C]

    band_powers = []
    for band_name, (lo, hi) in bands.items():
        # Create bandpass mask
        mask = ((freqs >= lo) & (freqs <= hi)).astype(np.float64)
        # Smooth edges to avoid ringing
        mask_2d = mask[:, np.newaxis]

        # Apply filter in frequency domain
        filtered_fft = fft_data * mask_2d
        filtered = np.fft.irfft(filtered_fft, n=T_samples, axis=0)

        # Compute power envelope
        if method == "power":
            envelope = filtered ** 2
        else:  # amplitude
            envelope = np.abs(filtered)

        # Smooth with a short moving average (50ms window)
        win_size = max(1, int(0.05 * sfreq))
        kernel = np.ones(win_size) / win_size
        smoothed = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"), axis=0, arr=envelope
        )

        band_powers.append(smoothed)

    # Stack bands: [T_samples, C, n_bands]
    band_powers = np.stack(band_powers, axis=-1)

    # Normalize each band to [0, 1] per channel
    for b in range(n_bands):
        bmin = band_powers[:, :, b].min(axis=0, keepdims=True)
        bmax = band_powers[:, :, b].max(axis=0, keepdims=True)
        band_powers[:, :, b] = (band_powers[:, :, b] - bmin) / (bmax - bmin + 1e-8)

    # Reshape to [T_samples, C * n_bands]
    band_powers = band_powers.reshape(T_samples, C * n_bands)

    # Downsample if needed
    if num_steps is not None and num_steps < T_samples:
        indices = np.linspace(0, T_samples - 1, num_steps, dtype=int)
        band_powers = band_powers[indices]
    elif num_steps is None:
        num_steps = T_samples

    # Poisson rate coding on the normalized power
    rates = mx.array(band_powers.astype(np.float32))
    rand = mx.random.uniform(shape=rates.shape)
    spikes = mx.where(rand < rates, 1.0, 0.0)

    return spikes
