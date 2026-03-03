"""Tests for v0.2 encoding: delta modulation, EEG encoder."""

import mlx.core as mx
import pytest

from mlxsnn.encoding import delta_encode
from mlxsnn.encoding.medical.eeg import EEGEncoder


class TestDeltaEncode:

    def test_output_shape_with_padding(self):
        data = mx.array([
            [[0.0, 0.5]],
            [[0.2, 0.3]],
            [[0.5, 0.1]],
        ])  # [3, 1, 2]
        spikes = delta_encode(data, threshold=0.1)
        mx.eval(spikes)
        assert spikes.shape == data.shape

    def test_output_shape_without_padding(self):
        data = mx.array([
            [[0.0, 0.5]],
            [[0.2, 0.3]],
            [[0.5, 0.1]],
        ])
        spikes = delta_encode(data, threshold=0.1, padding=False)
        mx.eval(spikes)
        assert spikes.shape[0] == data.shape[0] - 1

    def test_positive_spike(self):
        data = mx.array([[[0.0]], [[0.5]]])  # change of 0.5
        spikes = delta_encode(data, threshold=0.1, padding=False)
        mx.eval(spikes)
        assert spikes[0, 0, 0].item() == 1.0

    def test_negative_spike(self):
        data = mx.array([[[0.5]], [[0.0]]])  # change of -0.5
        spikes = delta_encode(data, threshold=0.1, off_spike=True, padding=False)
        mx.eval(spikes)
        assert spikes[0, 0, 0].item() == -1.0

    def test_no_off_spike(self):
        data = mx.array([[[0.5]], [[0.0]]])
        spikes = delta_encode(data, threshold=0.1, off_spike=False, padding=False)
        mx.eval(spikes)
        assert spikes[0, 0, 0].item() == 0.0

    def test_below_threshold_no_spike(self):
        data = mx.array([[[0.0]], [[0.05]]])  # change of 0.05 < 0.1
        spikes = delta_encode(data, threshold=0.1, padding=False)
        mx.eval(spikes)
        assert spikes[0, 0, 0].item() == 0.0


class TestEEGEncoder:

    def test_rate_encoding_shape(self):
        enc = EEGEncoder(method="rate", num_steps=50)
        signal = mx.random.normal(shape=(4, 64, 256))
        spikes = enc(signal)
        mx.eval(spikes)
        assert spikes.shape == (50, 4, 64)

    def test_rate_encoding_2d_input(self):
        enc = EEGEncoder(method="rate", num_steps=30)
        signal = mx.random.normal(shape=(64, 256))
        spikes = enc(signal)
        mx.eval(spikes)
        assert spikes.shape == (30, 1, 64)

    def test_delta_encoding_shape(self):
        enc = EEGEncoder(method="delta", num_steps=50)
        signal = mx.random.normal(shape=(2, 8, 100))
        spikes = enc(signal)
        mx.eval(spikes)
        assert spikes.shape == (50, 2, 8)

    def test_threshold_crossing_shape(self):
        enc = EEGEncoder(method="threshold_crossing", num_steps=50)
        signal = mx.random.normal(shape=(2, 8, 100))
        spikes = enc(signal)
        mx.eval(spikes)
        assert spikes.shape == (50, 2, 8)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown encoding method"):
            EEGEncoder(method="invalid")

    def test_invalid_dimensions(self):
        enc = EEGEncoder(method="rate", num_steps=10)
        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            enc(mx.ones((2,)))
