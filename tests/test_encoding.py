"""Tests for spike encoding methods."""

import mlx.core as mx
import pytest

from mlxsnn.encoding import rate_encode, latency_encode


class TestRateEncoding:
    """Tests for Poisson rate encoding."""

    def test_output_shape(self):
        data = mx.ones((4, 10)) * 0.5
        spikes = rate_encode(data, num_steps=25)
        mx.eval(spikes)
        assert spikes.shape == (25, 4, 10)

    def test_binary_output(self):
        """Output should only contain 0s and 1s."""
        data = mx.ones((4, 10)) * 0.5
        spikes = rate_encode(data, num_steps=50)
        mx.eval(spikes)
        unique_vals = set()
        flat = spikes.reshape(-1)
        mx.eval(flat)
        for i in range(flat.shape[0]):
            unique_vals.add(flat[i].item())
        assert unique_vals.issubset({0.0, 1.0})

    def test_high_rate_more_spikes(self):
        """Higher input values should produce more spikes on average."""
        high = mx.ones((1, 100)) * 0.9
        low = mx.ones((1, 100)) * 0.1
        spikes_high = rate_encode(high, num_steps=200)
        spikes_low = rate_encode(low, num_steps=200)
        mx.eval(spikes_high, spikes_low)
        count_high = mx.sum(spikes_high).item()
        count_low = mx.sum(spikes_low).item()
        assert count_high > count_low

    def test_zero_input_no_spikes(self):
        """Zero input should produce no spikes."""
        data = mx.zeros((2, 5))
        spikes = rate_encode(data, num_steps=100)
        mx.eval(spikes)
        assert mx.sum(spikes).item() == 0.0

    def test_gain_and_offset(self):
        """Gain and offset should modify firing rates."""
        data = mx.ones((1, 10)) * 0.3
        spikes_base = rate_encode(data, num_steps=500)
        spikes_gain = rate_encode(data, num_steps=500, gain=2.0)
        mx.eval(spikes_base, spikes_gain)
        # gain=2.0 makes effective rate 0.6 > 0.3
        assert mx.sum(spikes_gain).item() > mx.sum(spikes_base).item()


class TestLatencyEncoding:
    """Tests for latency (time-to-first-spike) encoding."""

    def test_output_shape(self):
        data = mx.ones((4, 10)) * 0.5
        spikes = latency_encode(data, num_steps=25)
        mx.eval(spikes)
        assert spikes.shape == (25, 4, 10)

    def test_single_spike_per_neuron(self):
        """Each neuron should fire exactly once."""
        data = mx.array([[0.2, 0.8, 0.5]])
        spikes = latency_encode(data, num_steps=20)
        mx.eval(spikes)
        # Sum over time for each neuron should be 1
        spike_counts = mx.sum(spikes, axis=0)
        mx.eval(spike_counts)
        assert mx.allclose(spike_counts, mx.ones_like(spike_counts)).item()

    def test_higher_value_earlier_spike_linear(self):
        """Higher values should spike earlier with linear encoding."""
        data = mx.array([[0.9, 0.1]])
        spikes = latency_encode(data, num_steps=20, linear=True)
        mx.eval(spikes)
        # Find spike times (argmax along time axis)
        spike_time_high = mx.argmax(spikes[:, 0, 0], axis=0)
        spike_time_low = mx.argmax(spikes[:, 0, 1], axis=0)
        mx.eval(spike_time_high, spike_time_low)
        assert spike_time_high.item() < spike_time_low.item()

    def test_binary_output(self):
        """Output should be binary."""
        data = mx.ones((2, 5)) * 0.5
        spikes = latency_encode(data, num_steps=15)
        mx.eval(spikes)
        flat = spikes.reshape(-1)
        mx.eval(flat)
        for i in range(flat.shape[0]):
            assert flat[i].item() in (0.0, 1.0)
