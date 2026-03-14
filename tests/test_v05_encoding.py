"""Tests for v0.5 direct and repeat encoding."""

import mlx.core as mx
import pytest

from mlxsnn.encoding.direct import direct_encode, repeat_encode


class TestDirectEncode:
    def test_shape_2d(self):
        data = mx.ones((4, 10))
        out = direct_encode(data, num_steps=25)
        assert list(out.shape) == [25, 4, 10]

    def test_shape_3d(self):
        data = mx.ones((2, 3, 4))
        out = direct_encode(data, num_steps=10)
        assert list(out.shape) == [10, 2, 3, 4]

    def test_values_preserved(self):
        data = mx.array([[1.0, 2.0], [3.0, 4.0]])
        out = direct_encode(data, num_steps=5)
        mx.eval(out)
        for t in range(5):
            assert mx.allclose(out[t], data).item()

    def test_single_step(self):
        data = mx.ones((4, 10))
        out = direct_encode(data, num_steps=1)
        assert list(out.shape) == [1, 4, 10]

    def test_4d_input(self):
        data = mx.ones((2, 8, 8, 3))
        out = direct_encode(data, num_steps=16)
        assert list(out.shape) == [16, 2, 8, 8, 3]


class TestRepeatEncode:
    def test_shape(self):
        spikes = mx.ones((5, 4, 10))
        out = repeat_encode(spikes, num_repeats=3)
        assert list(out.shape) == [15, 4, 10]

    def test_values_preserved(self):
        spikes = mx.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # [2, 1, 2]
        out = repeat_encode(spikes, num_repeats=2)
        mx.eval(out)
        assert list(out.shape) == [4, 1, 2]
        assert mx.allclose(out[0], spikes[0]).item()
        assert mx.allclose(out[1], spikes[1]).item()
        assert mx.allclose(out[2], spikes[0]).item()
        assert mx.allclose(out[3], spikes[1]).item()

    def test_single_repeat(self):
        spikes = mx.ones((5, 4, 10))
        out = repeat_encode(spikes, num_repeats=1)
        assert list(out.shape) == [5, 4, 10]
