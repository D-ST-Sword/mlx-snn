"""Tests for v0.5 visualization utilities."""

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")  # Non-interactive backend for testing

import mlx.core as mx

from mlxsnn.utils.visualization import plot_raster, plot_membrane, plot_firing_rate


class TestPlotRaster:
    def test_returns_axes(self):
        spk = (mx.random.uniform(shape=(50, 20)) > 0.8).astype(mx.float32)
        ax = plot_raster(spk, show=False)
        assert hasattr(ax, "plot")  # Is a matplotlib Axes

    def test_3d_input(self):
        spk = (mx.random.uniform(shape=(50, 4, 20)) > 0.8).astype(mx.float32)
        ax = plot_raster(spk, show=False)
        assert hasattr(ax, "plot")

    def test_neuron_ids(self):
        spk = (mx.random.uniform(shape=(50, 20)) > 0.8).astype(mx.float32)
        ax = plot_raster(spk, neuron_ids=[0, 5, 10], show=False)
        assert hasattr(ax, "plot")


class TestPlotMembrane:
    def test_returns_axes(self):
        mem = mx.random.uniform(shape=(50, 5))
        ax = plot_membrane(mem, show=False)
        assert hasattr(ax, "plot")

    def test_3d_input(self):
        mem = mx.random.uniform(shape=(50, 4, 10))
        ax = plot_membrane(mem, show=False)
        assert hasattr(ax, "plot")

    def test_no_threshold(self):
        mem = mx.random.uniform(shape=(50, 5))
        ax = plot_membrane(mem, threshold=None, show=False)
        assert hasattr(ax, "plot")

    def test_neuron_ids(self):
        mem = mx.random.uniform(shape=(50, 20))
        ax = plot_membrane(mem, neuron_ids=[0, 3], show=False)
        assert hasattr(ax, "plot")


class TestPlotFiringRate:
    def test_returns_axes(self):
        spk = (mx.random.uniform(shape=(100, 20)) > 0.7).astype(mx.float32)
        ax = plot_firing_rate(spk, show=False)
        assert hasattr(ax, "plot")

    def test_3d_input(self):
        spk = (mx.random.uniform(shape=(100, 4, 20)) > 0.7).astype(mx.float32)
        ax = plot_firing_rate(spk, show=False)
        assert hasattr(ax, "plot")
