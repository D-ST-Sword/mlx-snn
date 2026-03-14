"""Visualization utilities for spiking neural networks.

Provides convenience functions for plotting spike rasters, membrane
potential traces, and firing rate distributions.  All functions
accept mlx arrays and return matplotlib ``Axes`` objects for further
customization.

Requires ``matplotlib`` (install via ``pip install mlx-snn[viz]``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _ensure_matplotlib():
    """Lazily import matplotlib, raising a clear error if missing."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install mlx-snn[viz]"
        )


def _to_numpy(arr) -> np.ndarray:
    """Convert an mx.array to a numpy ndarray."""
    import mlx.core as mx
    if isinstance(arr, mx.array):
        mx.eval(arr)
        return np.array(arr)
    return np.asarray(arr)


def plot_raster(
    spikes,
    neuron_ids: Optional[list] = None,
    ax=None,
    title: str = "Spike Raster",
    xlabel: str = "Timestep",
    ylabel: str = "Neuron",
    marker: str = "|",
    markersize: float = 2.0,
    color: str = "black",
    show: bool = True,
):
    """Plot a spike raster diagram.

    Args:
        spikes: Spike tensor ``[T, neurons]`` or ``[T, batch, neurons]``.
            If 3-D, the first batch element is plotted.
        neuron_ids: Optional list of neuron indices to plot.
        ax: Existing matplotlib ``Axes`` to draw on. If ``None``, a new
            figure is created.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        marker: Marker style for spikes.
        markersize: Size of spike markers.
        color: Color of spike markers.
        show: Whether to call ``plt.show()`` after plotting.

    Returns:
        matplotlib ``Axes`` object.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.utils.visualization import plot_raster
        >>> spikes = (mx.random.uniform(shape=(50, 20)) > 0.8).astype(mx.float32)
        >>> ax = plot_raster(spikes, show=False)
    """
    _, plt = _ensure_matplotlib()
    spk = _to_numpy(spikes)

    # Handle 3-D input: take first batch element
    if spk.ndim == 3:
        spk = spk[:, 0, :]

    T, N = spk.shape
    if neuron_ids is not None:
        spk = spk[:, neuron_ids]
        N = len(neuron_ids)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    times, neurons = np.where(spk > 0.5)
    ax.scatter(times, neurons, marker=marker, s=markersize, c=color, linewidths=0.5)
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show:
        plt.show()
    return ax


def plot_membrane(
    mem,
    neuron_ids: Optional[list] = None,
    threshold: Optional[float] = 1.0,
    ax=None,
    title: str = "Membrane Potential",
    xlabel: str = "Timestep",
    ylabel: str = "Membrane Potential",
    show: bool = True,
):
    """Plot membrane potential traces over time.

    Args:
        mem: Membrane potential ``[T, neurons]`` or ``[T, batch, neurons]``.
            If 3-D, the first batch element is plotted.
        neuron_ids: Optional list of neuron indices to plot. If ``None``,
            plots all neurons (up to 10).
        threshold: If not ``None``, draws a horizontal dashed line at
            the threshold value.
        ax: Existing matplotlib ``Axes``.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        show: Whether to call ``plt.show()``.

    Returns:
        matplotlib ``Axes`` object.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.utils.visualization import plot_membrane
        >>> mem = mx.random.uniform(shape=(50, 5))
        >>> ax = plot_membrane(mem, show=False)
    """
    _, plt = _ensure_matplotlib()
    m = _to_numpy(mem)

    if m.ndim == 3:
        m = m[:, 0, :]

    T, N = m.shape
    if neuron_ids is None:
        neuron_ids = list(range(min(N, 10)))
    m = m[:, neuron_ids]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    for i, nid in enumerate(neuron_ids):
        ax.plot(m[:, i], label=f"Neuron {nid}")

    if threshold is not None:
        ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.5, label="Threshold")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize="small")

    if show:
        plt.show()
    return ax


def plot_firing_rate(
    spk_out,
    ax=None,
    title: str = "Firing Rate",
    xlabel: str = "Neuron",
    ylabel: str = "Firing Rate",
    color: str = "steelblue",
    show: bool = True,
):
    """Plot per-neuron firing rate as a bar chart.

    Args:
        spk_out: Spike tensor ``[T, neurons]`` or ``[T, batch, neurons]``.
            If 3-D, averages over the batch dimension.
        ax: Existing matplotlib ``Axes``.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        color: Bar color.
        show: Whether to call ``plt.show()``.

    Returns:
        matplotlib ``Axes`` object.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.utils.visualization import plot_firing_rate
        >>> spk = (mx.random.uniform(shape=(100, 20)) > 0.7).astype(mx.float32)
        >>> ax = plot_firing_rate(spk, show=False)
    """
    _, plt = _ensure_matplotlib()
    spk = _to_numpy(spk_out)

    if spk.ndim == 3:
        # Average over batch: [T, B, N] -> [T, N]
        spk = spk.mean(axis=1)

    # Mean over time: [T, N] -> [N]
    rates = spk.mean(axis=0)
    N = len(rates)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.bar(range(N), rates, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, N - 0.5)

    if show:
        plt.show()
    return ax
