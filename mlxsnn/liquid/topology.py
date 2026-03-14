"""Reservoir topology generators.

Functions that create sparse connectivity matrices for use in
:class:`~mlxsnn.liquid.reservoir.LiquidReservoir`.

All functions return a numpy adjacency matrix which is then converted
to ``mx.array`` by the reservoir constructor.
"""

from __future__ import annotations

import numpy as np


def erdos_renyi(
    n: int,
    p: float = 0.1,
    exc_ratio: float = 0.8,
    spectral_radius: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Erdos-Renyi random sparse connectivity.

    Each possible connection exists independently with probability *p*.
    Weights are drawn from N(0,1) and rescaled so the matrix has the
    desired spectral radius.  Dale's law is enforced: the first
    ``int(exc_ratio * n)`` neurons are excitatory (positive outgoing
    weights) and the rest are inhibitory (negative outgoing weights).

    Args:
        n: Number of reservoir neurons.
        p: Connection probability.
        exc_ratio: Fraction of excitatory neurons.
        spectral_radius: Target spectral radius for weight scaling.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Weight matrix of shape ``(n, n)`` as float32 numpy array.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Random sparse mask
    mask = rng.random((n, n)) < p
    np.fill_diagonal(mask, False)

    # Random weights
    W = rng.standard_normal((n, n)).astype(np.float32) * mask

    # Dale's law: excitatory/inhibitory split
    n_exc = int(exc_ratio * n)
    W[:, :n_exc] = np.abs(W[:, :n_exc])     # excitatory columns
    W[:, n_exc:] = -np.abs(W[:, n_exc:])     # inhibitory columns

    # Rescale to target spectral radius
    eigvals = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigvals))
    if current_radius > 0:
        W = W * (spectral_radius / current_radius)

    return W


def small_world(
    n: int,
    k: int = 6,
    p_rewire: float = 0.1,
    exc_ratio: float = 0.8,
    spectral_radius: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Watts-Strogatz small-world topology.

    Starts with a ring lattice where each neuron connects to its *k*
    nearest neighbours, then randomly rewires edges with probability
    *p_rewire*.

    Args:
        n: Number of reservoir neurons.
        k: Each node connects to k nearest neighbours (must be even).
        p_rewire: Rewiring probability.
        exc_ratio: Fraction of excitatory neurons.
        spectral_radius: Target spectral radius.
        rng: Optional numpy random generator.

    Returns:
        Weight matrix of shape ``(n, n)`` as float32 numpy array.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = k - (k % 2)  # ensure even
    mask = np.zeros((n, n), dtype=bool)

    # Ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            mask[i, (i + j) % n] = True
            mask[i, (i - j) % n] = True

    # Rewire
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < p_rewire:
                target = (i + j) % n
                mask[i, target] = False
                new_target = rng.integers(0, n)
                while new_target == i or mask[i, new_target]:
                    new_target = rng.integers(0, n)
                mask[i, new_target] = True

    W = rng.standard_normal((n, n)).astype(np.float32) * mask

    n_exc = int(exc_ratio * n)
    W[:, :n_exc] = np.abs(W[:, :n_exc])
    W[:, n_exc:] = -np.abs(W[:, n_exc:])

    eigvals = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigvals))
    if current_radius > 0:
        W = W * (spectral_radius / current_radius)

    return W


def scale_free(
    n: int,
    m: int = 3,
    exc_ratio: float = 0.8,
    spectral_radius: float = 0.9,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Barabasi-Albert scale-free topology.

    Grows the network by attaching new nodes with *m* edges using
    preferential attachment.

    Args:
        n: Number of reservoir neurons.
        m: Number of edges per new node.
        exc_ratio: Fraction of excitatory neurons.
        spectral_radius: Target spectral radius.
        rng: Optional numpy random generator.

    Returns:
        Weight matrix of shape ``(n, n)`` as float32 numpy array.
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = np.zeros((n, n), dtype=bool)

    # Start with a fully connected seed of m+1 nodes
    for i in range(m + 1):
        for j in range(m + 1):
            if i != j:
                mask[i, j] = True

    degrees = np.sum(mask, axis=0).astype(float)

    for new_node in range(m + 1, n):
        # Preferential attachment
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = rng.choice(new_node, size=min(m, new_node), replace=False, p=probs)
        for t in targets:
            mask[new_node, t] = True
            mask[t, new_node] = True
        degrees[:new_node] = np.sum(mask[:new_node, :new_node], axis=0).astype(float)
        degrees[new_node] = np.sum(mask[new_node])

    W = rng.standard_normal((n, n)).astype(np.float32) * mask

    n_exc = int(exc_ratio * n)
    W[:, :n_exc] = np.abs(W[:, :n_exc])
    W[:, n_exc:] = -np.abs(W[:, n_exc:])

    eigvals = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigvals))
    if current_radius > 0:
        W = W * (spectral_radius / current_radius)

    return W
