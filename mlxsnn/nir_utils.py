"""Conversion utilities between mlx-snn and NIR parameter spaces.

The core relationship between mlx-snn's discrete-time decay factor (beta)
and NIR's continuous-time time constant (tau) is based on forward Euler
discretization:

    beta = 1 - dt / tau
    tau  = dt / (1 - beta)
    r    = 1 / (1 - beta)   (gain factor)

where dt is the simulation timestep (default 1e-4, matching snnTorch).
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx


DEFAULT_DT = 1e-4


def beta_to_tau(beta: float, dt: float = DEFAULT_DT) -> float:
    """Convert discrete decay factor to continuous time constant.

    Args:
        beta: Membrane decay factor, must satisfy 0 <= beta < 1.
        dt: Simulation timestep in seconds.

    Returns:
        Time constant tau in seconds.

    Raises:
        ValueError: If beta is out of range [0, 1).
    """
    if beta < 0 or beta >= 1:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    return dt / (1.0 - beta)


def tau_to_beta(tau: float, dt: float = DEFAULT_DT) -> float:
    """Convert continuous time constant to discrete decay factor.

    Args:
        tau: Time constant in seconds, must be > 0.
        dt: Simulation timestep in seconds.

    Returns:
        Decay factor beta in [0, 1).

    Raises:
        ValueError: If tau <= 0.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    return 1.0 - dt / tau


def beta_to_r(beta: float) -> float:
    """Convert discrete decay factor to gain factor r.

    Args:
        beta: Membrane decay factor, must satisfy 0 <= beta < 1.

    Returns:
        Gain factor r = 1 / (1 - beta).

    Raises:
        ValueError: If beta is out of range [0, 1).
    """
    if beta < 0 or beta >= 1:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    return 1.0 / (1.0 - beta)


def mx_to_numpy(arr) -> np.ndarray:
    """Convert an mlx array or Python scalar to a numpy array.

    Args:
        arr: An ``mx.array``, Python float/int, or numpy array.

    Returns:
        A numpy ndarray (float32).
    """
    if isinstance(arr, mx.array):
        return np.array(arr, copy=False).astype(np.float32)
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32)
    return np.float32(arr)


def numpy_to_mx(arr: np.ndarray) -> mx.array:
    """Convert a numpy array to an mlx array.

    Args:
        arr: A numpy ndarray.

    Returns:
        An ``mx.array`` with float32 dtype.
    """
    return mx.array(arr.astype(np.float32))
