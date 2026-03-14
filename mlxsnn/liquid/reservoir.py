"""Liquid State Machine reservoir with random sparse connectivity.

The reservoir is a recurrently connected pool of spiking neurons.
Input is projected into the reservoir, and the reservoir state
(spike pattern) can be read out by a downstream classifier.

Reference:
    Maass, W., Natschlager, T., & Markram, H. (2002).
    Real-time computing without stable states: A new framework
    for neural computation based on perturbations.
    Neural Computation, 14(11), 2531-2560.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlxsnn.liquid.topology import erdos_renyi


class LiquidReservoir(nn.Module):
    """Recurrent spiking reservoir with sparse random connectivity.

    The reservoir receives input via a random input projection,
    processes it through recurrent spiking dynamics, and outputs
    the resulting spike pattern.

    Args:
        input_size: Dimensionality of the input.
        reservoir_size: Number of neurons in the reservoir.
        connectivity: Connection probability (for Erdos-Renyi).
        spectral_radius: Target spectral radius of recurrent weights.
        exc_ratio: Fraction of excitatory neurons (Dale's law).
        beta: Membrane decay factor for reservoir LIF neurons.
        threshold: Spike threshold.
        topology: Connectivity topology ('erdos_renyi', 'small_world',
            'scale_free').
        input_scaling: Scaling factor for input weights.
        seed: Random seed for reproducibility.

    Examples:
        >>> reservoir = LiquidReservoir(input_size=64, reservoir_size=200)
        >>> state = reservoir.init_state(batch_size=8)
        >>> x = mx.random.normal((8, 64))
        >>> spk, state = reservoir(x, state)
        >>> spk.shape
        [8, 200]
    """

    def __init__(
        self,
        input_size: int,
        reservoir_size: int = 500,
        connectivity: float = 0.1,
        spectral_radius: float = 0.9,
        exc_ratio: float = 0.8,
        beta: float = 0.9,
        threshold: float = 1.0,
        topology: str = "erdos_renyi",
        input_scaling: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.reservoir_size = reservoir_size
        self._beta = beta
        self._threshold = threshold

        rng = np.random.default_rng(seed)

        # Build reservoir weight matrix
        if topology == "erdos_renyi":
            from mlxsnn.liquid.topology import erdos_renyi as topo_fn
        elif topology == "small_world":
            from mlxsnn.liquid.topology import small_world as topo_fn
        elif topology == "scale_free":
            from mlxsnn.liquid.topology import scale_free as topo_fn
        else:
            raise ValueError(f"Unknown topology: {topology}")

        W_res = topo_fn(
            n=reservoir_size,
            spectral_radius=spectral_radius,
            exc_ratio=exc_ratio,
            rng=rng,
        )
        self.W_res = mx.array(W_res)

        # Input projection (random, sparse)
        W_in = rng.standard_normal((input_size, reservoir_size)).astype(np.float32)
        W_in *= input_scaling / np.sqrt(input_size)
        self.W_in = mx.array(W_in)

    def init_state(self, batch_size: int) -> dict:
        """Initialize reservoir state.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            Dictionary with 'mem' (membrane potential) and 'spk'
            (previous spikes) tensors.
        """
        return {
            "mem": mx.zeros((batch_size, self.reservoir_size)),
            "spk": mx.zeros((batch_size, self.reservoir_size)),
        }

    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        """Process one timestep through the reservoir.

        Args:
            x: Input of shape ``[batch, input_size]``.
            state: Reservoir state from previous timestep.

        Returns:
            Tuple of ``(spikes, new_state)`` where spikes has shape
            ``[batch, reservoir_size]``.
        """
        mem = state["mem"]
        prev_spk = state["spk"]

        # Input current + recurrent current
        I_in = x @ self.W_in
        I_rec = prev_spk @ self.W_res
        I = I_in + I_rec

        # LIF dynamics
        mem = self._beta * mem + I

        # Spike generation (hard threshold, no surrogate needed for reservoir)
        spk = mx.where(mem >= self._threshold, 1.0, 0.0)

        # Reset by subtraction
        mem = mem - spk * self._threshold

        return spk, {"mem": mem, "spk": spk}
