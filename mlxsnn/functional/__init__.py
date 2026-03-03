"""Stateless functional API for spiking neural network operations.

Provides pure functions for membrane dynamics, spike generation,
and reset operations — suitable for use with ``mx.compile`` and
custom training loops.
"""

from mlxsnn.functional.neuron_dynamics import lif_step, if_step
from mlxsnn.functional.spike_ops import fire, reset_subtract, reset_zero
from mlxsnn.functional.loss import rate_coding_loss, membrane_loss
