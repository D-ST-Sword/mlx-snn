"""Stateless functional API for spiking neural network operations.

Provides pure functions for membrane dynamics, spike generation,
and reset operations — suitable for use with ``mx.compile`` and
custom training loops.
"""

from mlxsnn.functional.loss import (
    activity_reg_loss,
    ce_count_loss,
    ce_rate_loss,
    l1_spike_loss,
    l2_spike_loss,
    membrane_loss,
    mse_count_loss,
    mse_membrane_loss,
    rate_coding_loss,
    spike_count,
    spike_rate,
)
from mlxsnn.functional.neuron_dynamics import if_step, lif_step
from mlxsnn.functional.spike_ops import fire, reset_subtract, reset_zero
