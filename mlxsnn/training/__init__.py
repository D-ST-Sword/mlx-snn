"""Training utilities for spiking neural networks.

Provides BPTT forward pass helpers and activity regularization.
"""

from mlxsnn.training.bptt import bptt_forward
from mlxsnn.training.compile import compiled_step, compiled_forward
