"""Training utilities for spiking neural networks.

Provides BPTT forward pass helpers, truncated/chunked BPTT for
memory-efficient training, and activity regularization.
"""

from mlxsnn.training.bptt import (
    bptt_forward,
    chunked_bptt_forward,
    detach_state,
)
from mlxsnn.training.compile import compiled_forward, compiled_step
