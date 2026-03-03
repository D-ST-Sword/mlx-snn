"""mlx-snn: Spiking Neural Network library built natively on Apple MLX.

mlx-snn provides spiking neuron models, spike encoding, surrogate gradient
functions, and training utilities — all implemented with MLX for Apple Silicon.

Examples:
    >>> import mlxsnn
    >>> lif = mlxsnn.Leaky(beta=0.9)
    >>> state = lif.init_state(batch_size=32, features=128)
"""

__version__ = "0.2.1"

# Neuron models
from mlxsnn.neurons import SpikingNeuron, Leaky, IF, Izhikevich, ALIF, Synaptic, Alpha

# Surrogate gradient functions
from mlxsnn.surrogate import get_surrogate

# Spike encoding
from mlxsnn.encoding import rate_encode, latency_encode, delta_encode, EEGEncoder

# Functional API
from mlxsnn.functional import lif_step, if_step, fire, reset_subtract, reset_zero
from mlxsnn.functional import rate_coding_loss, membrane_loss

# Training utilities
from mlxsnn.training import bptt_forward
