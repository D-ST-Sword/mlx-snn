"""mlx-snn: Spiking Neural Network library built natively on Apple MLX.

mlx-snn provides spiking neuron models, spike encoding, surrogate gradient
functions, and training utilities — all implemented with MLX for Apple Silicon.

Examples:
    >>> import mlxsnn
    >>> lif = mlxsnn.Leaky(beta=0.9)
    >>> state = lif.init_state(32, 128)
"""

__version__ = "0.6.0"

# Neuron models
from mlxsnn.neurons import (
    SpikingNeuron, Leaky, IF, Izhikevich, ALIF, Synaptic, Alpha,
    RLeaky, RSynaptic,
)

# Surrogate gradient functions
from mlxsnn.surrogate import get_surrogate

# Spike encoding
from mlxsnn.encoding import (
    rate_encode, latency_encode, delta_encode, direct_encode, repeat_encode,
    EEGEncoder,
)

# Functional API
from mlxsnn.functional import lif_step, if_step, fire, reset_subtract, reset_zero
from mlxsnn.functional import rate_coding_loss, membrane_loss, mse_count_loss
from mlxsnn.functional import ce_rate_loss, ce_count_loss, mse_membrane_loss
from mlxsnn.functional import spike_rate, spike_count
from mlxsnn.functional import activity_reg_loss, l1_spike_loss, l2_spike_loss

# Composite layers (conv, pooling, flatten)
from mlxsnn.layers import (
    SpikingConv2d, SpikingMaxPool2d, SpikingAvgPool2d, SpikingFlatten,
    SpikeDropout, create_neuron,
)

# Training utilities
from mlxsnn.training import bptt_forward

# Utility functions
from mlxsnn.utils import init_states

# Visualization (optional, requires matplotlib)
try:
    from mlxsnn.utils.visualization import plot_raster, plot_membrane, plot_firing_rate
except ImportError:
    pass

# Dataset loaders
from mlxsnn.datasets import (
    DVSGestureDataset,
    CIFAR10DVSDataset,
    NMNISTDataset,
    EventDataloader,
    create_dataloader,
)
try:
    from mlxsnn.datasets import SHDDataset
except (ImportError, NameError):
    pass

# Liquid State Machine
from mlxsnn.liquid import LiquidReservoir, LSM

# Mathematically-principled Conv SNN operators
from mlxsnn.operators import (
    TemporalAggregatedConv,
    FourierTemporalConv,
    InfoMaxSpikeConv,
    TemporalCollapseConv,
)

# NIR interoperability (optional dependency)
try:
    from mlxsnn.nir_export import export_to_nir
    from mlxsnn.nir_import import import_from_nir, NIRSequential
except ImportError:
    pass
