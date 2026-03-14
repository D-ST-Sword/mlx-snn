"""mlx-snn: Spiking Neural Network library built natively on Apple MLX.

mlx-snn provides spiking neuron models, spike encoding, surrogate gradient
functions, and training utilities — all implemented with MLX for Apple Silicon.

Examples:
    >>> import mlxsnn
    >>> lif = mlxsnn.Leaky(beta=0.9)
    >>> state = lif.init_state(32, 128)
"""

__version__ = "0.7.0"

# Neuron models
# Spike encoding
from mlxsnn.encoding import (
    EEGEncoder,
    delta_encode,
    direct_encode,
    latency_encode,
    rate_encode,
    repeat_encode,
)

# Functional API
from mlxsnn.functional import (
    activity_reg_loss,
    ce_count_loss,
    ce_rate_loss,
    fire,
    if_step,
    l1_spike_loss,
    l2_spike_loss,
    lif_step,
    membrane_loss,
    mse_count_loss,
    mse_membrane_loss,
    rate_coding_loss,
    reset_subtract,
    reset_zero,
    spike_count,
    spike_rate,
)

# Composite layers (conv, pooling, flatten)
from mlxsnn.layers import (
    SpikeDropout,
    SpikingAvgPool2d,
    SpikingConv2d,
    SpikingFlatten,
    SpikingMaxPool2d,
    create_neuron,
)
from mlxsnn.neurons import (
    ALIF,
    IF,
    Alpha,
    Izhikevich,
    Leaky,
    RLeaky,
    RSynaptic,
    SpikingNeuron,
    Synaptic,
)

# Surrogate gradient functions
from mlxsnn.surrogate import get_surrogate

# Training utilities
from mlxsnn.training import bptt_forward, compiled_forward, compiled_step

# Utility functions
from mlxsnn.utils import init_states

# Visualization (optional, requires matplotlib)
try:
    from mlxsnn.utils.visualization import plot_firing_rate, plot_membrane, plot_raster
except ImportError:
    pass

# Dataset loaders
from mlxsnn.datasets import (
    CIFAR10DVSDataset,
    DVSGestureDataset,
    EventDataloader,
    NMNISTDataset,
    create_dataloader,
)

try:
    from mlxsnn.datasets import SHDDataset
except (ImportError, NameError):
    pass

# Liquid State Machine
from mlxsnn.liquid import LSM, LiquidReservoir

# Mathematically-principled Conv SNN operators
from mlxsnn.operators import (
    FourierTemporalConv,
    InfoMaxSpikeConv,
    TemporalAggregatedConv,
    TemporalCollapseConv,
)

# NIR interoperability (optional dependency)
try:
    from mlxsnn.nir_export import export_to_nir
    from mlxsnn.nir_import import NIRSequential, import_from_nir
except ImportError:
    pass
