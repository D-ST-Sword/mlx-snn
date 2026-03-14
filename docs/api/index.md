# API Reference

mlx-snn's API is organized into the following modules:

| Module | Description |
|--------|-------------|
| [Neurons](neurons.md) | Spiking neuron models (LIF, IF, Izhikevich, ALIF, Synaptic, Alpha, RLeaky, RSynaptic) |
| [Surrogate Gradients](surrogate.md) | Differentiable surrogate gradient functions for spike generation |
| [Encoding](encoding.md) | Spike encoding methods (rate, latency, delta, direct, repeat, EEG) |
| [Functional](functional.md) | Stateless pure functions — neuron dynamics, spike ops, loss functions |
| [Layers](layers.md) | Composite spiking layers (conv, pooling, dropout, flatten) |
| [Operators](operators.md) | Mathematically-principled Conv SNN optimization operators |
| [Training](training.md) | BPTT helpers and training utilities |
| [Datasets](datasets.md) | Neuromorphic dataset loaders (DVS-Gesture, CIFAR10-DVS, N-MNIST, SHD) |
| [Utilities](utils.md) | State management, visualization, metrics |
| [NIR Interop](nir.md) | Cross-framework model exchange via NIR |

## Import Conventions

```python
import mlxsnn                    # Main package (most classes/functions)
from mlxsnn.neurons import Leaky  # Direct submodule import
from mlxsnn.operators import TemporalAggregatedConv  # Operators
```

All neuron models, encoding functions, loss functions, and layers are re-exported from the top-level `mlxsnn` namespace.
