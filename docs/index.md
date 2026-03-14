# mlx-snn

**A general-purpose Spiking Neural Network library built on Apple [MLX](https://github.com/ml-explore/mlx).**

mlx-snn provides an efficient, research-friendly SNN framework that leverages MLX's unified memory architecture and lazy evaluation. Whether you're exploring neuron dynamics, training classifiers with surrogate gradients, or exchanging models via [NIR](https://github.com/neuromorphs/NIR), mlx-snn offers a clean, Pythonic API that integrates naturally into the MLX ecosystem.

## Key Features

- **MLX-native** — All operations use `mlx.core`. No PyTorch/CUDA dependency. Runs on Apple Silicon with zero-copy unified memory.
- **Research-friendly** — Explicit state dicts, composable surrogate gradients, and standard `mlx.nn.Module` patterns.
- **Cross-framework** — NIR support lets you import/export models to snnTorch, Norse, SpikingJelly, and neuromorphic hardware.
- **8 neuron models** — LIF, IF, Izhikevich, ALIF, Synaptic, Alpha, RLeaky, RSynaptic
- **6 Conv SNN operators** — TAC, TAC-TP, L-TAC, FTC, IMC, TCC for efficient spatiotemporal processing
- **5 surrogate gradients** — Fast sigmoid, arctan, sigmoid, triangular, straight-through
- **4 neuromorphic datasets** — DVS-Gesture, CIFAR10-DVS, N-MNIST, SHD

## Quick Start

```python
import mlx.core as mx
import mlx.nn as nn
import mlxsnn

# Build a spiking network
fc = nn.Linear(784, 10)
lif = mlxsnn.Leaky(beta=0.95, threshold=1.0)

# Encode input as spike train and run over time
spikes_in = mlxsnn.rate_encode(mx.random.uniform(shape=(8, 784)), num_steps=25)
state = lif.init_state(batch_size=8, features=10)

for t in range(25):
    spk, state = lif(fc(spikes_in[t]), state)

print("Output membrane:", state["mem"].shape)  # (8, 10)
```

## Installation

```bash
pip install mlx-snn
```

Requires Python 3.9+ and Apple Silicon (M1/M2/M3/M4).

### Optional dependencies

```bash
pip install mlx-snn[viz]       # matplotlib for visualization
pip install mlx-snn[datasets]  # aedat, h5py for neuromorphic datasets
pip install mlx-snn[nir]       # NIR interoperability
```

## Citation

If you use mlx-snn in your research, please cite:

```bibtex
@misc{qin2026mlxsnn,
  title         = {mlx-snn: Spiking Neural Networks on Apple Silicon via {MLX}},
  author        = {Jiahao Qin},
  year          = {2026},
  eprint        = {2603.03529},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2603.03529}
}
```
