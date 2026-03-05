# mlx-snn

**A general-purpose Spiking Neural Network library built on Apple [MLX](https://github.com/ml-explore/mlx).**

mlx-snn aims to provide an efficient, research-friendly SNN framework that leverages MLX's unified memory architecture and lazy evaluation. Whether you're exploring neuron dynamics, training classifiers with surrogate gradients, or exchanging models via [NIR](https://github.com/neuromorphs/NIR), mlx-snn offers a clean, Pythonic API that integrates naturally into the MLX ecosystem.

[![PyPI version](https://img.shields.io/pypi/v/mlx-snn.svg)](https://pypi.org/project/mlx-snn/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Why mlx-snn?

- **MLX-native** — All operations use `mlx.core`. No PyTorch/CUDA dependency. Runs on Apple Silicon with zero-copy unified memory.
- **Research-friendly** — Explicit state dicts, composable surrogate gradients, and standard `mlx.nn.Module` patterns make it easy to experiment and extend.
- **Cross-framework** — NIR support lets you import and export models to/from snnTorch, Norse, SpikingJelly, and neuromorphic hardware platforms.
- **Hardware tested** — Currently validated on Apple M3 Max. Future Apple Silicon releases will be tested and supported as they become available.

## Installation

```bash
pip install mlx-snn
```

Requires Python 3.9+ and Apple Silicon (M1/M2/M3/M4).

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

## Features

### Neuron Models

| Model | Since | Description |
|-------|-------|-------------|
| **Leaky (LIF)** | v0.1 | Leaky Integrate-and-Fire with configurable decay |
| **IF** | v0.1 | Integrate-and-Fire (non-leaky) |
| **Izhikevich** | v0.2 | 2D dynamics with RS/IB/CH/FS presets |
| **Adaptive LIF** | v0.2 | LIF with adaptive threshold |
| **Synaptic** | v0.2 | Conductance-based dual-state LIF |
| **Alpha** | v0.2 | Dual-exponential synaptic model |
| **RLeaky** | v0.4 | Recurrent LIF with learnable feedback weight |
| **RSynaptic** | v0.4 | Recurrent Synaptic with learnable feedback weight |

### Surrogate Gradients

All neuron models support differentiable training via surrogate gradients:
- **Fast Sigmoid** — default, good balance of speed and accuracy
- **Arctan** — smoother gradient landscape
- **Sigmoid** — standard logistic sigmoid derivative
- **Triangular (Tent)** — localized, compact support near threshold
- **Straight-Through Estimator** — simplest, unit gradient everywhere
- **Custom** — plug in any smooth approximation

### Spike Encoding

| Method | Since | Use Case |
|--------|-------|----------|
| **Rate (Poisson)** | v0.1 | Static images, general-purpose |
| **Latency (TTFS)** | v0.1 | Energy-efficient, temporal coding |
| **Delta Modulation** | v0.2 | Temporal signals, change detection |
| **EEG Encoder** | v0.2 | EEG-to-spike with frequency band support |

### Training & Loss Functions

- BPTT forward pass helper (`bptt_forward`)
- Loss functions: `ce_rate_loss`, `ce_count_loss`, `mse_membrane_loss`, `membrane_loss`, `rate_coding_loss`
- Learnable parameters: `learn_beta`, `learn_threshold`, `learn_V` on all neurons
- Works with standard MLX optimizers (`mlx.optimizers.Adam`, etc.)

### NIR Interoperability

[NIR](https://github.com/neuromorphs/NIR) (Neuromorphic Intermediate Representation) enables cross-framework SNN model exchange between simulators and neuromorphic hardware platforms.

```bash
pip install mlx-snn[nir]
```

**Export** an mlx-snn model to NIR:

```python
import mlx.nn as nn
import mlxsnn, nir

layers = [
    ('fc1', nn.Linear(784, 128)),
    ('lif1', mlxsnn.Leaky(beta=0.9)),
    ('fc2', nn.Linear(128, 10)),
    ('lif2', mlxsnn.Leaky(beta=0.9)),
]
graph = mlxsnn.export_to_nir(layers)
nir.write('model.nir', graph)
```

**Import** a NIR model into mlx-snn:

```python
graph = nir.read('model.nir')
model = mlxsnn.import_from_nir(graph)
state = model.init_states(batch_size=32)
out, state = model(x, state)
```

Supported conversions: `nn.Linear` <-> `nir.Affine`/`nir.Linear`, `Leaky` <-> `nir.LIF`, `IF` <-> `nir.IF`, `Synaptic` <-> `nir.CubaLIF`.

## Benchmark Highlights

Experiments on MNIST (784-128-10 SNN, 25 timesteps, 5 seeds) on Apple M3 Max, compared with snnTorch on NVIDIA V100:

| Configuration | mlx-snn (M3 Max) | snnTorch (V100) | Speed (mlx-snn) | Speed (snnTorch) |
|---------------|-------------------|-----------------|------------------|------------------|
| Leaky (LIF) | 96.3% | 97.3% | **5.7 s/epoch** | 20.9 s/epoch |
| Synaptic | 94.4% | 95.8% | 6.1 s/epoch | 25.2 s/epoch |
| RLeaky (V=0.1, learn) | 91.6% | 68.1% | 6.8 s/epoch | 25.7 s/epoch |
| RSynaptic (V=0.1, learn) | 89.0% | 52.2% | 7.3 s/epoch | 29.2 s/epoch |
| Fast Sigmoid surrogate | 96.3% | 96.7% | 5.7 s/epoch | 20.9 s/epoch |
| Triangular (Tent) surrogate | 86.0% | 50.8% | 10.9 s/epoch | 20.9 s/epoch |

mlx-snn achieves ~3.7-4.1x faster training per epoch on the M3 Max compared to the V100, while maintaining competitive accuracy. Recurrent neurons with learnable weights significantly outperform snnTorch's default configurations.

For full results, see our benchmarking paper and the [experiments/](experiments/) directory.

## Migrating from snnTorch

mlx-snn is designed to feel familiar to snnTorch users:

```python
# snnTorch                          # mlx-snn
import snntorch as snn              import mlxsnn
lif = snn.Leaky(beta=0.9)          lif = mlxsnn.Leaky(beta=0.9)
spk, mem = lif(x, mem)             spk, state = lif(x, state)
                                    # state["mem"] == mem
```

Key differences:
- **State is a dict**, not separate tensors — plays well with MLX functional transforms
- **No global hidden state** — state is always explicit (pass in, get out)
- **MLX arrays** instead of PyTorch tensors — use `mx.array`, not `torch.Tensor`
- **Surrogate gradients** use the STE pattern with `mx.stop_gradient`

## Project Structure

```
mlxsnn/
├── neurons/       # SpikingNeuron base, Leaky, IF, Izhikevich, ALIF, Synaptic, Alpha, RLeaky, RSynaptic
├── surrogate/     # fast_sigmoid, arctan, sigmoid, triangular, straight_through, custom
├── encoding/      # rate, latency, delta, EEG encoder
├── functional/    # Stateless pure functions (lif_step, if_step, fire, reset)
├── training/      # BPTT helpers, loss functions
└── nir_*.py       # NIR export/import utilities
```

## Roadmap

- [x] **v0.1** — Core neurons (LIF, IF), surrogate gradients, rate/latency encoding
- [x] **v0.2** — Extended neurons (Izhikevich, ALIF, Synaptic, Alpha), EEG encoder, delta encoding
- [x] **v0.3** — NIR interoperability (export/import)
- [x] **v0.4** — Recurrent neurons (RLeaky, RSynaptic), learnable thresholds, expanded surrogates and losses
- [ ] **v0.5** — Liquid State Machine, reservoir topology, `mx.compile` optimization
- [ ] **v1.0** — Full documentation, comprehensive benchmarks, neuromorphic dataset loaders

## Publications

- **mlx-snn v0.1**: [Spiking Neural Networks on Apple Silicon via MLX](https://arxiv.org/abs/2603.03529) (arXiv, 2026)
- **mlx-snn v0.4**: Spiking Neural Network Training on Apple Silicon: Cross-Framework Benchmarking (in preparation)

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

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/D-ST-Sword/mlx-snn).

## License

MIT
