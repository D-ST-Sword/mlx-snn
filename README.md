# mlx-snn

**The first Spiking Neural Network library built natively on Apple MLX.**

mlx-snn brings SNN research to Apple Silicon. It provides spiking neuron models, surrogate gradient training, and spike encoding — all implemented with [MLX](https://github.com/ml-explore/mlx) for unified memory and lazy evaluation on M-series chips.

[![PyPI version](https://img.shields.io/pypi/v/mlx-snn.svg)](https://pypi.org/project/mlx-snn/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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

| Model | Status | Description |
|-------|--------|-------------|
| **Leaky (LIF)** | v0.1 | Leaky Integrate-and-Fire with configurable decay |
| **IF** | v0.1 | Integrate-and-Fire (non-leaky) |
| **Izhikevich** | v0.2 | 2D dynamics with RS/IB/CH/FS presets |
| **Adaptive LIF** | v0.2 | LIF with adaptive threshold |
| **Synaptic** | v0.2 | Conductance-based dual-state LIF |
| **Alpha** | v0.2 | Dual-exponential synaptic model |
| **Liquid State Machine** | *coming soon* | Reservoir computing with spiking neurons |

### Surrogate Gradients

All neuron models support differentiable training via surrogate gradients:
- **Fast Sigmoid** — default, good balance of speed and accuracy
- **Arctan** — smoother gradient landscape
- **Straight-Through Estimator** — simplest, pass-through in a window
- **Custom** — plug in any smooth approximation

### Spike Encoding

| Method | Status | Use Case |
|--------|--------|----------|
| **Rate (Poisson)** | v0.1 | Static images, general-purpose |
| **Latency (TTFS)** | v0.1 | Energy-efficient, temporal coding |
| **Delta Modulation** | v0.2 | Temporal signals, change detection |
| **EEG Encoder** | v0.2 | EEG-to-spike with frequency band support |
| **fMRI BOLD Encoder** | *coming soon* | fMRI signal encoding with HRF handling |

### Training

- BPTT forward pass helper (`bptt_forward`)
- SNN loss functions: `rate_coding_loss`, `membrane_loss`, `mse_count_loss`
- Works with standard MLX optimizers (`mlx.optimizers.Adam`, etc.)

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
- **Surrogate gradients** use the STE pattern with `mx.stop_gradient` — no `autograd.Function` needed

## Architecture

```
mlxsnn/
├── neurons/       # SpikingNeuron base, Leaky, IF, Izhikevich, ALIF, Synaptic, Alpha
├── surrogate/     # fast_sigmoid, arctan, straight_through, custom
├── encoding/      # rate, latency, delta, EEG encoder
├── functional/    # Stateless pure functions (lif_step, if_step, fire, reset)
├── training/      # BPTT helpers, loss functions
└── utils/         # Visualization, metrics (coming soon)
```

## Roadmap

- [x] **v0.1** — LIF/IF neurons, surrogate gradients, rate/latency encoding, MNIST example
- [x] **v0.2** — Izhikevich, ALIF, Synaptic, Alpha neurons, EEG encoder, delta encoding
- [x] **v0.2.1** — Fix fast sigmoid surrogate to match snnTorch rational approximation (97%+ MNIST accuracy)
- [ ] **v0.3** — Liquid State Machine, reservoir topology, EEG epilepsy example
- [ ] **v0.4** — `mx.compile` optimization, neuromorphic datasets, visualization
- [ ] **v1.0** — Full docs, benchmarks, JOSS paper, numerical validation vs snnTorch

## Citation

If you use mlx-snn in your research, please cite:

```bibtex
@software{mlxsnn2025,
  title   = {mlx-snn: Spiking Neural Networks on Apple Silicon via MLX},
  author  = {Qin, Jiahao},
  year    = {2025},
  version = {0.2.1},
  url     = {https://github.com/D-ST-Sword/mlx-snn},
  note    = {https://pypi.org/project/mlx-snn/}
}
```

## License

MIT
