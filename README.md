# mlx-snn

**A general-purpose Spiking Neural Network library built on Apple [MLX](https://github.com/ml-explore/mlx).**

mlx-snn aims to provide an efficient, research-friendly SNN framework that leverages MLX's unified memory architecture and lazy evaluation. Whether you're exploring neuron dynamics, training classifiers with surrogate gradients, or exchanging models via [NIR](https://github.com/neuromorphs/NIR), mlx-snn offers a clean, Pythonic API that integrates naturally into the MLX ecosystem.

[![CI](https://github.com/D-ST-Sword/mlx-snn/actions/workflows/ci.yml/badge.svg)](https://github.com/D-ST-Sword/mlx-snn/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/mlx-snn.svg)](https://pypi.org/project/mlx-snn/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://d-st-sword.github.io/mlx-snn/)

## Highlights

<table>
<tr>
<td width="33%" valign="top">

**Unified Memory SNNs**

SNNs store per-neuron states across every timestep — a memory bottleneck on discrete-GPU architectures. Apple Silicon's unified memory eliminates CPU↔GPU transfers, enabling extended temporal windows and larger reservoirs without the VRAM wall.

</td>
<td width="33%" valign="top">

**17–25× Energy Efficiency**

M3 Max trains SNNs **2.6–3.7× faster** than Tesla V100 at 1/7th the power. Recurrent spiking dynamics are latency-bound, not compute-bound — favoring Apple Silicon's high-bandwidth unified architecture over datacenter parallelism.

</td>
<td width="33%" valign="top">

**Multi-Scale Temporal Modeling**

`MSLeaky` assigns frequency-matched decay rates to parallel spiking branches — capturing delta through gamma dynamics in a single network. Chunked BPTT with state detachment scales to long biosignal sequences (EEG, fMRI) without exploding memory.

</td>
</tr>
</table>

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
- **Arctan** — default, matches snnTorch's default for stable BPTT convergence
- **Fast Sigmoid** — rational approximation with heavier tails
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

## Benchmarks

All experiments use identical hyperparameters: Adam (LR=1e-3), Poisson rate encoding, T=25 timesteps, batch size 128, 5 random seeds, 10 epochs. Full scripts in [`benchmarks/`](benchmarks/).

### Training Accuracy (identical within noise)

| Task | mlx-snn (M3 Max) | snnTorch (V100) |
|------|-------------------|-----------------|
| FC SNN on MNIST | **97.0%** | 97.2% |
| FC SNN on FashionMNIST | **85.2%** | 85.2% |
| LSM Reservoir on MNIST | 92.4% | **93.6%** |

### Training Speed

| Task | mlx-snn (M3 Max) | snnTorch (V100) | Speedup |
|------|-------------------|-----------------|---------|
| FC SNN (784→128→10) | **7.4 s/epoch** | 19.4 s/epoch | **2.6x** |
| LSM Reservoir (500 neurons) | **3.2 s/epoch** | 5.7 s/epoch | **1.8x** |

### Inference Throughput (samples/sec, T=25)

| Model | Batch | mlx-snn (M3 Max) | snnTorch (V100) | Speedup |
|-------|-------|-------------------|-----------------|---------|
| FC-SNN | 128 | **23,875** | 6,552 | **3.6x** |
| FC-SNN | 512 | **95,735** | 26,058 | **3.7x** |
| Conv-SNN | 128 | **6,671** | 3,859 | **1.7x** |
| Conv-SNN | 512 | **6,434** | 4,252 | **1.5x** |

### `mx.compile` Acceleration

| Mode | Time (25-step forward) |
|------|----------------------|
| Uncompiled | 2.72 ms |
| Compiled | **0.92 ms** |
| **Speedup** | **2.9x** |

### Power Efficiency

The M3 Max (TDP ~45 W) delivers 2.6–3.7x faster SNN training/inference than the V100 (TDP 300 W), yielding an estimated **17–25x better energy efficiency** (performance per watt).

For detailed results and reproduction scripts, see [`benchmarks/`](benchmarks/) and our [benchmarking paper](https://arxiv.org/abs/2603.03529).

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
├── neurons/       # Leaky, IF, Izhikevich, ALIF, Synaptic, Alpha, RLeaky, RSynaptic
├── surrogate/     # arctan, fast_sigmoid, sigmoid, triangular, straight_through, custom
├── encoding/      # rate, latency, delta, direct, repeat, frequency-band, threshold-crossing, EEG
├── functional/    # Stateless pure functions, 9 loss functions, metrics
├── layers/        # SpikingConv2d, MaxPool2d, AvgPool2d, Flatten, SpikeDropout
├── operators/     # TAC, TAC-TP, L-TAC, FTC, IMC, TCC
├── liquid/        # LiquidReservoir, LSM, topology generators
├── datasets/      # DVSGesture, CIFAR10DVS, NMNIST, SHD, SSC
├── training/      # BPTT, chunked BPTT, mx.compile wrappers
├── utils/         # Visualization, state management
└── nir_*.py       # NIR export/import utilities
```

## Roadmap

- [x] **v0.1** — Core neurons (LIF, IF), surrogate gradients, rate/latency encoding
- [x] **v0.2** — Extended neurons (Izhikevich, ALIF, Synaptic, Alpha), EEG encoder, delta encoding
- [x] **v0.3** — NIR interoperability (export/import)
- [x] **v0.4** — Recurrent neurons, conv/pooling layers, neuromorphic datasets, TAC operators
- [x] **v0.5** — Direct/repeat encoding, activity regularization, SpikeDropout, visualization, SHD dataset
- [x] **v0.6** — CI/CD, API documentation site, complete examples
- [x] **v0.7** — LSM, MSLeaky neuron, chunked BPTT, `mx.compile`, frequency-band & threshold-crossing encoding
- [ ] **v1.0** — Full documentation, comprehensive benchmarks, JOSS paper

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

GPL-3.0
