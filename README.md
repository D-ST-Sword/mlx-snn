# mlx-snn

**A general-purpose Spiking Neural Network library built on Apple [MLX](https://github.com/ml-explore/mlx).**

mlx-snn aims to provide an efficient, research-friendly SNN framework that leverages MLX's unified memory architecture and lazy evaluation. Whether you're exploring neuron dynamics, training classifiers with surrogate gradients, or exchanging models via [NIR](https://github.com/neuromorphs/NIR), mlx-snn offers a clean, Pythonic API that integrates naturally into the MLX ecosystem.

[![CI](https://github.com/D-ST-Sword/mlx-snn/actions/workflows/ci.yml/badge.svg)](https://github.com/D-ST-Sword/mlx-snn/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/mlx-snn.svg)](https://pypi.org/project/mlx-snn/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://d-st-sword.github.io/mlx-snn/)
[![arXiv](https://img.shields.io/badge/arXiv-2603.03529-b31b1b.svg)](https://arxiv.org/abs/2603.03529)

> **9 neuron models** · **6 surrogate gradients** · **8 spike encodings** · **5 neuromorphic datasets** · **LSM reservoir** · **NIR interop** · **403 tests**

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

All neurons support `learn_beta`, `learn_threshold`, and configurable reset mechanisms (`subtract` / `zero` / `none`). State is always an explicit dict — compatible with MLX's functional transforms and `mx.compile`.

| Model | Description | State Variables |
|-------|-------------|-----------------|
| **Leaky (LIF)** | Leaky Integrate-and-Fire with configurable decay | `mem` |
| **IF** | Integrate-and-Fire (non-leaky, perfect integrator) | `mem` |
| **Izhikevich** | 2D dynamics with RS/IB/CH/FS presets | `mem`, `recovery` |
| **ALIF** | Adaptive LIF with dynamic threshold | `mem`, `threshold` |
| **Synaptic** | Conductance-based dual-state LIF | `syn`, `mem` |
| **Alpha** | Dual-exponential synaptic model | `syn`, `mem` |
| **RLeaky** | Recurrent LIF with learnable feedback | `mem`, `spk` |
| **RSynaptic** | Recurrent Synaptic with learnable feedback | `syn`, `mem`, `spk` |
| **MSLeaky** | Multi-scale LIF with per-branch frequency-matched `beta` | `mem` per branch |

### Surrogate Gradients

All neurons support differentiable training through 6 surrogate gradient functions:

| Function | Formula (backward) | Properties |
|----------|-------------------|------------|
| **Arctan** (default) | `α / (2(1 + (πα x/2)²))` | Stable BPTT convergence, moderate locality |
| **Fast Sigmoid** | `scale / (1 + scale·\|x\|)²` | Heavier tails, smoother gradients |
| **Sigmoid** | `scale · σ(scale·x) · (1 − σ(scale·x))` | Standard logistic derivative |
| **Triangular** | `max(0, 1 − scale·\|x\|)` | Compact support, localized near threshold |
| **Straight-Through** | `1` | Simplest, unit gradient everywhere |
| **Custom** | User-defined | Plug in any differentiable approximation |

### Spike Encoding

8 encoding methods for converting continuous signals into spike trains:

| Method | Input | Use Case |
|--------|-------|----------|
| **Rate (Poisson)** | Static values | Images, general-purpose classification |
| **Latency (TTFS)** | Static values | Energy-efficient temporal coding |
| **Delta Modulation** | Temporal signals | Change detection, event-like encoding |
| **Direct** | Any tensor | Pass-through for pre-computed inputs |
| **Repeat** | Spike patterns | Tile spike trains across longer windows |
| **Frequency-Band** | EEG signals | FFT-based decomposition into delta/theta/alpha/beta/gamma bands |
| **Threshold-Crossing** | Temporal signals | Multi-level amplitude crossing detection |
| **EEG Encoder** | Raw EEG | Configurable rate/delta/threshold encoding for biosignals |

### Convolutional SNN Layers

Build deep spiking convolutional networks with spatial pooling:

```python
import mlxsnn

conv1 = mlxsnn.SpikingConv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
pool1 = mlxsnn.SpikingMaxPool2d(kernel_size=2, stride=2)
lif1  = mlxsnn.Leaky(beta=0.95)
drop  = mlxsnn.SpikeDropout(p=0.2)  # spike-aware (no rescaling)
flat  = mlxsnn.SpikingFlatten()
```

Also includes 6 mathematically-principled temporal operators for Conv SNNs: **TAC** (Temporal Aggregated Conv), **TAC-TP** (Temporal-Preserving), **L-TAC** (Learnable), **FTC** (Fourier Temporal Conv), **IMC** (InfoMax Spike Conv), **TCC** (Temporal Collapse Conv).

### Liquid State Machine

Reservoir computing with spiking neurons — random sparse recurrent connectivity with configurable topology:

```python
import mlxsnn

lsm = mlxsnn.LSM(
    input_size=64, reservoir_size=500, output_size=10,
    connectivity=0.1, spectral_radius=0.9,
    topology="small_world",  # also: "erdos_renyi", "scale_free"
    exc_ratio=0.8,           # Dale's law: 80% excitatory
)
state = lsm.init_state(batch_size=32)

for t in range(num_steps):
    output, state = lsm(spikes[t], state)
```

### Training Utilities

**BPTT variants:**
- `bptt_forward(model, spikes, state)` — standard backpropagation through time
- `chunked_bptt_forward(model, spikes, state, chunk_size)` — memory-efficient training on long sequences via state detachment at chunk boundaries
- `detach_state(state)` — detach all tensors in a state dict (for truncated BPTT)

**`mx.compile` wrappers:**
- `compiled_step(model)` — compile a single-timestep forward pass
- `compiled_forward(model, num_steps)` — compile a full temporal forward pass

**Loss functions (11 total):**

| Loss | Approach |
|------|----------|
| `ce_rate_loss` | Cross-entropy on spike rates (spike count / T) |
| `ce_count_loss` | Cross-entropy on raw spike counts |
| `mse_membrane_loss` | MSE on final membrane potential |
| `mse_count_loss` | MSE on spike counts vs targets |
| `membrane_loss` | Cross-entropy on final membrane potential |
| `rate_coding_loss` | Cross-entropy on log-softmax of spike counts |
| `activity_reg_loss` | Penalize deviation from target firing rate |
| `l1_spike_loss` | L1 sparsity penalty on spike counts |
| `l2_spike_loss` | L2 regularization on spike counts |

Utility functions: `spike_rate`, `spike_count`.

**Learnable parameters:** `learn_beta`, `learn_threshold`, `learn_V` on all neurons. Works with standard MLX optimizers (`mlx.optimizers.Adam`, etc.).

### Neuromorphic Datasets

5 built-in dataset loaders with automatic download, caching, and event-to-frame conversion:

| Dataset | Modality | Classes | Samples |
|---------|----------|---------|---------|
| **DVS-Gesture** | Event camera (hand gestures) | 11 | 1,342 |
| **CIFAR10-DVS** | Event camera (natural images) | 10 | 10,000 |
| **N-MNIST** | Event camera (digits) | 10 | 70,000 |
| **SHD** | Audio (spoken digits) | 20 | 10,420 |
| **SSC** | Audio (spoken commands) | 35 | 100,000+ |

```python
from mlxsnn.datasets import DVSGestureDataset, create_dataloader

dataset = DVSGestureDataset(root="./data", split="train", dt=5000)
loader = create_dataloader(dataset, batch_size=16, shuffle=True)
```

### Visualization

Requires `pip install mlx-snn[viz]` (matplotlib).

```python
from mlxsnn.utils.visualization import plot_raster, plot_membrane, plot_firing_rate

plot_raster(spike_tensor, title="Layer 1 Spikes")       # spike raster over time
plot_membrane(state["mem"], title="Membrane Potential")  # membrane trace
plot_firing_rate(spike_tensor, title="Firing Rates")     # per-neuron rates
```

### NIR Interoperability

[NIR](https://github.com/neuromorphs/NIR) enables cross-framework SNN model exchange — import/export models to snnTorch, Norse, SpikingJelly, and neuromorphic hardware.

```bash
pip install mlx-snn[nir]
```

```python
# Export
layers = [('fc1', nn.Linear(784, 128)), ('lif1', mlxsnn.Leaky(beta=0.9)),
          ('fc2', nn.Linear(128, 10)),   ('lif2', mlxsnn.Leaky(beta=0.9))]
nir.write('model.nir', mlxsnn.export_to_nir(layers))

# Import
model = mlxsnn.import_from_nir(nir.read('model.nir'))
out, state = model(x, model.init_states(batch_size=32))
```

Supported: `nn.Linear` ↔ `nir.Affine`/`nir.Linear`, `Leaky` ↔ `nir.LIF`, `IF` ↔ `nir.IF`, `Synaptic` ↔ `nir.CubaLIF`.

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

| | snnTorch (PyTorch) | mlx-snn (MLX) |
|---|---|---|
| Import | `import snntorch as snn` | `import mlxsnn` |
| Create | `lif = snn.Leaky(beta=0.9)` | `lif = mlxsnn.Leaky(beta=0.9)` |
| Forward | `spk, mem = lif(x, mem)` | `spk, state = lif(x, state)` |
| State | Separate tensors (`mem`) | Explicit dict (`state["mem"]`) |
| Tensors | `torch.Tensor` | `mx.array` |
| Gradients | `autograd` + surrogate | STE pattern + `mx.stop_gradient` |

Key design difference: **state is always an explicit dict** — pass in, get out. No hidden instance variables. This plays well with MLX's functional transforms (`mx.grad`, `mx.vmap`, `mx.compile`).

## Project Structure

```
mlxsnn/
├── neurons/       # Leaky, IF, Izhikevich, ALIF, Synaptic, Alpha, RLeaky, RSynaptic, MSLeaky
├── surrogate/     # arctan, fast_sigmoid, sigmoid, triangular, straight_through, custom
├── encoding/      # rate, latency, delta, direct, repeat, frequency-band, threshold-crossing, EEG
├── functional/    # Stateless pure functions, 11 loss functions, metrics
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
