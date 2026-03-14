# Changelog

All notable changes to mlx-snn are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.5.0] - 2026-03-14

### Added
- **Direct encoding** (`direct_encode`) — repeat static data across timesteps
- **Repeat encoding** (`repeat_encode`) — tile spike patterns N times
- **Activity regularization losses** — `activity_reg_loss`, `l1_spike_loss`, `l2_spike_loss`
- **SpikeDropout** layer — spike-aware dropout without rescaling (binary semantics)
- **Visualization utilities** — `plot_raster`, `plot_membrane`, `plot_firing_rate`
- **SHD dataset loader** — Spiking Heidelberg Digits with HDF5 support

### Changed
- Added `numpy` to core dependencies (required by dataset loaders)

## [0.4.0] - 2026-03-05

### Added
- **Recurrent neurons** — `RLeaky` (recurrent LIF), `RSynaptic` (recurrent Synaptic)
- **Expanded surrogate gradients** — `sigmoid`, `triangular` added alongside existing `fast_sigmoid`, `arctan`, `straight_through`
- **Learnable thresholds** — all neurons support `learn_threshold=True`
- **Mathematically-principled Conv SNN operators** — TAC, TAC-TP, L-TAC, FourierTemporalConv, InfoMaxSpikeConv, TemporalCollapseConv
- **Composite layers** — `SpikingConv2d`, `SpikingMaxPool2d`, `SpikingAvgPool2d`, `SpikingFlatten`
- **Neuromorphic datasets** — `DVSGestureDataset`, `CIFAR10DVSDataset`, `NMNISTDataset`
- **Event data utilities** — `events_to_frames`, `EventDataloader`, `create_dataloader`
- Additional loss functions: `ce_rate_loss`, `ce_count_loss`, `mse_membrane_loss`, `mse_count_loss`
- `spike_rate` and `spike_count` utility functions

## [0.3.0] - 2026-02-28

### Added
- **NIR interoperability** — `export_to_nir`, `import_from_nir` for cross-platform SNN model exchange
- `NIRSequential` wrapper for importing NIR graphs

## [0.2.1] - 2026-02-25

### Fixed
- Rational fast sigmoid to match snnTorch surrogate gradient behavior

## [0.2.0] - 2026-02-22

### Added
- **Izhikevich neuron** with RS/IB/CH/FS presets
- **Adaptive LIF** (ALIF) with learnable adaptation
- **Synaptic neuron** — conductance-based dual-state LIF
- **Alpha neuron** — dual-exponential synaptic model
- **EEG encoder** — rate/delta/threshold-crossing encoding for EEG signals
- **Delta encoding** — change-based spike encoding for temporal signals

## [0.1.0] - 2026-02-18

### Added
- **Leaky Integrate-and-Fire** (LIF) neuron with learnable `beta`
- **Integrate-and-Fire** (IF) neuron
- **Surrogate gradients** — `fast_sigmoid`, `arctan`, `straight_through`
- **Spike encoding** — `rate_encode`, `latency_encode`
- **Functional API** — `lif_step`, `if_step`, `fire`, `reset_subtract`, `reset_zero`
- **Loss functions** — `rate_coding_loss`, `membrane_loss`
- **BPTT helper** — `bptt_forward`
- MNIST quickstart example

[0.5.0]: https://github.com/D-ST-Sword/mlx-snn/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/D-ST-Sword/mlx-snn/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/D-ST-Sword/mlx-snn/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/D-ST-Sword/mlx-snn/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/D-ST-Sword/mlx-snn/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/D-ST-Sword/mlx-snn/releases/tag/v0.1.0
