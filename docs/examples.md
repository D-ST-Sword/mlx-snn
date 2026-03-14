# Examples

Complete runnable examples are available in the [`examples/`](https://github.com/D-ST-Sword/mlx-snn/tree/main/examples) directory.

## 01 — Quick Start: MNIST with LIF

A minimal fully-connected SNN trained on MNIST using rate encoding and surrogate gradient BPTT.

```bash
python examples/01_quickstart.py
```

- Rate encoding of static images
- 2-layer FC SNN (784 → 128 → 10)
- `mse_membrane_loss` for training
- ~96% accuracy in 10 epochs

## 02 — Conv SNN on DVS-Gesture

Convolutional SNN for event-driven gesture recognition using DVS128 camera data.

```bash
python examples/02_conv_snn_dvsg.py --data_dir /path/to/DVSGesture
```

- 3-block Conv SNN with `SpikingConv2d` + `SpikingMaxPool2d`
- Processes native event camera data (128x128, 2 polarity channels)
- 11-class gesture classification

## 03 — TAC Speedup Benchmark

Benchmarks the Temporal Aggregated Convolution (TAC) operator against standard per-timestep convolution.

```bash
python examples/03_tac_speedup.py
```

- Compares forward pass timing
- Tests chunk sizes K = 2, 4, 8, 16
- Demonstrates near-linear speedup

## 04 — Visualization

Demonstrates mlx-snn's built-in visualization utilities for inspecting network dynamics.

```bash
pip install mlx-snn[viz]
python examples/04_visualization.py
```

- Spike raster plots
- Membrane potential traces
- Per-neuron firing rate bar charts

## 05 — SHD Audio Classification

Fully-connected SNN for spoken digit recognition on the Spiking Heidelberg Digits dataset.

```bash
pip install mlx-snn[datasets]
python examples/05_shd_audio.py --data_dir /path/to/shd
```

- 700-channel cochlea input (HDF5 format)
- 20-class classification (digits 0-9, English + German)
- 3-layer FC SNN with LIF neurons

## 06 — Custom Neuron Model

Tutorial on subclassing `SpikingNeuron` to create new neuron models.

```bash
python examples/06_custom_neuron.py
```

- Implements Exponential Integrate-and-Fire (EIF) neuron
- Demonstrates `init_state`, `fire`, `reset` API
- Trains on synthetic classification task
