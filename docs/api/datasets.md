# Datasets

Neuromorphic dataset loaders for event-driven vision and audio tasks.

!!! note "Optional dependency"
    Dataset loaders require additional packages:
    ```bash
    pip install mlx-snn[datasets]  # installs aedat, h5py
    ```

## DVS-Gesture

IBM DVS128 Gesture dataset — 11 hand gestures captured with a DVS camera.

::: mlxsnn.datasets.dvs_gesture.DVSGestureDataset

## CIFAR10-DVS

CIFAR-10 converted to neuromorphic events via a DVS camera.

::: mlxsnn.datasets.cifar10dvs.CIFAR10DVSDataset

## N-MNIST

Neuromorphic MNIST — handwritten digits captured with saccading DVS sensor.

::: mlxsnn.datasets.nmnist.NMNISTDataset

## SHD (Spiking Heidelberg Digits)

Spoken digit recognition from cochlea-like spike encoding.

::: mlxsnn.datasets.shd.SHDDataset

## Data Utilities

### Dataloader

::: mlxsnn.datasets.utils.EventDataloader

### Frame Processing

::: mlxsnn.datasets.utils.events_to_frames

::: mlxsnn.datasets.utils.resize_frames
