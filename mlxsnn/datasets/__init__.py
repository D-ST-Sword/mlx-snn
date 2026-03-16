"""Neuromorphic dataset loaders for mlx-snn.

Provides lightweight, dependency-minimal loaders for popular event-based
vision datasets.  All loaders return ``mx.array`` tensors in
**time-first NHWC** format: ``(T, H, W, C)`` per sample, and
``(B, T, H, W, C)`` when batched.

Available datasets:
    - :class:`DVSGestureDataset` -- IBM DVS128 Gesture (11 classes, 128x128)
    - :class:`CIFAR10DVSDataset` -- CIFAR10-DVS (10 classes, 128x128)
    - :class:`NMNISTDataset` -- Neuromorphic MNIST (10 classes, 34x34)

Utilities:
    - :func:`events_to_frames` -- bin raw events into dense frame tensors
    - :class:`EventDataloader` / :func:`create_dataloader` -- simple
      batch iterator (no PyTorch dependency)
"""

from mlxsnn.datasets.cifar10dvs import CIFAR10DVSDataset
from mlxsnn.datasets.dvs_gesture import DVSGestureDataset
from mlxsnn.datasets.nmnist import NMNISTDataset
from mlxsnn.datasets.utils import (
    EventDataloader,
    create_dataloader,
    events_to_frames,
    resize_frames,
)

# SHD/SSC require h5py (optional)
try:
    from mlxsnn.datasets.shd import SHDDataset
except ImportError:
    pass

try:
    from mlxsnn.datasets.ssc import SSCDataset
except ImportError:
    pass

__all__ = [
    "DVSGestureDataset",
    "CIFAR10DVSDataset",
    "NMNISTDataset",
    "SHDDataset",
    "SSCDataset",
    "events_to_frames",
    "resize_frames",
    "EventDataloader",
    "create_dataloader",
]
