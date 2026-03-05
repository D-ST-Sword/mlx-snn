"""DVS128 Gesture dataset loader for mlx-snn.

Loads pre-extracted ``.npy`` event files from the IBM DVS128 Gesture dataset,
bins them into dense frames, and returns ``mx.array`` tensors ready for SNN
training.

Dataset details:
    - Sensor: iniVation DVS128, 128x128 pixels, 2 polarities
    - Classes: 11 gesture types (0 -- 10)
    - Directory layout (after extraction by *tonic* or manual download)::

        DVSGesture/
            ibmGestureTrain/
                user01_fluorescent/
                    0.npy   # gesture class 0
                    1.npy   # gesture class 1
                    ...
                    10.npy  # gesture class 10
                user01_fluorescent_led/
                    ...
            ibmGestureTest/
                ...

    Each ``.npy`` file is a ``(N, 4)`` float64 array with columns
    ``[x, y, polarity, timestamp_seconds]``, sorted by timestamp.

References:
    Amir et al., "A Low Power, Fully Event-Based Gesture Recognition
    System", CVPR 2017.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from mlxsnn.datasets.utils import events_to_frames, resize_frames


class DVSGestureDataset:
    """DVS128 Gesture dataset loader for mlx-snn.

    Loads event data from ``.npy`` files, bins into dense frames, and
    returns ``mx.array`` tensors suitable for spiking network input.

    Args:
        root: Path to the ``DVSGesture`` directory that contains
            ``ibmGestureTrain/`` and ``ibmGestureTest/``.
        train: If ``True``, load the training split; otherwise the test
            split.
        num_steps: Number of temporal bins (timesteps T).
        spatial_size: Target spatial resolution ``(H, W)``.  If different
            from the native 128x128, frames are resized.
        transform: Optional callable applied to each frame tensor
            **after** conversion to ``mx.array``.  Receives and must
            return an ``mx.array`` of shape ``(T, H, W, 2)``.

    Attributes:
        samples: List of ``(file_path, label)`` tuples.
        classes: List of class indices ``[0, 1, ..., 10]``.
        num_classes: ``11``.
        sensor_size: Native sensor resolution ``(128, 128)``.

    Examples:
        >>> ds = DVSGestureDataset("/data/DVSGesture", train=True,
        ...                        num_steps=16)
        >>> len(ds)
        1077
        >>> frames, label = ds[0]
        >>> frames.shape
        [16, 128, 128, 2]
    """

    SENSOR_SIZE: Tuple[int, int] = (128, 128)
    NUM_POLARITIES: int = 2
    NUM_CLASSES: int = 11

    def __init__(
        self,
        root: str,
        train: bool = True,
        num_steps: int = 16,
        spatial_size: Tuple[int, int] = (128, 128),
        transform: Optional[Callable[[mx.array], mx.array]] = None,
    ):
        self.root = root
        self.train = train
        self.num_steps = num_steps
        self.spatial_size = spatial_size
        self.transform = transform

        self.sensor_size = self.SENSOR_SIZE
        self.num_classes = self.NUM_CLASSES
        self.classes = list(range(self.NUM_CLASSES))

        split_dir = "ibmGestureTrain" if train else "ibmGestureTest"
        split_path = os.path.join(root, split_dir)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"Split directory not found: {split_path}. "
                f"Ensure the DVSGesture dataset is extracted under {root}."
            )

        self.samples: List[Tuple[str, int]] = self._scan_samples(split_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[mx.array, int]:
        """Load and bin a single sample.

        Args:
            idx: Sample index.

        Returns:
            A tuple ``(frames, label)`` where ``frames`` is an
            ``mx.array`` of shape ``(T, H, W, 2)`` and ``label`` is an
            ``int`` in ``[0, 10]``.
        """
        path, label = self.samples[idx]
        events = np.load(path)  # (N, 4): [x, y, polarity, timestamp]

        frames = events_to_frames(
            events,
            num_steps=self.num_steps,
            sensor_size=self.SENSOR_SIZE,
            num_polarities=self.NUM_POLARITIES,
        )

        # Resize if spatial_size differs from native 128x128
        if self.spatial_size != self.SENSOR_SIZE:
            frames = resize_frames(frames, self.spatial_size)

        frames_mx = mx.array(frames)

        if self.transform is not None:
            frames_mx = self.transform(frames_mx)

        return frames_mx, label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_samples(split_path: str) -> List[Tuple[str, int]]:
        """Walk through user directories and collect (path, label) pairs.

        Each user folder contains ``0.npy`` through ``10.npy``, where the
        filename (without extension) is the gesture class label.
        """
        samples: List[Tuple[str, int]] = []
        user_dirs = sorted(
            d
            for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d))
        )
        for user_dir in user_dirs:
            user_path = os.path.join(split_path, user_dir)
            for fname in sorted(os.listdir(user_path)):
                if not fname.endswith(".npy"):
                    continue
                label = int(os.path.splitext(fname)[0])
                samples.append((os.path.join(user_path, fname), label))
        return samples

    def __repr__(self) -> str:
        split = "train" if self.train else "test"
        return (
            f"DVSGestureDataset(root={self.root!r}, split={split!r}, "
            f"num_steps={self.num_steps}, spatial_size={self.spatial_size}, "
            f"samples={len(self.samples)})"
        )
