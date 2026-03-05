"""Neuromorphic MNIST (N-MNIST) dataset loader for mlx-snn.

Loads ``.bin`` event files from the N-MNIST dataset, bins them into dense
frames, and returns ``mx.array`` tensors ready for SNN training.

Dataset details:
    - Sensor: ATIS, 34x34 pixels, 2 polarities
    - Classes: 10 (digits 0 -- 9)
    - Directory layout (after extraction by *tonic* or manual download)::

        NMNIST/
            Train/
                0/
                    00002.bin
                    00022.bin
                    ...
                1/
                    ...
            Test/
                0/
                    ...

    Each ``.bin`` file stores events in the standard N-MNIST binary
    format: **5 bytes per event** -- ``[x (1B), y (1B), pol|timestamp
    (3B big-endian)]``.  The 3-byte field packs polarity in bit 23 and a
    23-bit timestamp (microseconds) in bits 0--22.

References:
    Orchard et al., "Converting Static Image Datasets to Spiking
    Neuromorphic Datasets Using Saccades", Frontiers in Neuroscience, 2015.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from mlxsnn.datasets.utils import events_to_frames, resize_frames


class NMNISTDataset:
    """Neuromorphic MNIST dataset loader for mlx-snn.

    Loads event data from ``.bin`` files, bins events into dense frames,
    and returns ``mx.array`` tensors suitable for spiking network input.

    Args:
        root: Path to the ``NMNIST`` directory containing ``Train/`` and
            ``Test/`` sub-directories.
        train: If ``True``, load the training split; otherwise the test
            split.
        num_steps: Number of temporal bins (timesteps T).
        spatial_size: Target spatial resolution ``(H, W)``.  Defaults to
            the native ``(34, 34)``.
        transform: Optional callable applied to each frame tensor
            **after** conversion to ``mx.array``.

    Attributes:
        samples: List of ``(file_path, label)`` tuples.
        classes: List of class indices ``[0, 1, ..., 9]``.
        num_classes: ``10``.
        sensor_size: Native sensor resolution ``(34, 34)``.

    Examples:
        >>> ds = NMNISTDataset("/data/NMNIST", train=True, num_steps=20)
        >>> len(ds)
        60000
        >>> frames, label = ds[0]
        >>> frames.shape
        [20, 34, 34, 2]
    """

    SENSOR_SIZE: Tuple[int, int] = (34, 34)
    NUM_POLARITIES: int = 2
    NUM_CLASSES: int = 10

    def __init__(
        self,
        root: str,
        train: bool = True,
        num_steps: int = 20,
        spatial_size: Tuple[int, int] = (34, 34),
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

        split_dir = "Train" if train else "Test"
        split_path = os.path.join(root, split_dir)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"Split directory not found: {split_path}. "
                f"Ensure the N-MNIST dataset is extracted under {root}."
            )

        self.samples: List[Tuple[str, int]] = self._scan_samples(split_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[mx.array, int]:
        """Load and bin a single N-MNIST sample.

        Args:
            idx: Sample index.

        Returns:
            ``(frames, label)`` where ``frames`` is ``mx.array`` of
            shape ``(T, H, W, 2)`` and ``label`` is ``int``.
        """
        path, label = self.samples[idx]
        events = self._load_bin(path)

        frames = events_to_frames(
            events,
            num_steps=self.num_steps,
            sensor_size=self.SENSOR_SIZE,
            num_polarities=self.NUM_POLARITIES,
        )

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
    def _load_bin(path: str) -> np.ndarray:
        """Parse an N-MNIST ``.bin`` file into an ``(N, 4)`` event array.

        The binary format stores 5 bytes per event:
            - byte 0: x address  (uint8)
            - byte 1: y address  (uint8)
            - bytes 2-4: polarity (bit 23) | timestamp in microseconds
              (bits 0--22), big-endian.

        Returns:
            ``np.ndarray`` of shape ``(N, 4)`` with columns
            ``[x, y, polarity, timestamp]`` and dtype ``float64``.
        """
        with open(path, "rb") as f:
            raw = f.read()

        n_events = len(raw) // 5
        if n_events == 0:
            return np.zeros((0, 4), dtype=np.float64)

        # Vectorised parsing
        raw_arr = np.frombuffer(raw, dtype=np.uint8)
        raw_arr = raw_arr[: n_events * 5].reshape(n_events, 5)

        x = raw_arr[:, 0].astype(np.float64)
        y = raw_arr[:, 1].astype(np.float64)

        # 3 bytes big-endian => 24-bit integer
        combined = (
            raw_arr[:, 2].astype(np.int64) << 16
            | raw_arr[:, 3].astype(np.int64) << 8
            | raw_arr[:, 4].astype(np.int64)
        )
        polarity = ((combined >> 23) & 1).astype(np.float64)
        timestamp = (combined & 0x7FFFFF).astype(np.float64)

        events = np.column_stack([x, y, polarity, timestamp])
        return events

    @staticmethod
    def _scan_samples(split_path: str) -> List[Tuple[str, int]]:
        """Collect all ``(file_path, label)`` pairs from class dirs."""
        samples: List[Tuple[str, int]] = []
        class_dirs = sorted(
            d
            for d in os.listdir(split_path)
            if os.path.isdir(os.path.join(split_path, d)) and d.isdigit()
        )
        for class_name in class_dirs:
            label = int(class_name)
            class_path = os.path.join(split_path, class_name)
            for fname in sorted(os.listdir(class_path)):
                if fname.endswith(".bin"):
                    samples.append(
                        (os.path.join(class_path, fname), label)
                    )
        return samples

    def __repr__(self) -> str:
        split = "train" if self.train else "test"
        return (
            f"NMNISTDataset(root={self.root!r}, split={split!r}, "
            f"num_steps={self.num_steps}, spatial_size={self.spatial_size}, "
            f"samples={len(self.samples)})"
        )
