"""CIFAR10-DVS dataset loader for mlx-snn.

Loads ``.aedat4`` event files from the CIFAR10-DVS dataset, bins them into
dense frames, and returns ``mx.array`` tensors ready for SNN training.

Dataset details:
    - Sensor: 128x128 pixels, 2 polarities
    - Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog,
      horse, ship, truck)
    - No official train/test split -- the loader creates a deterministic
      90/10 split (seed-controlled) consistent with common practice.
    - Directory layout (after extraction by *tonic* or manual download)::

        CIFAR10DVS/
            airplane/
                cifar10_airplane_0.aedat4
                cifar10_airplane_1.aedat4
                ...
            automobile/
                ...

    Each ``.aedat4`` file is read with the ``aedat`` package and produces
    a structured numpy array with fields ``(t, x, y, p)`` per packet.

Dependencies:
    Requires the ``aedat`` Python package (``pip install aedat``).

References:
    Li et al., "CIFAR10-DVS: An Event-Stream Dataset for Object
    Classification", Frontiers in Neuroscience, 2017.
"""

from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from mlxsnn.datasets.utils import events_to_frames, resize_frames

# Class name -> integer label mapping (alphabetical order, matching tonic)
_CLASS_NAMES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
_CLASS_TO_IDX = {name: idx for idx, name in enumerate(_CLASS_NAMES)}


class CIFAR10DVSDataset:
    """CIFAR10-DVS dataset loader for mlx-snn.

    Loads event data from ``.aedat4`` files using the lightweight
    ``aedat`` package, bins into dense frames, and returns ``mx.array``
    tensors.

    Because CIFAR10-DVS has no official train/test split, the loader
    deterministically assigns 90 % of samples to training and 10 % to
    testing (controlled by ``split_seed``).

    Args:
        root: Path to the ``CIFAR10DVS`` directory containing class
            sub-directories (``airplane/``, ``automobile/``, etc.).
        train: If ``True``, use the training portion of the split.
        num_steps: Number of temporal bins (timesteps T).
        spatial_size: Target spatial resolution ``(H, W)``.
        transform: Optional callable applied to each frame tensor
            **after** conversion to ``mx.array``.
        split_ratio: Fraction of data used for training (default 0.9).
        split_seed: Random seed for the train/test split (default 42).

    Attributes:
        samples: List of ``(file_path, label)`` tuples.
        classes: List of class names.
        num_classes: ``10``.
        sensor_size: Native sensor resolution ``(128, 128)``.

    Examples:
        >>> ds = CIFAR10DVSDataset("/data/CIFAR10DVS", train=True,
        ...                        num_steps=10)
        >>> frames, label = ds[0]
        >>> frames.shape
        [10, 128, 128, 2]
    """

    SENSOR_SIZE: Tuple[int, int] = (128, 128)
    NUM_POLARITIES: int = 2
    NUM_CLASSES: int = 10

    def __init__(
        self,
        root: str,
        train: bool = True,
        num_steps: int = 10,
        spatial_size: Tuple[int, int] = (128, 128),
        transform: Optional[Callable[[mx.array], mx.array]] = None,
        split_ratio: float = 0.9,
        split_seed: int = 42,
    ):
        self.root = root
        self.train = train
        self.num_steps = num_steps
        self.spatial_size = spatial_size
        self.transform = transform

        self.sensor_size = self.SENSOR_SIZE
        self.num_classes = self.NUM_CLASSES
        self.classes = list(_CLASS_NAMES)

        all_samples = self._scan_samples(root)
        self.samples = self._split(
            all_samples, train=train, ratio=split_ratio, seed=split_seed
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[mx.array, int]:
        """Load and bin a single CIFAR10-DVS sample.

        Args:
            idx: Sample index.

        Returns:
            ``(frames, label)`` where ``frames`` is ``mx.array`` of
            shape ``(T, H, W, 2)`` and ``label`` is ``int``.
        """
        path, label = self.samples[idx]
        events = self._load_aedat4(path)

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
    def _load_aedat4(path: str) -> np.ndarray:
        """Read an ``.aedat4`` file and return a structured events array.

        Returns a structured ``np.ndarray`` with fields
        ``(t, x, y, p)`` -- the same field names that the ``aedat``
        package produces.

        Raises:
            ImportError: If the ``aedat`` package is not installed.
        """
        try:
            import aedat as _aedat
        except ImportError as exc:
            raise ImportError(
                "The 'aedat' package is required for loading CIFAR10-DVS "
                "(.aedat4 files).  Install it with: pip install aedat"
            ) from exc

        decoder = _aedat.Decoder(path)
        chunks: List[np.ndarray] = []
        for packet in decoder:
            if "events" in packet:
                chunks.append(packet["events"])

        if not chunks:
            # Return an empty structured array with the expected dtype
            dtype = np.dtype(
                [("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "?")]
            )
            return np.empty(0, dtype=dtype)

        return np.concatenate(chunks)

    @staticmethod
    def _scan_samples(root: str) -> List[Tuple[str, int]]:
        """Collect all ``(file_path, label)`` pairs from class dirs."""
        samples: List[Tuple[str, int]] = []
        for class_name in sorted(_CLASS_NAMES):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = _CLASS_TO_IDX[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if fname.endswith(".aedat4"):
                    samples.append((os.path.join(class_dir, fname), label))
        return samples

    @staticmethod
    def _split(
        samples: List[Tuple[str, int]],
        train: bool,
        ratio: float,
        seed: int,
    ) -> List[Tuple[str, int]]:
        """Deterministic train/test split."""
        rng = np.random.RandomState(seed)
        indices = np.arange(len(samples))
        rng.shuffle(indices)
        n_train = int(len(samples) * ratio)
        if train:
            sel = indices[:n_train]
        else:
            sel = indices[n_train:]
        return [samples[i] for i in sel]

    def __repr__(self) -> str:
        split = "train" if self.train else "test"
        return (
            f"CIFAR10DVSDataset(root={self.root!r}, split={split!r}, "
            f"num_steps={self.num_steps}, spatial_size={self.spatial_size}, "
            f"samples={len(self.samples)})"
        )
