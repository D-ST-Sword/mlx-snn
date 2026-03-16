"""Spiking Speech Commands (SSC) dataset loader.

Loads the SSC dataset from HDF5 files and bins spike times into dense
frame tensors.  The SSC dataset contains spoken commands encoded as spike
times across 700 input channels, forming a 35-class classification task.

Dataset details:
    - Channels: 700 (cochlea-like filterbank)
    - Classes: 35 (speech commands)
    - Training samples: ~75,466
    - Validation samples: ~9,981
    - Test samples: ~20,382
    - File format: HDF5 with ``spikes/times``, ``spikes/units``, ``labels``

References:
    Cramer, B., Stradmann, Y., Schemmel, J., & Zenke, F. (2020).
    The Heidelberg Spiking Data Sets for the Systematic Evaluation
    of Spiking Neural Networks. IEEE TNNLS.
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import mlx.core as mx
import numpy as np


def _ensure_h5py():
    """Lazily import h5py, raising a clear error if missing."""
    try:
        import h5py
        return h5py
    except ImportError:
        raise ImportError(
            "h5py is required for loading SSC datasets. "
            "Install it with: pip install mlx-snn[datasets]"
        )


class SSCDataset:
    """Spiking Speech Commands dataset loader.

    Loads spike times from HDF5 files and bins them into dense
    binary tensors of shape ``(T, 700)``.

    Args:
        root: Path to directory containing ``ssc_train.h5``,
            ``ssc_valid.h5``, and ``ssc_test.h5``.
        split: Which split to load: ``'train'``, ``'valid'``, or
            ``'test'``.
        num_steps: Number of temporal bins (timesteps T).
        max_time: Maximum spike time in seconds.  Spikes after this
            time are discarded.  If ``None``, uses the maximum time
            found in the data.
        num_units: Number of input channels (default 700).
        transform: Optional callable applied to each sample after
            conversion to ``mx.array``.

    Examples:
        >>> from mlxsnn.datasets import SSCDataset
        >>> ds = SSCDataset("./data/ssc", split="train", num_steps=100)
        >>> spikes, label = ds[0]
        >>> spikes.shape
        [100, 700]
    """

    NUM_CLASSES = 35

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_steps: int = 100,
        max_time: Optional[float] = None,
        num_units: int = 700,
        transform: Optional[Callable] = None,
    ):
        if split not in ("train", "valid", "test"):
            raise ValueError(
                f"split must be 'train', 'valid', or 'test', got '{split}'"
            )
        self.root = root
        self.split = split
        self.num_steps = num_steps
        self.max_time = max_time
        self.num_units = num_units
        self.transform = transform

        h5py = _ensure_h5py()
        fname = f"ssc_{split}.h5"
        filepath = os.path.join(root, fname)

        with h5py.File(filepath, "r") as f:
            times_ds = f["spikes/times"]
            units_ds = f["spikes/units"]
            # Handle both ragged datasets (variable-length arrays) and
            # group-of-datasets formats.
            if hasattr(times_ds, "keys"):
                keys = sorted(times_ds.keys(), key=int)
                self.spike_times = [np.array(times_ds[k]) for k in keys]
                self.spike_units = [
                    np.array(units_ds[k]).astype(int) for k in keys
                ]
            else:
                n = times_ds.shape[0]
                self.spike_times = [np.array(times_ds[i]) for i in range(n)]
                self.spike_units = [
                    np.array(units_ds[i]).astype(int) for i in range(n)
                ]
            self.labels = np.array(f["labels"]).astype(int)

        # Determine max_time from data if not provided
        if self.max_time is None:
            self.max_time = max(
                t.max() for t in self.spike_times if t.size > 0
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[mx.array, int]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of ``(spikes, label)`` where spikes is
            ``mx.array`` of shape ``[T, num_units]`` and label
            is an integer.
        """
        spikes = self._bin_spikes(
            self.spike_times[idx],
            self.spike_units[idx],
        )
        spikes = mx.array(spikes)
        if self.transform is not None:
            spikes = self.transform(spikes)
        return spikes, int(self.labels[idx])

    def _bin_spikes(
        self,
        times: np.ndarray,
        units: np.ndarray,
    ) -> np.ndarray:
        """Bin spike times into a dense ``(T, num_units)`` array.

        Args:
            times: Spike timestamps in seconds.
            units: Channel indices for each spike.

        Returns:
            Binary numpy array of shape ``(T, num_units)``.
        """
        frame = np.zeros((self.num_steps, self.num_units), dtype=np.float32)
        if times.size == 0:
            return frame

        # Clip to max_time and compute bin indices
        valid = times <= self.max_time
        times = times[valid]
        units = units[valid]

        bin_idx = np.floor(times / self.max_time * self.num_steps).astype(int)
        bin_idx = np.clip(bin_idx, 0, self.num_steps - 1)
        units = np.clip(units, 0, self.num_units - 1)

        frame[bin_idx, units] = 1.0
        return frame

    def __repr__(self) -> str:
        return (
            f"SSCDataset(split={self.split}, samples={len(self)}, "
            f"num_steps={self.num_steps}, num_units={self.num_units})"
        )
