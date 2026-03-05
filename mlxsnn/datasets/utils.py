"""Shared utilities for neuromorphic dataset loading.

Provides event-to-frame binning, a lightweight dataloader, and common
preprocessing functions used by all event-based dataset classes.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, List, Optional, Tuple

import mlx.core as mx
import numpy as np


def events_to_frames(
    events: np.ndarray,
    num_steps: int,
    sensor_size: Tuple[int, int],
    num_polarities: int = 2,
) -> np.ndarray:
    """Convert raw events to temporally-binned frames.

    Divides the full event time range into ``num_steps`` equal bins and
    accumulates event counts per pixel and polarity into dense frames.

    Args:
        events: Structured array or ``(N, 4)`` array with columns
            ``[x, y, polarity, timestamp]``.  Timestamps must be sorted in
            non-decreasing order.  Polarity values are integers in
            ``[0, num_polarities)``.
        num_steps: Number of temporal bins (T).
        sensor_size: Spatial resolution as ``(H, W)``.
        num_polarities: Number of polarity channels (default 2).

    Returns:
        frames: ``np.ndarray`` of shape ``(T, H, W, P)`` with dtype
            ``np.float32``, where P = ``num_polarities``.

    Examples:
        >>> import numpy as np
        >>> events = np.array([[10, 20, 1, 100],
        ...                    [11, 21, 0, 200]], dtype=np.float64)
        >>> frames = events_to_frames(events, num_steps=2,
        ...                           sensor_size=(34, 34))
        >>> frames.shape
        (2, 34, 34, 2)
    """
    H, W = sensor_size
    frames = np.zeros((num_steps, H, W, num_polarities), dtype=np.float32)

    if len(events) == 0:
        return frames

    # Extract columns -------------------------------------------------------
    if events.dtype.names is not None:
        # Structured array (e.g. from aedat)
        x = events["x"].astype(np.int64)
        y = events["y"].astype(np.int64)
        p = events["p"].astype(np.int64)
        t = events["t"].astype(np.float64)
    else:
        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        p = events[:, 2].astype(np.int64)
        t = events[:, 3].astype(np.float64)

    # Clip spatial coordinates to sensor bounds
    np.clip(x, 0, W - 1, out=x)
    np.clip(y, 0, H - 1, out=y)
    np.clip(p, 0, num_polarities - 1, out=p)

    # Compute bin indices ---------------------------------------------------
    t_min, t_max = t.min(), t.max()
    if t_max <= t_min:
        # All events at the same timestamp -- put everything in the first bin
        bin_idx = np.zeros(len(t), dtype=np.int64)
    else:
        bin_idx = ((t - t_min) / (t_max - t_min) * num_steps).astype(np.int64)
        np.clip(bin_idx, 0, num_steps - 1, out=bin_idx)

    # Accumulate events into frames using np.add.at for performance ---------
    np.add.at(frames, (bin_idx, y, x, p), 1.0)

    return frames


def resize_frames(
    frames: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Resize spatial dimensions of a frame tensor using area-based binning.

    Uses simple block averaging (area interpolation) which is appropriate
    for event count data.  No external dependency required.

    Args:
        frames: ``(T, H, W, C)`` float array.
        target_size: ``(H_new, W_new)``.

    Returns:
        Resized frames with shape ``(T, H_new, W_new, C)``.
    """
    _, H, W, C = frames.shape
    H_new, W_new = target_size

    if (H, W) == (H_new, W_new):
        return frames

    T = frames.shape[0]
    out = np.zeros((T, H_new, W_new, C), dtype=frames.dtype)

    # Block averaging via reshape (works when target divides source evenly)
    if H % H_new == 0 and W % W_new == 0:
        bh = H // H_new
        bw = W // W_new
        out = frames.reshape(T, H_new, bh, W_new, bw, C).mean(axis=(2, 4))
        return out.astype(frames.dtype)

    # Fallback: nearest-neighbour sampling
    row_idx = (np.arange(H_new) * H / H_new).astype(int)
    col_idx = (np.arange(W_new) * W / W_new).astype(int)
    out = frames[:, row_idx][:, :, col_idx]
    return out


class EventDataloader:
    """Simple batch iterator for MLX event-based datasets.

    Produces batches of ``(frames, labels)`` as ``mx.array`` tensors without
    depending on PyTorch or any external data-loading library.

    Args:
        dataset: A dataset object that supports ``__len__`` and
            ``__getitem__``, returning ``(mx.array, int)`` pairs.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset at the start of each epoch.
        drop_last: If ``True``, drop the final incomplete batch.
        seed: Random seed for shuffling reproducibility.

    Yields:
        Tuple of ``(batch_frames, batch_labels)`` where
        ``batch_frames`` has shape ``(B, T, H, W, C)`` and
        ``batch_labels`` has shape ``(B,)``.

    Examples:
        >>> loader = EventDataloader(dataset, batch_size=32, shuffle=True)
        >>> for frames, labels in loader:
        ...     print(frames.shape, labels.shape)
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[Tuple[mx.array, mx.array]]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, n, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                break

            frames_list: List[np.ndarray] = []
            labels_list: List[int] = []
            for idx in batch_idx:
                frame, label = self.dataset[int(idx)]
                # Materialise mx.array to numpy for stacking
                if isinstance(frame, mx.array):
                    frame = np.array(frame)
                frames_list.append(frame)
                labels_list.append(label)

            batch_frames = mx.array(np.stack(frames_list, axis=0))  # (B, T, H, W, C)
            batch_labels = mx.array(np.array(labels_list, dtype=np.int32))  # (B,)
            yield batch_frames, batch_labels


# Convenience alias
create_dataloader = EventDataloader
