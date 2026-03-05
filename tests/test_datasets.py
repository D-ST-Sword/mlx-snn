"""Tests for mlxsnn.datasets — neuromorphic dataset loaders.

These tests exercise the dataset loaders and utility functions against
the actual data stored at /Volumes/ws01/myPaperCode/mambaspike/data/.
Tests that require the external data paths are marked with
``@pytest.mark.skipif`` so that CI can skip them gracefully.
"""

from __future__ import annotations

import math
import os

import mlx.core as mx
import numpy as np
import pytest

from mlxsnn.datasets.utils import events_to_frames, resize_frames, EventDataloader

# ---------------------------------------------------------------------------
# Data root paths
# ---------------------------------------------------------------------------
DVS_GESTURE_ROOT = (
    "/Volumes/ws01/myPaperCode/mambaspike/data/dvsgesture/DVSGesture"
)
CIFAR10DVS_ROOT = (
    "/Volumes/ws01/myPaperCode/mambaspike/data/cifar10dvs/CIFAR10DVS"
)
NMNIST_ROOT = "/Volumes/ws01/myPaperCode/mambaspike/data/nmnist/NMNIST"

_has_dvs = os.path.isdir(DVS_GESTURE_ROOT)
_has_cifar = os.path.isdir(CIFAR10DVS_ROOT)
_has_nmnist = os.path.isdir(NMNIST_ROOT)

try:
    import aedat as _aedat  # noqa: F401
    _has_aedat = True
except ImportError:
    _has_aedat = False


# ===================================================================
# Unit tests for utility functions (no external data needed)
# ===================================================================

class TestEventsToFrames:
    """Test the generic events_to_frames utility."""

    def test_basic_shape(self):
        """Output has correct (T, H, W, P) shape."""
        events = np.array(
            [[5, 10, 0, 100], [6, 11, 1, 200], [7, 12, 0, 300]],
            dtype=np.float64,
        )
        frames = events_to_frames(events, num_steps=4, sensor_size=(34, 34))
        assert frames.shape == (4, 34, 34, 2)
        assert frames.dtype == np.float32

    def test_empty_events(self):
        """Empty events produce all-zero frames."""
        events = np.zeros((0, 4), dtype=np.float64)
        frames = events_to_frames(events, num_steps=5, sensor_size=(10, 10))
        assert frames.shape == (5, 10, 10, 2)
        assert frames.sum() == 0.0

    def test_single_event(self):
        """A single event lands in the first bin."""
        events = np.array([[3, 7, 1, 42.0]])
        frames = events_to_frames(events, num_steps=3, sensor_size=(10, 10))
        assert frames.shape == (3, 10, 10, 2)
        # Single timestamp => all events go to bin 0
        assert frames[0, 7, 3, 1] == 1.0
        assert frames.sum() == 1.0

    def test_event_accumulation(self):
        """Multiple events at the same pixel accumulate counts."""
        events = np.array(
            [[5, 5, 0, 0.0], [5, 5, 0, 0.0], [5, 5, 0, 0.0]],
            dtype=np.float64,
        )
        frames = events_to_frames(events, num_steps=1, sensor_size=(10, 10))
        assert frames[0, 5, 5, 0] == 3.0

    def test_polarity_separation(self):
        """ON and OFF events go to separate polarity channels."""
        events = np.array(
            [[5, 5, 0, 10.0], [5, 5, 1, 10.0]], dtype=np.float64
        )
        frames = events_to_frames(events, num_steps=1, sensor_size=(10, 10))
        assert frames[0, 5, 5, 0] == 1.0
        assert frames[0, 5, 5, 1] == 1.0

    def test_time_binning(self):
        """Events are distributed across correct temporal bins."""
        # 4 events at equally spaced timestamps => one per bin
        events = np.array(
            [
                [0, 0, 0, 0.0],
                [0, 0, 0, 1.0],
                [0, 0, 0, 2.0],
                [0, 0, 0, 3.0],
            ],
            dtype=np.float64,
        )
        frames = events_to_frames(events, num_steps=4, sensor_size=(5, 5))
        # Each bin should have exactly 1 event at (0, 0, pol=0)
        for t in range(4):
            assert frames[t, 0, 0, 0] == 1.0
        assert frames.sum() == 4.0

    def test_structured_array_input(self):
        """Structured arrays (from aedat) are handled correctly."""
        dtype = np.dtype(
            [("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "?")]
        )
        events = np.array([(100, 5, 10, True), (200, 6, 11, False)], dtype=dtype)
        frames = events_to_frames(events, num_steps=2, sensor_size=(34, 34))
        assert frames.shape == (2, 34, 34, 2)
        assert frames.sum() == 2.0

    def test_variable_num_steps(self):
        """Different num_steps values produce correct shapes."""
        events = np.random.rand(100, 4)
        events[:, 0] = np.random.randint(0, 34, 100)
        events[:, 1] = np.random.randint(0, 34, 100)
        events[:, 2] = np.random.randint(0, 2, 100)
        events[:, 3] = np.sort(np.random.rand(100))

        for T in [1, 5, 10, 50]:
            frames = events_to_frames(events, num_steps=T, sensor_size=(34, 34))
            assert frames.shape == (T, 34, 34, 2)


class TestResizeFrames:
    """Test spatial resizing of frame tensors."""

    def test_identity(self):
        """No resize when target matches source."""
        frames = np.random.rand(4, 128, 128, 2).astype(np.float32)
        out = resize_frames(frames, (128, 128))
        assert np.array_equal(out, frames)

    def test_downsample_exact_divisor(self):
        """Downsampling by exact divisor uses block averaging."""
        frames = np.ones((2, 128, 128, 2), dtype=np.float32)
        out = resize_frames(frames, (64, 64))
        assert out.shape == (2, 64, 64, 2)
        # Each output pixel is the mean of a 2x2 block of 1s => still 1
        assert np.allclose(out, 1.0)

    def test_arbitrary_resize(self):
        """Arbitrary resize produces correct shape."""
        frames = np.random.rand(3, 128, 128, 2).astype(np.float32)
        out = resize_frames(frames, (48, 48))
        assert out.shape == (3, 48, 48, 2)


# ===================================================================
# Integration tests: DVS Gesture
# ===================================================================

@pytest.mark.skipif(not _has_dvs, reason="DVSGesture data not available")
class TestDVSGestureDataset:
    """Integration tests for DVSGestureDataset with real data."""

    def test_load_train(self):
        from mlxsnn.datasets import DVSGestureDataset

        ds = DVSGestureDataset(DVS_GESTURE_ROOT, train=True, num_steps=16)
        assert len(ds) > 0
        assert ds.num_classes == 11

    def test_load_test(self):
        from mlxsnn.datasets import DVSGestureDataset

        ds = DVSGestureDataset(DVS_GESTURE_ROOT, train=False, num_steps=16)
        assert len(ds) > 0

    def test_sample_shape(self):
        from mlxsnn.datasets import DVSGestureDataset

        ds = DVSGestureDataset(DVS_GESTURE_ROOT, train=True, num_steps=16)
        frames, label = ds[0]
        assert isinstance(frames, mx.array)
        assert frames.shape == (16, 128, 128, 2)
        assert frames.dtype == mx.float32
        assert 0 <= label <= 10

    def test_custom_spatial_size(self):
        from mlxsnn.datasets import DVSGestureDataset

        ds = DVSGestureDataset(
            DVS_GESTURE_ROOT, train=True, num_steps=8, spatial_size=(64, 64)
        )
        frames, label = ds[0]
        assert frames.shape == (8, 64, 64, 2)

    def test_variable_num_steps(self):
        from mlxsnn.datasets import DVSGestureDataset

        for T in [4, 16, 32]:
            ds = DVSGestureDataset(DVS_GESTURE_ROOT, train=True, num_steps=T)
            frames, _ = ds[0]
            assert frames.shape[0] == T

    def test_repr(self):
        from mlxsnn.datasets import DVSGestureDataset

        ds = DVSGestureDataset(DVS_GESTURE_ROOT, train=True, num_steps=16)
        r = repr(ds)
        assert "DVSGestureDataset" in r
        assert "train" in r

    def test_label_range(self):
        from mlxsnn.datasets import DVSGestureDataset

        ds = DVSGestureDataset(DVS_GESTURE_ROOT, train=True, num_steps=8)
        labels = set()
        # Check first 50 samples for speed
        for i in range(min(50, len(ds))):
            _, label = ds[i]
            labels.add(label)
        assert all(0 <= l <= 10 for l in labels)


# ===================================================================
# Integration tests: CIFAR10-DVS
# ===================================================================

@pytest.mark.skipif(
    not (_has_cifar and _has_aedat),
    reason="CIFAR10-DVS data or aedat package not available",
)
class TestCIFAR10DVSDataset:
    """Integration tests for CIFAR10DVSDataset with real data."""

    def test_load_train(self):
        from mlxsnn.datasets import CIFAR10DVSDataset

        ds = CIFAR10DVSDataset(CIFAR10DVS_ROOT, train=True, num_steps=10)
        assert len(ds) > 0
        assert ds.num_classes == 10

    def test_load_test(self):
        from mlxsnn.datasets import CIFAR10DVSDataset

        ds = CIFAR10DVSDataset(CIFAR10DVS_ROOT, train=False, num_steps=10)
        assert len(ds) > 0

    def test_train_test_disjoint(self):
        from mlxsnn.datasets import CIFAR10DVSDataset

        train_ds = CIFAR10DVSDataset(CIFAR10DVS_ROOT, train=True, num_steps=4)
        test_ds = CIFAR10DVSDataset(CIFAR10DVS_ROOT, train=False, num_steps=4)
        train_paths = {p for p, _ in train_ds.samples}
        test_paths = {p for p, _ in test_ds.samples}
        assert len(train_paths & test_paths) == 0
        assert len(train_ds) + len(test_ds) == 10000

    def test_sample_shape(self):
        from mlxsnn.datasets import CIFAR10DVSDataset

        ds = CIFAR10DVSDataset(CIFAR10DVS_ROOT, train=True, num_steps=10)
        frames, label = ds[0]
        assert isinstance(frames, mx.array)
        assert frames.shape == (10, 128, 128, 2)
        assert frames.dtype == mx.float32
        assert 0 <= label <= 9

    def test_repr(self):
        from mlxsnn.datasets import CIFAR10DVSDataset

        ds = CIFAR10DVSDataset(CIFAR10DVS_ROOT, train=True, num_steps=10)
        r = repr(ds)
        assert "CIFAR10DVSDataset" in r


# ===================================================================
# Integration tests: N-MNIST
# ===================================================================

@pytest.mark.skipif(not _has_nmnist, reason="N-MNIST data not available")
class TestNMNISTDataset:
    """Integration tests for NMNISTDataset with real data."""

    def test_load_train(self):
        from mlxsnn.datasets import NMNISTDataset

        ds = NMNISTDataset(NMNIST_ROOT, train=True, num_steps=20)
        assert len(ds) == 60000
        assert ds.num_classes == 10

    def test_load_test(self):
        from mlxsnn.datasets import NMNISTDataset

        ds = NMNISTDataset(NMNIST_ROOT, train=False, num_steps=20)
        assert len(ds) == 10000

    def test_sample_shape(self):
        from mlxsnn.datasets import NMNISTDataset

        ds = NMNISTDataset(NMNIST_ROOT, train=True, num_steps=20)
        frames, label = ds[0]
        assert isinstance(frames, mx.array)
        assert frames.shape == (20, 34, 34, 2)
        assert frames.dtype == mx.float32
        assert 0 <= label <= 9

    def test_custom_spatial_size(self):
        from mlxsnn.datasets import NMNISTDataset

        ds = NMNISTDataset(
            NMNIST_ROOT, train=True, num_steps=10, spatial_size=(17, 17)
        )
        frames, _ = ds[0]
        assert frames.shape == (10, 17, 17, 2)

    def test_variable_num_steps(self):
        from mlxsnn.datasets import NMNISTDataset

        for T in [5, 20, 50]:
            ds = NMNISTDataset(NMNIST_ROOT, train=True, num_steps=T)
            frames, _ = ds[0]
            assert frames.shape[0] == T

    def test_bin_parser_consistency(self):
        """Parsed events have valid coordinate ranges."""
        from mlxsnn.datasets.nmnist import NMNISTDataset

        path = os.path.join(NMNIST_ROOT, "Train", "0")
        bin_files = sorted(f for f in os.listdir(path) if f.endswith(".bin"))
        events = NMNISTDataset._load_bin(os.path.join(path, bin_files[0]))
        assert events.shape[1] == 4
        assert events[:, 0].min() >= 0 and events[:, 0].max() <= 33
        assert events[:, 1].min() >= 0 and events[:, 1].max() <= 33
        assert set(np.unique(events[:, 2]).astype(int)).issubset({0, 1})
        assert events[:, 3].min() >= 0

    def test_repr(self):
        from mlxsnn.datasets import NMNISTDataset

        ds = NMNISTDataset(NMNIST_ROOT, train=True, num_steps=20)
        r = repr(ds)
        assert "NMNISTDataset" in r
        assert "train" in r


# ===================================================================
# Dataloader tests
# ===================================================================

class _DummyDataset:
    """Minimal dataset for testing the dataloader without real data."""

    def __init__(self, n: int = 20, T: int = 4, H: int = 8, W: int = 8):
        self.n = n
        self.T = T
        self.H = H
        self.W = W

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        frames = mx.zeros((self.T, self.H, self.W, 2))
        label = idx % 10
        return frames, label


class TestEventDataloader:
    """Test the EventDataloader batch iterator."""

    def test_batch_shape(self):
        ds = _DummyDataset(n=20, T=4, H=8, W=8)
        loader = EventDataloader(ds, batch_size=5, shuffle=False)
        frames, labels = next(iter(loader))
        assert frames.shape == (5, 4, 8, 8, 2)
        assert labels.shape == (5,)

    def test_num_batches(self):
        ds = _DummyDataset(n=20)
        loader = EventDataloader(ds, batch_size=7, shuffle=False)
        assert len(loader) == 3  # ceil(20 / 7)

    def test_drop_last(self):
        ds = _DummyDataset(n=20)
        loader = EventDataloader(ds, batch_size=7, drop_last=True)
        assert len(loader) == 2  # floor(20 / 7)
        batches = list(loader)
        assert len(batches) == 2
        for frames, labels in batches:
            assert frames.shape[0] == 7

    def test_last_batch_smaller(self):
        ds = _DummyDataset(n=10)
        loader = EventDataloader(ds, batch_size=7, shuffle=False, drop_last=False)
        batches = list(loader)
        assert len(batches) == 2
        assert batches[0][0].shape[0] == 7
        assert batches[1][0].shape[0] == 3

    def test_shuffle_deterministic(self):
        """Same seed produces same order."""
        ds = _DummyDataset(n=20)
        loader1 = EventDataloader(ds, batch_size=5, shuffle=True, seed=123)
        loader2 = EventDataloader(ds, batch_size=5, shuffle=True, seed=123)
        for (f1, l1), (f2, l2) in zip(loader1, loader2):
            assert mx.array_equal(l1, l2)

    def test_labels_dtype(self):
        ds = _DummyDataset(n=5)
        loader = EventDataloader(ds, batch_size=5, shuffle=False)
        _, labels = next(iter(loader))
        assert labels.dtype == mx.int32

    def test_full_epoch_coverage(self):
        """All samples are visited in one full epoch (no shuffle)."""
        ds = _DummyDataset(n=13)
        loader = EventDataloader(ds, batch_size=4, shuffle=False, drop_last=False)
        all_labels = []
        for _, labels in loader:
            all_labels.extend(np.array(labels).tolist())
        assert len(all_labels) == 13


# ===================================================================
# Dataloader with real datasets
# ===================================================================

@pytest.mark.skipif(not _has_nmnist, reason="N-MNIST data not available")
class TestDataloaderWithNMNIST:
    """Test EventDataloader with real NMNISTDataset."""

    def test_batch_loading(self):
        from mlxsnn.datasets import NMNISTDataset

        ds = NMNISTDataset(NMNIST_ROOT, train=True, num_steps=10)
        loader = EventDataloader(ds, batch_size=4, shuffle=False)
        frames, labels = next(iter(loader))
        assert frames.shape == (4, 10, 34, 34, 2)
        assert labels.shape == (4,)
        assert frames.dtype == mx.float32
        assert labels.dtype == mx.int32


@pytest.mark.skipif(not _has_dvs, reason="DVSGesture data not available")
class TestDataloaderWithDVSGesture:
    """Test EventDataloader with real DVSGestureDataset."""

    def test_batch_loading(self):
        from mlxsnn.datasets import DVSGestureDataset

        ds = DVSGestureDataset(DVS_GESTURE_ROOT, train=True, num_steps=8)
        loader = EventDataloader(ds, batch_size=2, shuffle=False)
        frames, labels = next(iter(loader))
        assert frames.shape == (2, 8, 128, 128, 2)
        assert labels.shape == (2,)
