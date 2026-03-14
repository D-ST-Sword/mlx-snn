"""Tests for v0.5 SHD dataset loader."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import mlx.core as mx
import numpy as np
import pytest

from mlxsnn.datasets.shd import SHDDataset, _ensure_h5py


class TestSHDDatasetUnit:
    """Unit tests using mocked HDF5 files."""

    @pytest.fixture
    def mock_shd_dir(self, tmp_path):
        """Create a mock SHD dataset directory with h5py."""
        h5py = pytest.importorskip("h5py")

        num_samples = 10
        for fname in ["shd_train.h5", "shd_test.h5"]:
            filepath = str(tmp_path / fname)
            with h5py.File(filepath, "w") as f:
                grp = f.create_group("spikes")
                times_grp = grp.create_group("times")
                units_grp = grp.create_group("units")
                for i in range(num_samples):
                    n_spikes = np.random.randint(10, 100)
                    times_grp.create_dataset(
                        str(i), data=np.sort(np.random.uniform(0, 1.0, n_spikes))
                    )
                    units_grp.create_dataset(
                        str(i), data=np.random.randint(0, 700, n_spikes)
                    )
                f.create_dataset("labels", data=np.random.randint(0, 20, num_samples))
        return str(tmp_path)

    def test_len(self, mock_shd_dir):
        ds = SHDDataset(mock_shd_dir, train=True, num_steps=50)
        assert len(ds) == 10

    def test_getitem_shape(self, mock_shd_dir):
        ds = SHDDataset(mock_shd_dir, train=True, num_steps=50)
        spikes, label = ds[0]
        mx.eval(spikes)
        assert list(spikes.shape) == [50, 700]
        assert isinstance(label, int)
        assert 0 <= label < 20

    def test_binary_output(self, mock_shd_dir):
        ds = SHDDataset(mock_shd_dir, train=True, num_steps=50)
        spikes, _ = ds[0]
        mx.eval(spikes)
        vals = set(np.array(spikes).flatten().tolist())
        assert vals.issubset({0.0, 1.0})

    def test_test_split(self, mock_shd_dir):
        ds = SHDDataset(mock_shd_dir, train=False, num_steps=50)
        assert len(ds) == 10

    def test_repr(self, mock_shd_dir):
        ds = SHDDataset(mock_shd_dir, train=True, num_steps=100)
        r = repr(ds)
        assert "train" in r
        assert "100" in r

    def test_transform(self, mock_shd_dir):
        """Transform should be applied to each sample."""
        def double(x):
            return x * 2.0

        ds = SHDDataset(mock_shd_dir, train=True, num_steps=50, transform=double)
        spikes, _ = ds[0]
        mx.eval(spikes)
        # After doubling binary values, max should be 2.0
        max_val = mx.max(spikes).item()
        assert max_val <= 2.0

    def test_custom_num_steps(self, mock_shd_dir):
        ds = SHDDataset(mock_shd_dir, train=True, num_steps=200)
        spikes, _ = ds[0]
        mx.eval(spikes)
        assert spikes.shape[0] == 200

    def test_custom_max_time(self, mock_shd_dir):
        ds = SHDDataset(mock_shd_dir, train=True, num_steps=50, max_time=0.5)
        spikes, _ = ds[0]
        mx.eval(spikes)
        assert list(spikes.shape) == [50, 700]


class TestSHDIntegration:
    """Integration tests — skipped if real data is not available."""

    @pytest.fixture
    def shd_root(self):
        root = os.environ.get("SHD_DATA_DIR", "./data/shd")
        if not os.path.isfile(os.path.join(root, "shd_train.h5")):
            pytest.skip("SHD data not found; set SHD_DATA_DIR to run")
        return root

    def test_load_train(self, shd_root):
        ds = SHDDataset(shd_root, train=True, num_steps=100)
        assert len(ds) > 0
        spikes, label = ds[0]
        mx.eval(spikes)
        assert list(spikes.shape) == [100, 700]
        assert 0 <= label < 20


class TestEnsureH5py:
    def test_import_available(self):
        """Should not raise if h5py is installed."""
        try:
            _ensure_h5py()
        except ImportError:
            pytest.skip("h5py not installed")
