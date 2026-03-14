"""Tests for the Liquid State Machine module."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlxsnn.liquid import LiquidReservoir, LSM
from mlxsnn.liquid.topology import erdos_renyi, small_world, scale_free


# ---------------------------------------------------------------------------
# Topology tests
# ---------------------------------------------------------------------------

class TestTopology:
    def test_erdos_renyi_shape(self):
        W = erdos_renyi(100, p=0.1)
        assert W.shape == (100, 100)
        assert W.dtype == np.float32

    def test_erdos_renyi_no_self_connections(self):
        W = erdos_renyi(50, p=0.5)
        assert np.all(np.diag(W) == 0)

    def test_erdos_renyi_dales_law(self):
        n = 100
        W = erdos_renyi(n, p=0.2, exc_ratio=0.8)
        n_exc = int(0.8 * n)
        # Excitatory columns should be non-negative
        assert np.all(W[:, :n_exc] >= 0)
        # Inhibitory columns should be non-positive
        assert np.all(W[:, n_exc:] <= 0)

    def test_erdos_renyi_spectral_radius(self):
        W = erdos_renyi(50, p=0.2, spectral_radius=0.9)
        eigvals = np.linalg.eigvals(W)
        radius = np.max(np.abs(eigvals))
        assert abs(radius - 0.9) < 0.05

    def test_erdos_renyi_reproducible(self):
        W1 = erdos_renyi(50, rng=np.random.default_rng(42))
        W2 = erdos_renyi(50, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(W1, W2)

    def test_small_world_shape(self):
        W = small_world(50, k=4, p_rewire=0.1)
        assert W.shape == (50, 50)

    def test_small_world_dales_law(self):
        n = 50
        W = small_world(n, k=4, exc_ratio=0.8)
        n_exc = int(0.8 * n)
        assert np.all(W[:, :n_exc] >= 0)
        assert np.all(W[:, n_exc:] <= 0)

    def test_scale_free_shape(self):
        W = scale_free(50, m=3)
        assert W.shape == (50, 50)

    def test_scale_free_dales_law(self):
        n = 50
        W = scale_free(n, m=3, exc_ratio=0.8)
        n_exc = int(0.8 * n)
        assert np.all(W[:, :n_exc] >= 0)
        assert np.all(W[:, n_exc:] <= 0)


# ---------------------------------------------------------------------------
# LiquidReservoir tests
# ---------------------------------------------------------------------------

class TestLiquidReservoir:
    def test_init_state_shape(self):
        res = LiquidReservoir(input_size=32, reservoir_size=100)
        state = res.init_state(batch_size=4)
        assert state["mem"].shape == (4, 100)
        assert state["spk"].shape == (4, 100)

    def test_forward_shape(self):
        res = LiquidReservoir(input_size=16, reservoir_size=50)
        state = res.init_state(4)
        x = mx.random.normal((4, 16))
        spk, new_state = res(x, state)
        mx.eval(spk, new_state["mem"], new_state["spk"])
        assert spk.shape == (4, 50)
        assert new_state["mem"].shape == (4, 50)

    def test_spikes_are_binary(self):
        res = LiquidReservoir(input_size=16, reservoir_size=50)
        state = res.init_state(4)
        x = mx.random.normal((4, 16)) * 3.0  # strong input
        spk, _ = res(x, state)
        mx.eval(spk)
        vals = set(spk.tolist()[0])  # flatten first batch
        assert vals.issubset({0.0, 1.0})

    def test_topology_erdos_renyi(self):
        res = LiquidReservoir(input_size=8, reservoir_size=30, topology="erdos_renyi")
        assert res.W_res.shape == (30, 30)

    def test_topology_small_world(self):
        res = LiquidReservoir(input_size=8, reservoir_size=30, topology="small_world")
        assert res.W_res.shape == (30, 30)

    def test_topology_scale_free(self):
        res = LiquidReservoir(input_size=8, reservoir_size=30, topology="scale_free")
        assert res.W_res.shape == (30, 30)

    def test_unknown_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            LiquidReservoir(input_size=8, reservoir_size=30, topology="unknown")

    def test_recurrent_dynamics(self):
        """Verify that reservoir state evolves over multiple steps."""
        res = LiquidReservoir(input_size=8, reservoir_size=30)
        state = res.init_state(2)
        x = mx.random.normal((2, 8))

        states = []
        for _ in range(5):
            _, state = res(x, state)
            mx.eval(state["mem"])
            states.append(state["mem"].tolist())

        # Membrane potentials should change over time
        assert states[0] != states[4]


# ---------------------------------------------------------------------------
# LSM tests
# ---------------------------------------------------------------------------

class TestLSM:
    def test_init_state(self):
        lsm = LSM(input_size=16, reservoir_size=50, output_size=5)
        state = lsm.init_state(4)
        assert "mem" in state
        assert "spk" in state

    def test_forward_shape(self):
        lsm = LSM(input_size=16, reservoir_size=50, output_size=5)
        state = lsm.init_state(4)
        x = mx.random.normal((4, 16))
        out, state = lsm(x, state)
        mx.eval(out)
        assert out.shape == (4, 5)

    def test_forward_sequence(self):
        lsm = LSM(input_size=16, reservoir_size=50, output_size=5)
        x_seq = mx.random.normal((10, 4, 16))
        outputs, state = lsm.forward_sequence(x_seq)
        mx.eval(outputs)
        assert outputs.shape == (10, 4, 5)

    def test_forward_sequence_with_state(self):
        lsm = LSM(input_size=8, reservoir_size=30, output_size=3)
        state = lsm.init_state(2)
        x_seq = mx.random.normal((5, 2, 8))
        outputs, final_state = lsm.forward_sequence(x_seq, state)
        mx.eval(outputs)
        assert outputs.shape == (5, 2, 3)

    def test_readout_linear(self):
        lsm = LSM(input_size=8, reservoir_size=30, output_size=3, readout="linear")
        assert isinstance(lsm.readout, nn.Linear)

    def test_readout_mlp(self):
        lsm = LSM(input_size=8, reservoir_size=30, output_size=3, readout="mlp")
        assert isinstance(lsm.readout, nn.Sequential)

    def test_unknown_readout_raises(self):
        with pytest.raises(ValueError, match="Unknown readout"):
            LSM(input_size=8, reservoir_size=30, readout="svm")

    def test_reservoir_frozen_by_default(self):
        lsm = LSM(input_size=8, reservoir_size=30, output_size=3)
        # Reservoir weights should not be trainable
        from mlx.utils import tree_flatten
        trainable = tree_flatten(lsm.trainable_parameters())
        param_names = [name for name, _ in trainable]
        # Should have readout params but not reservoir W_res/W_in
        assert any("readout" in n for n in param_names)
        assert not any("W_res" in n for n in param_names)
        assert not any("W_in" in n for n in param_names)

    def test_train_reservoir(self):
        lsm = LSM(input_size=8, reservoir_size=30, output_size=3,
                   train_reservoir=True)
        from mlx.utils import tree_flatten
        trainable = tree_flatten(lsm.trainable_parameters())
        param_names = [name for name, _ in trainable]
        assert any("W_res" in n for n in param_names)

    def test_gradient_flow_through_readout(self):
        """Verify gradients flow through the readout layer."""
        lsm = LSM(input_size=8, reservoir_size=30, output_size=3)
        x_seq = mx.random.normal((5, 4, 8))

        def loss_fn(model, x):
            out, _ = model.forward_sequence(x)
            return mx.mean(out)

        loss, grads = nn.value_and_grad(lsm, loss_fn)(lsm, x_seq)
        mx.eval(loss, grads)

        from mlx.utils import tree_flatten
        grad_flat = tree_flatten(grads)
        readout_grads = [(n, g) for n, g in grad_flat if "readout" in n]
        assert len(readout_grads) > 0
        assert any(mx.any(mx.abs(g) > 0).item() for _, g in readout_grads)


# ---------------------------------------------------------------------------
# Integration: top-level import
# ---------------------------------------------------------------------------

class TestTopLevelImport:
    def test_import_from_mlxsnn(self):
        import mlxsnn
        assert hasattr(mlxsnn, "LiquidReservoir")
        assert hasattr(mlxsnn, "LSM")
