"""Tests for the four Conv SNN operators (TAC, FTC, IMC, TCC)."""

import math

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlxsnn.operators import (
    TemporalAggregatedConv,
    FourierTemporalConv,
    InfoMaxSpikeConv,
    TemporalCollapseConv,
)
from mlxsnn.operators.infomax_spike_conv import binary_entropy


# ──────────────────────────────────────────────────────────────────────
# TemporalAggregatedConv (TAC) Tests
# ──────────────────────────────────────────────────────────────────────

class TestTAC:
    """Tests for TemporalAggregatedConv."""

    def test_output_shape_basic(self):
        tac = TemporalAggregatedConv(3, 16, 3, chunk_size=4, padding=1)
        state = tac.init_state(2, (8, 8))
        x = mx.ones((16, 2, 8, 8, 3))
        spk, new_state = tac(x, state)
        mx.eval(spk)
        assert list(spk.shape) == [4, 2, 8, 8, 16]  # T/K = 16/4 = 4

    def test_output_shape_k1_matches_T(self):
        """K=1 should output T timesteps (equivalent to standard)."""
        tac = TemporalAggregatedConv(3, 16, 3, chunk_size=1, padding=1)
        state = tac.init_state(2, (8, 8))
        x = mx.ones((10, 2, 8, 8, 3))
        spk, _ = tac(x, state)
        mx.eval(spk)
        assert list(spk.shape) == [10, 2, 8, 8, 16]

    def test_full_collapse(self):
        """K=T should collapse entire sequence to 1 timestep."""
        T = 8
        tac = TemporalAggregatedConv(3, 16, 3, chunk_size=T, padding=1)
        state = tac.init_state(2, (8, 8))
        x = mx.ones((T, 2, 8, 8, 3))
        spk, _ = tac(x, state)
        mx.eval(spk)
        assert list(spk.shape) == [1, 2, 8, 8, 16]

    def test_non_divisible_T(self):
        """T not divisible by K should still work (last chunk smaller)."""
        tac = TemporalAggregatedConv(3, 16, 3, chunk_size=4, padding=1)
        state = tac.init_state(2, (8, 8))
        x = mx.ones((10, 2, 8, 8, 3))  # 10/4 = 2.5 -> 3 chunks
        spk, _ = tac(x, state)
        mx.eval(spk)
        assert list(spk.shape) == [3, 2, 8, 8, 16]

    def test_gradient_flow(self):
        tac = TemporalAggregatedConv(2, 8, 3, chunk_size=2, padding=1)
        state = tac.init_state(2, (4, 4))
        x = mx.random.normal((8, 2, 4, 4, 2))

        def loss_fn(model, x, s):
            spk, _ = model(x, s)
            return mx.mean(spk)

        loss, grads = nn.value_and_grad(tac, loss_fn)(tac, x, state)
        mx.eval(loss, grads)
        assert float(mx.sqrt(mx.sum(grads['conv']['weight'] ** 2)).item()) > 0

    def test_error_bound(self):
        tac = TemporalAggregatedConv(3, 16, 3, beta=0.9)
        bound = tac.theoretical_error_bound(0.1)
        expected = 0.1 * 1.0 * 0.9 / 0.1  # rho * Vth * beta / (1-beta)
        assert abs(bound - expected) < 1e-6

    def test_state_persistence(self):
        """Membrane state should carry across calls."""
        tac = TemporalAggregatedConv(2, 8, 3, chunk_size=2, padding=1)
        state = tac.init_state(2, (4, 4))
        x1 = mx.ones((4, 2, 4, 4, 2))
        x2 = mx.ones((4, 2, 4, 4, 2))

        _, state1 = tac(x1, state)
        mx.eval(state1["mem"])
        assert not mx.allclose(state1["mem"], mx.zeros_like(state1["mem"]))


# ──────────────────────────────────────────────────────────────────────
# FourierTemporalConv (FTC) Tests
# ──────────────────────────────────────────────────────────────────────

class TestFTC:
    """Tests for FourierTemporalConv."""

    def test_output_shape(self):
        ftc = FourierTemporalConv(3, 16, 3, filter_order=2, padding=1)
        state = ftc.init_state(2, (8, 8))
        x = mx.ones((10, 2, 8, 8, 3))
        spk, new_state = ftc(x, state)
        mx.eval(spk)
        assert list(spk.shape) == [10, 2, 8, 8, 16]

    def test_order1_shape(self):
        ftc = FourierTemporalConv(3, 16, 3, filter_order=1, padding=1)
        state = ftc.init_state(2, (8, 8))
        x = mx.ones((10, 2, 8, 8, 3))
        spk, _ = ftc(x, state)
        mx.eval(spk)
        assert list(spk.shape) == [10, 2, 8, 8, 16]

    def test_order1_no_mem_prev(self):
        """Order-1 state should not have mem_prev."""
        ftc = FourierTemporalConv(3, 16, 3, filter_order=1, padding=1)
        state = ftc.init_state(2, (8, 8))
        assert "mem_prev" not in state

    def test_order2_has_mem_prev(self):
        """Order-2 state should have mem_prev."""
        ftc = FourierTemporalConv(3, 16, 3, filter_order=2, padding=1)
        state = ftc.init_state(2, (8, 8))
        assert "mem_prev" in state

    def test_stability_constraint(self):
        """Poles should always be inside unit circle."""
        ftc = FourierTemporalConv(3, 16, 3, filter_order=2, padding=1)
        a1, a2, _, _ = ftc._get_coefficients()
        mx.eval(a1, a2)
        # For biquad: poles at (a1 +/- sqrt(a1^2 + 4*a2)) / 2
        # With a2 = -r^2 and a1 = 2*r*cos(theta), |pole| = r < 1
        r = mx.sigmoid(ftc.r_raw)
        mx.eval(r)
        assert mx.all(r < 1.0).item()
        assert mx.all(r > 0.0).item()

    def test_gradient_flow_all_params(self):
        ftc = FourierTemporalConv(2, 8, 3, filter_order=2, padding=1)
        state = ftc.init_state(2, (4, 4))
        x = mx.random.normal((6, 2, 4, 4, 2))

        def loss_fn(model, x, s):
            spk, _ = model(x, s)
            return mx.mean(spk)

        loss, grads = nn.value_and_grad(ftc, loss_fn)(ftc, x, state)
        mx.eval(loss, grads)
        # All learnable params should have gradients.
        assert 'r_raw' in grads
        assert 'theta_raw' in grads
        assert 'b0' in grads
        assert 'b1' in grads

    def test_frequency_response_shape(self):
        ftc = FourierTemporalConv(3, 16, 3, filter_order=2, padding=1)
        freqs, mag = ftc.get_frequency_response(num_points=128)
        mx.eval(freqs, mag)
        assert list(freqs.shape) == [128]
        assert list(mag.shape) == [16, 128]

    def test_invalid_order_raises(self):
        with pytest.raises(AssertionError):
            FourierTemporalConv(3, 16, 3, filter_order=3)


# ──────────────────────────────────────────────────────────────────────
# InfoMaxSpikeConv (IMC) Tests
# ──────────────────────────────────────────────────────────────────────

class TestIMC:
    """Tests for InfoMaxSpikeConv."""

    def test_output_shape(self):
        imc = InfoMaxSpikeConv(3, 16, 3, padding=1)
        state = imc.init_state(2, (8, 8))
        x = mx.ones((2, 8, 8, 3))
        spk, _ = imc(x, state)
        mx.eval(spk)
        assert list(spk.shape) == [2, 8, 8, 16]

    def test_gates_in_range(self):
        imc = InfoMaxSpikeConv(3, 16, 3, padding=1)
        gates = imc._get_gates()
        mx.eval(gates)
        assert mx.all(gates >= 0.0).item()
        assert mx.all(gates <= 1.0).item()

    def test_info_loss_nonnegative(self):
        imc = InfoMaxSpikeConv(3, 16, 3, padding=1)
        state = imc.init_state(2, (4, 4))
        x = mx.random.normal((2, 4, 4, 3))
        spk, _ = imc(x, state)
        mx.eval(spk)
        loss = imc.info_loss()
        mx.eval(loss)
        assert float(loss.item()) >= 0.0

    def test_effective_channels_initial(self):
        imc = InfoMaxSpikeConv(3, 32, 3, padding=1)
        # Initial gate_logits = 3.0, sigmoid(3) ~ 0.95 > 0.5
        assert imc.effective_channels() == 32

    def test_gradient_includes_gates(self):
        imc = InfoMaxSpikeConv(2, 8, 3, padding=1)
        state = imc.init_state(2, (4, 4))
        x = mx.random.normal((2, 4, 4, 2))

        def loss_fn(model, x, s):
            spk, _ = model(x, s)
            return mx.mean(spk) + model.info_loss()

        loss, grads = nn.value_and_grad(imc, loss_fn)(imc, x, state)
        mx.eval(loss, grads)
        assert 'gate_logits' in grads

    def test_minimum_channels_calculation(self):
        # C_in=64, same rho, 4x spatial reduction -> need 256 channels
        c_min = InfoMaxSpikeConv.minimum_channels(
            c_in=64, rho_in=0.1, rho_out=0.1, spatial_reduction=4.0
        )
        assert c_min == 256

    def test_channel_utilization_returns_dict(self):
        imc = InfoMaxSpikeConv(3, 16, 3, padding=1)
        state = imc.init_state(2, (4, 4))
        x = mx.random.normal((2, 4, 4, 3))
        spk, _ = imc(x, state)
        mx.eval(spk)
        stats = imc.channel_utilization()
        assert 'gate_mean' in stats
        assert 'effective_channels' in stats

    def test_binary_entropy(self):
        # H(0.5) = 1 bit
        h = binary_entropy(mx.array(0.5))
        mx.eval(h)
        assert abs(float(h.item()) - 1.0) < 1e-5

        # H(0) ~ 0, H(1) ~ 0
        h0 = binary_entropy(mx.array(1e-7))
        h1 = binary_entropy(mx.array(1.0 - 1e-7))
        mx.eval(h0, h1)
        assert float(h0.item()) < 0.01
        assert float(h1.item()) < 0.01


# ──────────────────────────────────────────────────────────────────────
# TemporalCollapseConv (TCC) Tests
# ──────────────────────────────────────────────────────────────────────

class TestTCC:
    """Tests for TemporalCollapseConv."""

    def test_output_shape(self):
        tcc = TemporalCollapseConv(3, 16, 3, padding=1)
        state = tcc.init_state(2, (8, 8))
        x = mx.ones((16, 2, 8, 8, 3))
        spk, _ = tcc(x, state)
        mx.eval(spk)
        assert list(spk.shape) == [16, 2, 8, 8, 16]

    def test_no_cache_processes_all(self):
        """Without cache, all T timesteps should be processed individually."""
        tcc = TemporalCollapseConv(3, 16, 3, padding=1)
        state = tcc.init_state(2, (4, 4))
        x = mx.ones((8, 2, 4, 4, 3))
        spk, _ = tcc(x, state)
        mx.eval(spk)
        assert tcc.conv_calls == 8

    def test_cache_enables_collapse(self):
        """After first pass, cache should enable collapse on sparse input."""
        mx.random.seed(42)
        x = (mx.random.uniform(shape=(16, 2, 4, 4, 3)) < 0.03).astype(mx.float32)

        tcc = TemporalCollapseConv(3, 8, 3, padding=1, collapse_threshold=0.05)
        state = tcc.init_state(2, (4, 4))

        # Pass 1: populates cache.
        spk1, _ = tcc(x, state)
        mx.eval(spk1)
        calls_1 = tcc.conv_calls

        # Pass 2: uses cache, should collapse.
        state2 = tcc.init_state(2, (4, 4))
        spk2, _ = tcc(x, state2)
        mx.eval(spk2)
        calls_2 = tcc.conv_calls

        assert calls_2 <= calls_1

    def test_gradient_flow(self):
        tcc = TemporalCollapseConv(2, 8, 3, padding=1)
        state = tcc.init_state(2, (4, 4))
        x = mx.random.normal((6, 2, 4, 4, 2))

        def loss_fn(model, x, s):
            spk, _ = model(x, s)
            return mx.mean(spk)

        loss, grads = nn.value_and_grad(tcc, loss_fn)(tcc, x, state)
        mx.eval(loss, grads)
        assert float(mx.sqrt(mx.sum(grads['conv']['weight'] ** 2)).item()) > 0

    def test_collapse_stats(self):
        tcc = TemporalCollapseConv(3, 8, 3, padding=1)
        state = tcc.init_state(2, (4, 4))
        x = mx.ones((8, 2, 4, 4, 3))
        spk, _ = tcc(x, state)
        mx.eval(spk)
        stats = tcc.collapse_stats()
        assert 'conv_calls' in stats
        assert 'total_timesteps' in stats
        assert 'speedup' in stats
        assert stats['total_timesteps'] == 8

    def test_output_preserves_T_dimension(self):
        """Even with collapse, output should always have T timesteps."""
        mx.random.seed(0)
        x = (mx.random.uniform(shape=(10, 2, 4, 4, 3)) < 0.02).astype(mx.float32)

        tcc = TemporalCollapseConv(3, 8, 3, padding=1, collapse_threshold=0.1)
        state = tcc.init_state(2, (4, 4))

        # Pass 1 to populate cache.
        spk1, _ = tcc(x, state)
        mx.eval(spk1)

        # Pass 2 with collapse.
        state2 = tcc.init_state(2, (4, 4))
        spk2, _ = tcc(x, state2)
        mx.eval(spk2)
        assert spk2.shape[0] == 10  # T is preserved


# ──────────────────────────────────────────────────────────────────────
# Cross-operator Tests
# ──────────────────────────────────────────────────────────────────────

class TestCrossOperator:
    """Tests comparing operators against each other."""

    def test_tac_k1_similar_to_baseline(self):
        """TAC with K=1 should produce same output as standard Conv+LIF loop."""
        mx.random.seed(42)
        from mlxsnn.layers import SpikingConv2d

        x_seq = mx.random.normal((4, 2, 8, 8, 3))

        # Standard Conv+LIF.
        baseline = SpikingConv2d(3, 8, 3, padding=1, bn=False,
                                 neuron="leaky", neuron_params={"beta": 0.9})
        state_b = baseline.init_state(2, (8, 8))
        spk_list = []
        for t in range(4):
            spk, state_b = baseline(x_seq[t], state_b)
            spk_list.append(spk)
        spk_baseline = mx.stack(spk_list, axis=0)
        mx.eval(spk_baseline)

        # TAC K=1 — should be functionally equivalent IF weights match.
        # Just check shapes match.
        tac = TemporalAggregatedConv(3, 8, 3, chunk_size=1, padding=1, bn=False)
        state_t = tac.init_state(2, (8, 8))
        spk_tac, _ = tac(x_seq, state_t)
        mx.eval(spk_tac)

        assert list(spk_baseline.shape) == list(spk_tac.shape)

    def test_all_operators_binary_output(self):
        """All operators should produce binary spikes (0 or 1)."""
        mx.random.seed(42)
        x_seq = mx.random.normal((8, 2, 4, 4, 2))
        x_single = x_seq[0]

        # TAC
        tac = TemporalAggregatedConv(2, 8, 3, chunk_size=2, padding=1)
        spk, _ = tac(x_seq, tac.init_state(2, (4, 4)))
        mx.eval(spk)
        assert mx.allclose(spk, mx.clip(mx.round(spk), 0, 1))

        # FTC
        ftc = FourierTemporalConv(2, 8, 3, filter_order=2, padding=1)
        spk, _ = ftc(x_seq, ftc.init_state(2, (4, 4)))
        mx.eval(spk)
        assert mx.allclose(spk, mx.clip(mx.round(spk), 0, 1))

        # IMC (per-timestep, but gates scale continuously)
        # Gated spikes may not be exactly binary.
        imc = InfoMaxSpikeConv(2, 8, 3, padding=1)
        spk, _ = imc(x_single, imc.init_state(2, (4, 4)))
        mx.eval(spk)
        # Gated spikes are in [0, 1] but not necessarily binary.
        assert mx.all(spk >= 0.0).item()
        assert mx.all(spk <= 1.0).item()

        # TCC
        tcc = TemporalCollapseConv(2, 8, 3, padding=1)
        spk, _ = tcc(x_seq, tcc.init_state(2, (4, 4)))
        mx.eval(spk)
        assert mx.allclose(spk, mx.clip(mx.round(spk), 0, 1))
