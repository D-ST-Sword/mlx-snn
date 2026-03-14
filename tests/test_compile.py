"""Tests for mx.compile optimization utilities."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

import mlxsnn
from mlxsnn.training.compile import compiled_step, compiled_forward


class TestCompiledStep:
    def test_basic_lif(self):
        lif = mlxsnn.Leaky(beta=0.9)
        step_fn = compiled_step(lif)
        state = lif.init_state(4, 10)
        x = mx.random.normal((4, 10))
        spk, new_state = step_fn(x, state)
        mx.eval(spk, new_state["mem"])
        assert spk.shape == (4, 10)
        assert new_state["mem"].shape == (4, 10)

    def test_output_matches_uncompiled(self):
        """Compiled and uncompiled should produce same results."""
        lif = mlxsnn.Leaky(beta=0.9)
        state = lif.init_state(4, 10)
        x = mx.random.normal((4, 10))
        mx.eval(x)

        # Uncompiled
        spk1, state1 = lif(x, state)
        mx.eval(spk1, state1["mem"])

        # Compiled
        step_fn = compiled_step(lif)
        spk2, state2 = step_fn(x, state)
        mx.eval(spk2, state2["mem"])

        assert mx.allclose(spk1, spk2).item()
        assert mx.allclose(state1["mem"], state2["mem"]).item()

    def test_if_neuron(self):
        ifn = mlxsnn.IF()
        step_fn = compiled_step(ifn)
        state = ifn.init_state(2, 8)
        x = mx.random.normal((2, 8))
        spk, new_state = step_fn(x, state)
        mx.eval(spk, new_state["mem"])
        assert spk.shape == (2, 8)


class TestCompiledForward:
    def test_basic_shape(self):
        lif = mlxsnn.Leaky(beta=0.9)
        state = lif.init_state(4, 10)
        spikes = mx.random.normal((10, 4, 10))

        all_spk, all_mem, final_state = compiled_forward(lif, spikes, state)
        mx.eval(all_spk, all_mem)
        assert all_spk.shape == (10, 4, 10)
        assert all_mem.shape == (10, 4, 10)

    def test_matches_bptt_forward(self):
        """compiled_forward should match bptt_forward output."""
        lif = mlxsnn.Leaky(beta=0.9)
        state = lif.init_state(4, 10)
        spikes = mx.random.normal((8, 4, 10))
        mx.eval(spikes)

        spk1, mem1, st1 = mlxsnn.bptt_forward(lif, spikes, state)
        mx.eval(spk1, mem1)

        spk2, mem2, st2 = compiled_forward(lif, spikes, state)
        mx.eval(spk2, mem2)

        assert mx.allclose(spk1, spk2).item()
        assert mx.allclose(mem1, mem2).item()

    def test_num_steps_override(self):
        lif = mlxsnn.Leaky(beta=0.9)
        state = lif.init_state(2, 5)
        spikes = mx.random.normal((20, 2, 5))

        all_spk, _, _ = compiled_forward(lif, spikes, state, num_steps=5)
        mx.eval(all_spk)
        assert all_spk.shape == (5, 2, 5)

    def test_import_from_mlxsnn(self):
        assert hasattr(mlxsnn, "compiled_step")
        assert hasattr(mlxsnn, "compiled_forward")
