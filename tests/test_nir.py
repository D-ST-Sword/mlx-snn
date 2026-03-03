"""Tests for NIR (Neuromorphic Intermediate Representation) backend.

Covers:
- nir_utils: beta<->tau conversion, array conversion
- nir_export: export mlx-snn layers to NIR graphs
- nir_import: import NIR graphs into mlx-snn models
- roundtrip: export -> import parameter and forward-pass equivalence
- file I/O: write -> read -> import via nir.write/nir.read
"""

import numpy as np
import pytest

import mlx.core as mx
import mlx.nn as nn

import nir

from mlxsnn.neurons.lif import Leaky
from mlxsnn.neurons.if_neuron import IF
from mlxsnn.neurons.synaptic import Synaptic
from mlxsnn.nir_utils import (
    DEFAULT_DT,
    beta_to_tau,
    tau_to_beta,
    beta_to_r,
    mx_to_numpy,
    numpy_to_mx,
)
from mlxsnn.nir_export import (
    export_to_nir,
    _convert_linear,
    _convert_leaky,
    _convert_if,
    _convert_synaptic,
)
from mlxsnn.nir_import import (
    import_from_nir,
    NIRSequential,
    _topological_sort,
    _convert_nir_affine,
    _convert_nir_linear,
    _convert_nir_lif,
    _convert_nir_if,
    _convert_nir_cubalif,
)


# ============================================================
# TestNIRUtils
# ============================================================

class TestNIRUtils:
    """Tests for beta <-> tau conversion utilities."""

    def test_beta_to_tau_typical(self):
        tau = beta_to_tau(0.9, dt=1e-4)
        assert np.isclose(tau, 1e-3)  # dt / (1 - 0.9) = 1e-4 / 0.1

    def test_beta_to_tau_zero(self):
        tau = beta_to_tau(0.0, dt=1e-4)
        assert np.isclose(tau, 1e-4)  # dt / 1.0

    def test_beta_to_tau_high(self):
        tau = beta_to_tau(0.99, dt=1e-4)
        assert np.isclose(tau, 1e-2)

    def test_beta_to_tau_invalid_ge1(self):
        with pytest.raises(ValueError, match="beta must be in"):
            beta_to_tau(1.0)

    def test_beta_to_tau_invalid_negative(self):
        with pytest.raises(ValueError, match="beta must be in"):
            beta_to_tau(-0.1)

    def test_tau_to_beta_typical(self):
        beta = tau_to_beta(1e-3, dt=1e-4)
        assert np.isclose(beta, 0.9)

    def test_tau_to_beta_invalid(self):
        with pytest.raises(ValueError, match="tau must be > 0"):
            tau_to_beta(0.0)

    def test_tau_to_beta_negative(self):
        with pytest.raises(ValueError, match="tau must be > 0"):
            tau_to_beta(-1.0)

    def test_beta_tau_roundtrip(self):
        """beta -> tau -> beta should be identity."""
        for beta in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]:
            tau = beta_to_tau(beta)
            beta_rt = tau_to_beta(tau)
            assert np.isclose(beta_rt, beta, atol=1e-10), \
                f"Roundtrip failed for beta={beta}"

    def test_tau_beta_roundtrip(self):
        """tau -> beta -> tau should be identity."""
        for tau in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
            beta = tau_to_beta(tau)
            tau_rt = beta_to_tau(beta)
            assert np.isclose(tau_rt, tau, atol=1e-10), \
                f"Roundtrip failed for tau={tau}"

    def test_beta_to_r_typical(self):
        r = beta_to_r(0.9)
        assert np.isclose(r, 10.0)

    def test_beta_to_r_zero(self):
        r = beta_to_r(0.0)
        assert np.isclose(r, 1.0)

    def test_beta_to_r_invalid(self):
        with pytest.raises(ValueError):
            beta_to_r(1.0)

    def test_mx_to_numpy_array(self):
        arr = mx.array([1.0, 2.0, 3.0])
        result = mx_to_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_mx_to_numpy_scalar(self):
        result = mx_to_numpy(0.9)
        assert isinstance(result, (np.floating, np.ndarray))
        assert np.isclose(float(result), 0.9)

    def test_mx_to_numpy_ndarray(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        result = mx_to_numpy(arr)
        assert result.dtype == np.float32

    def test_numpy_to_mx(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = numpy_to_mx(arr)
        assert isinstance(result, mx.array)
        mx.eval(result)
        np.testing.assert_allclose(np.array(result), [1.0, 2.0, 3.0])


# ============================================================
# TestNIRExport
# ============================================================

class TestNIRExport:
    """Tests for exporting mlx-snn layers to NIR."""

    def test_convert_linear_with_bias(self):
        layer = nn.Linear(784, 128)
        node = _convert_linear(layer)
        assert isinstance(node, nir.Affine)
        assert node.weight.shape == (128, 784)
        assert node.bias.shape == (128,)

    def test_convert_linear_no_bias(self):
        layer = nn.Linear(784, 128, bias=False)
        node = _convert_linear(layer)
        assert isinstance(node, nir.Linear)
        assert node.weight.shape == (128, 784)

    def test_convert_leaky(self):
        lif = Leaky(beta=0.9, threshold=1.5)
        node = _convert_leaky(lif, num_neurons=128)
        assert isinstance(node, nir.LIF)
        assert node.tau.shape == (128,)
        assert node.r.shape == (128,)
        assert node.v_leak.shape == (128,)
        assert node.v_threshold.shape == (128,)
        np.testing.assert_allclose(node.tau, 1e-3, rtol=1e-5)
        np.testing.assert_allclose(node.v_threshold, 1.5)

    def test_convert_leaky_learnable_beta(self):
        lif = Leaky(beta=0.95, learn_beta=True)
        mx.eval(lif.parameters())
        node = _convert_leaky(lif, num_neurons=64)
        expected_tau = DEFAULT_DT / (1.0 - 0.95)
        np.testing.assert_allclose(node.tau, expected_tau, rtol=1e-5)

    def test_convert_if(self):
        neuron = IF(threshold=2.0)
        node = _convert_if(neuron, num_neurons=64)
        assert isinstance(node, nir.IF)
        assert node.r.shape == (64,)
        np.testing.assert_allclose(node.r, 1.0)
        np.testing.assert_allclose(node.v_threshold, 2.0)

    def test_convert_synaptic(self):
        neuron = Synaptic(alpha=0.8, beta=0.9)
        node = _convert_synaptic(neuron, num_neurons=32)
        assert isinstance(node, nir.CubaLIF)
        assert node.tau_syn.shape == (32,)
        assert node.tau_mem.shape == (32,)
        expected_tau_syn = DEFAULT_DT / (1.0 - 0.8)
        expected_tau_mem = DEFAULT_DT / (1.0 - 0.9)
        np.testing.assert_allclose(node.tau_syn, expected_tau_syn, rtol=1e-5)
        np.testing.assert_allclose(node.tau_mem, expected_tau_mem, rtol=1e-5)

    def test_export_basic_graph(self):
        layers = [
            ('fc1', nn.Linear(784, 128)),
            ('lif1', Leaky(beta=0.9)),
            ('fc2', nn.Linear(128, 10)),
            ('lif2', Leaky(beta=0.9)),
        ]
        graph = export_to_nir(layers)
        assert isinstance(graph, nir.NIRGraph)
        assert 'input' in graph.nodes
        assert 'output' in graph.nodes
        assert 'fc1' in graph.nodes
        assert 'lif1' in graph.nodes
        assert 'fc2' in graph.nodes
        assert 'lif2' in graph.nodes

    def test_export_graph_edges(self):
        layers = [
            ('fc1', nn.Linear(10, 5)),
            ('lif1', Leaky(beta=0.9)),
        ]
        graph = export_to_nir(layers, input_shape=(10,))
        expected_edges = [
            ('input', 'fc1'), ('fc1', 'lif1'), ('lif1', 'output')
        ]
        assert graph.edges == expected_edges

    def test_export_infers_input_shape(self):
        layers = [
            ('fc1', nn.Linear(784, 128)),
            ('lif1', Leaky(beta=0.9)),
        ]
        graph = export_to_nir(layers)
        input_node = graph.nodes['input']
        np.testing.assert_array_equal(input_node.input_type['input'], [784])

    def test_export_explicit_input_shape(self):
        layers = [
            ('fc1', nn.Linear(100, 50)),
            ('lif1', Leaky(beta=0.9)),
        ]
        graph = export_to_nir(layers, input_shape=(100,))
        np.testing.assert_array_equal(
            graph.nodes['input'].input_type['input'], [100]
        )

    def test_export_weight_values_match(self):
        fc = nn.Linear(10, 5)
        mx.eval(fc.parameters())
        original_weight = np.array(fc.weight)
        layers = [('fc1', fc), ('lif1', Leaky(beta=0.9))]
        graph = export_to_nir(layers)
        np.testing.assert_allclose(
            graph.nodes['fc1'].weight, original_weight, atol=1e-6
        )

    def test_export_unsupported_type_raises(self):
        from mlxsnn.neurons.izhikevich import Izhikevich
        layers = [
            ('fc1', nn.Linear(10, 5)),
            ('izh', Izhikevich()),
        ]
        with pytest.raises(TypeError, match="Unsupported module type"):
            export_to_nir(layers)

    def test_export_neuron_without_linear_raises(self):
        layers = [('lif1', Leaky(beta=0.9))]
        with pytest.raises(ValueError, match="Cannot infer neuron count"):
            export_to_nir(layers, input_shape=(10,))

    def test_export_with_if_neuron(self):
        layers = [
            ('fc1', nn.Linear(10, 5)),
            ('if1', IF(threshold=1.0)),
        ]
        graph = export_to_nir(layers)
        assert isinstance(graph.nodes['if1'], nir.IF)

    def test_export_with_synaptic(self):
        layers = [
            ('fc1', nn.Linear(10, 5)),
            ('syn1', Synaptic(alpha=0.8, beta=0.9)),
        ]
        graph = export_to_nir(layers)
        assert isinstance(graph.nodes['syn1'], nir.CubaLIF)


# ============================================================
# TestNIRImport
# ============================================================

class TestNIRImport:
    """Tests for importing NIR graphs into mlx-snn."""

    def _make_simple_graph(self):
        """Helper: build a simple Linear -> LIF NIR graph."""
        return nir.NIRGraph(
            nodes={
                'input': nir.Input(input_type={'input': np.array([10])}),
                'fc1': nir.Affine(
                    weight=np.random.randn(5, 10).astype(np.float32),
                    bias=np.zeros(5, dtype=np.float32),
                ),
                'lif1': nir.LIF(
                    tau=np.full(5, 1e-3, dtype=np.float32),
                    r=np.full(5, 10.0, dtype=np.float32),
                    v_leak=np.zeros(5, dtype=np.float32),
                    v_threshold=np.ones(5, dtype=np.float32),
                ),
                'output': nir.Output(output_type={'output': np.array([5])}),
            },
            edges=[('input', 'fc1'), ('fc1', 'lif1'), ('lif1', 'output')],
        )

    def test_topological_sort_simple(self):
        graph = self._make_simple_graph()
        order = _topological_sort(graph.nodes, graph.edges)
        assert order == ['fc1', 'lif1']

    def test_topological_sort_multi(self):
        nodes = {
            'input': nir.Input(input_type={'input': np.array([10])}),
            'fc1': nir.Affine(
                weight=np.eye(5, 10, dtype=np.float32),
                bias=np.zeros(5, dtype=np.float32),
            ),
            'lif1': nir.LIF(
                tau=np.ones(5, dtype=np.float32) * 1e-3,
                r=np.ones(5, dtype=np.float32) * 10,
                v_leak=np.zeros(5, dtype=np.float32),
                v_threshold=np.ones(5, dtype=np.float32),
            ),
            'fc2': nir.Affine(
                weight=np.eye(3, 5, dtype=np.float32),
                bias=np.zeros(3, dtype=np.float32),
            ),
            'output': nir.Output(output_type={'output': np.array([3])}),
        }
        edges = [
            ('input', 'fc1'), ('fc1', 'lif1'),
            ('lif1', 'fc2'), ('fc2', 'output'),
        ]
        order = _topological_sort(nodes, edges)
        assert order == ['fc1', 'lif1', 'fc2']

    def test_convert_nir_affine(self):
        node = nir.Affine(
            weight=np.eye(5, 10, dtype=np.float32),
            bias=np.ones(5, dtype=np.float32),
        )
        layer = _convert_nir_affine(node)
        assert isinstance(layer, nn.Linear)
        mx.eval(layer.parameters())
        np.testing.assert_allclose(np.array(layer.weight), np.eye(5, 10), atol=1e-6)
        np.testing.assert_allclose(np.array(layer.bias), np.ones(5), atol=1e-6)

    def test_convert_nir_linear(self):
        node = nir.Linear(weight=np.eye(5, 10, dtype=np.float32))
        layer = _convert_nir_linear(node)
        assert isinstance(layer, nn.Linear)
        mx.eval(layer.parameters())
        assert not hasattr(layer, 'bias') or layer.get('bias') is None

    def test_convert_nir_lif(self):
        node = nir.LIF(
            tau=np.full(5, 1e-3, dtype=np.float32),
            r=np.full(5, 10.0, dtype=np.float32),
            v_leak=np.zeros(5, dtype=np.float32),
            v_threshold=np.full(5, 1.5, dtype=np.float32),
        )
        lif = _convert_nir_lif(node)
        assert isinstance(lif, Leaky)
        expected_beta = 1.0 - DEFAULT_DT / 1e-3  # 0.9
        assert np.isclose(lif._get_beta(), expected_beta, atol=1e-6)
        assert np.isclose(lif.threshold, 1.5)

    def test_convert_nir_if(self):
        node = nir.IF(
            r=np.ones(5, dtype=np.float32),
            v_threshold=np.full(5, 2.0, dtype=np.float32),
        )
        neuron = _convert_nir_if(node)
        assert isinstance(neuron, IF)
        assert np.isclose(neuron.threshold, 2.0)

    def test_convert_nir_cubalif(self):
        node = nir.CubaLIF(
            tau_syn=np.full(5, 5e-4, dtype=np.float32),
            tau_mem=np.full(5, 1e-3, dtype=np.float32),
            r=np.full(5, 10.0, dtype=np.float32),
            v_leak=np.zeros(5, dtype=np.float32),
            v_threshold=np.ones(5, dtype=np.float32),
        )
        neuron = _convert_nir_cubalif(node)
        assert isinstance(neuron, Synaptic)
        expected_alpha = 1.0 - DEFAULT_DT / 5e-4  # 0.8
        expected_beta = 1.0 - DEFAULT_DT / 1e-3   # 0.9
        assert np.isclose(neuron._get_alpha(), expected_alpha, atol=1e-6)
        assert np.isclose(neuron._get_beta(), expected_beta, atol=1e-6)

    def test_import_simple_graph(self):
        graph = self._make_simple_graph()
        model = import_from_nir(graph)
        assert isinstance(model, NIRSequential)
        assert model.layer_names == ['fc1', 'lif1']
        assert isinstance(model.fc1, nn.Linear)
        assert isinstance(model.lif1, Leaky)

    def test_import_forward_pass_shape(self):
        graph = self._make_simple_graph()
        model = import_from_nir(graph)
        state = model.init_states(batch_size=8)
        x = mx.ones((8, 10))
        out, new_state = model(x, state)
        mx.eval(out)
        assert out.shape == (8, 5)

    def test_import_with_if(self):
        graph = nir.NIRGraph(
            nodes={
                'input': nir.Input(input_type={'input': np.array([4])}),
                'fc1': nir.Affine(
                    weight=np.eye(3, 4, dtype=np.float32),
                    bias=np.zeros(3, dtype=np.float32),
                ),
                'if1': nir.IF(
                    r=np.ones(3, dtype=np.float32),
                    v_threshold=np.ones(3, dtype=np.float32),
                ),
                'output': nir.Output(output_type={'output': np.array([3])}),
            },
            edges=[('input', 'fc1'), ('fc1', 'if1'), ('if1', 'output')],
        )
        model = import_from_nir(graph)
        assert isinstance(model.if1, IF)

    def test_import_warns_unsupported(self):
        graph = nir.NIRGraph(
            nodes={
                'input': nir.Input(input_type={'input': np.array([4])}),
                'delay': nir.Delay(delay=np.ones(4, dtype=np.float32)),
                'output': nir.Output(output_type={'output': np.array([4])}),
            },
            edges=[('input', 'delay'), ('delay', 'output')],
        )
        with pytest.warns(UserWarning, match="Unsupported NIR node type"):
            model = import_from_nir(graph)
        assert len(model.layer_names) == 0


# ============================================================
# TestNIRRoundtrip
# ============================================================

class TestNIRRoundtrip:
    """Tests for export -> import parameter and forward-pass equivalence."""

    def test_roundtrip_lif_params(self):
        """Export Leaky -> LIF -> import Leaky: beta should match."""
        original_beta = 0.9
        lif = Leaky(beta=original_beta, threshold=1.0)
        fc = nn.Linear(10, 5)
        mx.eval(fc.parameters())

        layers = [('fc1', fc), ('lif1', lif)]
        graph = export_to_nir(layers)
        model = import_from_nir(graph)

        imported_beta = model.lif1._get_beta()
        assert np.isclose(imported_beta, original_beta, atol=1e-6)
        assert np.isclose(model.lif1.threshold, 1.0, atol=1e-6)

    def test_roundtrip_if_params(self):
        """Export IF -> NIR IF -> import IF: threshold should match."""
        neuron = IF(threshold=2.0)
        fc = nn.Linear(10, 5)
        mx.eval(fc.parameters())

        layers = [('fc1', fc), ('if1', neuron)]
        graph = export_to_nir(layers)
        model = import_from_nir(graph)

        assert np.isclose(model.if1.threshold, 2.0, atol=1e-6)

    def test_roundtrip_synaptic_params(self):
        """Export Synaptic -> CubaLIF -> import Synaptic: alpha, beta match."""
        neuron = Synaptic(alpha=0.8, beta=0.9, threshold=1.5)
        fc = nn.Linear(10, 5)
        mx.eval(fc.parameters())

        layers = [('fc1', fc), ('syn1', neuron)]
        graph = export_to_nir(layers)
        model = import_from_nir(graph)

        assert np.isclose(model.syn1._get_alpha(), 0.8, atol=1e-6)
        assert np.isclose(model.syn1._get_beta(), 0.9, atol=1e-6)
        assert np.isclose(model.syn1.threshold, 1.5, atol=1e-6)

    def test_roundtrip_linear_weights(self):
        """Export -> import preserves Linear weight values."""
        fc = nn.Linear(10, 5)
        mx.eval(fc.parameters())
        original_w = np.array(fc.weight).copy()
        original_b = np.array(fc.bias).copy()

        layers = [('fc1', fc), ('lif1', Leaky(beta=0.9))]
        graph = export_to_nir(layers)
        model = import_from_nir(graph)
        mx.eval(model.parameters())

        np.testing.assert_allclose(np.array(model.fc1.weight), original_w, atol=1e-6)
        np.testing.assert_allclose(np.array(model.fc1.bias), original_b, atol=1e-6)

    def test_roundtrip_forward_pass(self):
        """Export -> import produces same forward-pass output."""
        fc = nn.Linear(10, 5)
        lif = Leaky(beta=0.9, threshold=1.0)
        mx.eval(fc.parameters())

        # Original forward pass
        state = lif.init_state(batch_size=4, features=5)
        x = mx.random.normal((4, 10))
        mx.eval(x)
        out1, state1 = lif(fc(x), state)
        mx.eval(out1, state1["mem"])

        # Export -> import -> forward pass
        layers = [('fc1', fc), ('lif1', lif)]
        graph = export_to_nir(layers)
        model = import_from_nir(graph)
        mx.eval(model.parameters())

        state2 = model.init_states(batch_size=4)
        out2, state2 = model(x, state2)
        mx.eval(out2, state2["lif1"]["mem"])

        np.testing.assert_allclose(
            np.array(out1), np.array(out2), atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(state1["mem"]), np.array(state2["lif1"]["mem"]), atol=1e-5
        )

    def test_roundtrip_multi_layer(self):
        """Roundtrip with multiple Linear+LIF pairs."""
        fc1 = nn.Linear(20, 10)
        fc2 = nn.Linear(10, 5)
        lif1 = Leaky(beta=0.9)
        lif2 = Leaky(beta=0.85)
        mx.eval(fc1.parameters())
        mx.eval(fc2.parameters())

        layers = [
            ('fc1', fc1), ('lif1', lif1),
            ('fc2', fc2), ('lif2', lif2),
        ]
        graph = export_to_nir(layers)
        model = import_from_nir(graph)
        mx.eval(model.parameters())

        state = model.init_states(batch_size=4)
        x = mx.random.normal((4, 20))
        mx.eval(x)
        out, state = model(x, state)
        mx.eval(out)
        assert out.shape == (4, 5)

    def test_roundtrip_various_betas(self):
        """Roundtrip preserves beta across a range of values."""
        for beta_val in [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]:
            lif = Leaky(beta=beta_val)
            fc = nn.Linear(4, 3)
            mx.eval(fc.parameters())
            graph = export_to_nir([('fc', fc), ('lif', lif)])
            model = import_from_nir(graph)
            assert np.isclose(model.lif._get_beta(), beta_val, atol=1e-6), \
                f"Roundtrip failed for beta={beta_val}"


# ============================================================
# TestNIRFileIO
# ============================================================

class TestNIRFileIO:
    """Tests for NIR file write/read/import."""

    def test_write_read_roundtrip(self, tmp_path):
        """Write NIR graph to file, read back, and import."""
        fc = nn.Linear(10, 5)
        lif = Leaky(beta=0.9)
        mx.eval(fc.parameters())
        original_w = np.array(fc.weight).copy()

        graph = export_to_nir([('fc1', fc), ('lif1', lif)])
        filepath = str(tmp_path / "test_model.nir")
        nir.write(filepath, graph)

        graph2 = nir.read(filepath)
        model = import_from_nir(graph2)
        mx.eval(model.parameters())

        np.testing.assert_allclose(
            np.array(model.fc1.weight), original_w, atol=1e-6
        )
        assert np.isclose(model.lif1._get_beta(), 0.9, atol=1e-6)

    def test_write_read_complex(self, tmp_path):
        """Write/read a multi-layer graph with different neuron types."""
        layers = [
            ('fc1', nn.Linear(20, 10)),
            ('lif1', Leaky(beta=0.9)),
            ('fc2', nn.Linear(10, 5)),
            ('if1', IF(threshold=1.5)),
        ]
        for _, m in layers:
            if hasattr(m, 'parameters'):
                mx.eval(m.parameters())

        graph = export_to_nir(layers)
        filepath = str(tmp_path / "complex.nir")
        nir.write(filepath, graph)

        graph2 = nir.read(filepath)
        model = import_from_nir(graph2)
        mx.eval(model.parameters())

        assert isinstance(model.fc1, nn.Linear)
        assert isinstance(model.lif1, Leaky)
        assert isinstance(model.fc2, nn.Linear)
        assert isinstance(model.if1, IF)

    def test_write_read_forward_pass(self, tmp_path):
        """Verify forward pass after file roundtrip."""
        fc = nn.Linear(8, 4)
        lif = Leaky(beta=0.9)
        mx.eval(fc.parameters())

        graph = export_to_nir([('fc1', fc), ('lif1', lif)])
        filepath = str(tmp_path / "fwd.nir")
        nir.write(filepath, graph)

        graph2 = nir.read(filepath)
        model = import_from_nir(graph2)
        mx.eval(model.parameters())

        state = model.init_states(batch_size=2)
        x = mx.ones((2, 8))
        out, state = model(x, state)
        mx.eval(out)
        assert out.shape == (2, 4)
