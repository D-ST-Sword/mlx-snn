"""Import NIR graphs into mlx-snn models.

Converts a ``nir.NIRGraph`` into a runnable mlx-snn model by mapping NIR
node types to mlx-snn neuron layers and MLX linear layers.

Requires the ``nir`` package: ``pip install mlx-snn[nir]``

Examples:
    >>> import nir, mlxsnn
    >>> graph = nir.read('model.nir')
    >>> model = mlxsnn.import_from_nir(graph)
    >>> state = model.init_states(batch_size=32)
    >>> import mlx.core as mx
    >>> x = mx.ones((32, 784))
    >>> out, state = model(x, state)
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn

try:
    import nir
except ImportError:
    raise ImportError(
        "nir package is required for NIR import. "
        "Install with: pip install mlx-snn[nir]"
    )

from mlxsnn.neurons.lif import Leaky
from mlxsnn.neurons.if_neuron import IF
from mlxsnn.neurons.synaptic import Synaptic
from mlxsnn.nir_utils import tau_to_beta, numpy_to_mx, DEFAULT_DT


def _topological_sort(
    nodes: Dict[str, nir.NIRNode],
    edges: List[Tuple[str, str]],
) -> List[str]:
    """Topological sort of NIR graph nodes using Kahn's algorithm.

    Excludes ``nir.Input`` and ``nir.Output`` nodes from the result.

    Args:
        nodes: Dict mapping node names to NIR nodes.
        edges: List of ``(src, dst)`` edge tuples.

    Returns:
        List of node names in topological order.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    skip_types = (nir.Input, nir.Output)
    filtered = {k for k, v in nodes.items() if not isinstance(v, skip_types)}

    # Build adjacency and in-degree for filtered nodes
    in_degree = {k: 0 for k in filtered}
    adj = {k: [] for k in filtered}

    for src, dst in edges:
        if src in filtered and dst in filtered:
            adj[src].append(dst)
            in_degree[dst] += 1

    queue = deque(k for k, d in in_degree.items() if d == 0)
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(filtered):
        raise ValueError("NIR graph contains a cycle.")

    return result


def _convert_nir_affine(node: nir.Affine) -> nn.Linear:
    """Convert NIR Affine node to MLX Linear layer with bias."""
    out_features, in_features = node.weight.shape
    layer = nn.Linear(in_features, out_features)
    layer.weight = numpy_to_mx(node.weight)
    layer.bias = numpy_to_mx(node.bias)
    return layer


def _convert_nir_linear(node: nir.Linear) -> nn.Linear:
    """Convert NIR Linear node to MLX Linear layer without bias."""
    out_features, in_features = node.weight.shape
    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight = numpy_to_mx(node.weight)
    return layer


def _convert_nir_lif(node: nir.LIF, dt: float = DEFAULT_DT) -> Leaky:
    """Convert NIR LIF node to mlx-snn Leaky neuron."""
    # Use mean tau for scalar beta (all neurons share same beta in mlx-snn)
    tau = float(np.mean(node.tau))
    beta = tau_to_beta(tau, dt)
    threshold = float(np.mean(node.v_threshold))
    return Leaky(beta=beta, threshold=threshold, reset_mechanism="subtract")


def _convert_nir_if(node: nir.IF) -> IF:
    """Convert NIR IF node to mlx-snn IF neuron."""
    threshold = float(np.mean(node.v_threshold))
    return IF(threshold=threshold)


def _convert_nir_cubalif(node: nir.CubaLIF, dt: float = DEFAULT_DT) -> Synaptic:
    """Convert NIR CubaLIF node to mlx-snn Synaptic neuron."""
    tau_syn = float(np.mean(node.tau_syn))
    tau_mem = float(np.mean(node.tau_mem))
    alpha = tau_to_beta(tau_syn, dt)
    beta = tau_to_beta(tau_mem, dt)
    threshold = float(np.mean(node.v_threshold))
    return Synaptic(alpha=alpha, beta=beta, threshold=threshold)


class NIRSequential(nn.Module):
    """Sequential model built from an imported NIR graph.

    Stores layers as named attributes and runs them in topological order.
    Neuron layers receive and return state dicts; linear layers are stateless.

    Args:
        layer_names: Ordered list of layer names.
        layers: Dict mapping names to ``nn.Module`` instances.

    Examples:
        >>> model = import_from_nir(graph)
        >>> state = model.init_states(batch_size=32)
        >>> out, state = model(x, state)
    """

    def __init__(self, layer_names: List[str], layers: Dict[str, nn.Module]):
        super().__init__()
        self.layer_names = layer_names
        for name, layer in layers.items():
            setattr(self, name, layer)

    def __call__(
        self, x: mx.array, states: dict
    ) -> Tuple[mx.array, dict]:
        """Forward pass through all layers.

        Args:
            x: Input tensor ``[batch, features]``.
            states: Dict mapping neuron layer names to their state dicts.

        Returns:
            Tuple of ``(output, new_states)`` where new_states has the same
            keys as the input states.
        """
        new_states = {}
        for name in self.layer_names:
            layer = getattr(self, name)
            if isinstance(layer, (Leaky, IF, Synaptic)):
                x, new_states[name] = layer(x, states[name])
            else:
                x = layer(x)
        return x, new_states

    def init_states(self, batch_size: int) -> dict:
        """Initialize states for all neuron layers.

        Args:
            batch_size: Batch size for state tensors.

        Returns:
            Dict mapping neuron layer names to initialized state dicts.
        """
        states = {}
        for i, name in enumerate(self.layer_names):
            layer = getattr(self, name)
            if isinstance(layer, (Leaky, IF, Synaptic)):
                # Infer feature count from preceding linear layer
                features = self._infer_features(i)
                states[name] = layer.init_state(batch_size, features)
        return states

    def _infer_features(self, neuron_idx: int) -> int:
        """Infer neuron feature count from preceding Linear layer."""
        for i in range(neuron_idx - 1, -1, -1):
            layer = getattr(self, self.layer_names[i])
            if isinstance(layer, nn.Linear):
                return layer.weight.shape[0]
        raise ValueError(
            f"Cannot infer features for neuron at index {neuron_idx}. "
            "No preceding Linear layer found."
        )


def import_from_nir(
    graph: "nir.NIRGraph",
    dt: float = DEFAULT_DT,
) -> NIRSequential:
    """Import a NIR graph into an mlx-snn sequential model.

    Args:
        graph: A ``nir.NIRGraph`` to convert.
        dt: Simulation timestep for continuous-time to discrete-time
            conversion.

    Returns:
        A ``NIRSequential`` model with layers matching the NIR graph.

    Raises:
        ValueError: If the graph contains a cycle.

    Examples:
        >>> import nir, mlxsnn
        >>> graph = nir.read('model.nir')
        >>> model = mlxsnn.import_from_nir(graph)
    """
    sorted_names = _topological_sort(graph.nodes, graph.edges)

    layers = {}
    for name in sorted_names:
        node = graph.nodes[name]
        if isinstance(node, nir.Affine):
            layers[name] = _convert_nir_affine(node)
        elif isinstance(node, nir.Linear):
            layers[name] = _convert_nir_linear(node)
        elif isinstance(node, nir.LIF):
            layers[name] = _convert_nir_lif(node, dt)
        elif isinstance(node, nir.IF):
            layers[name] = _convert_nir_if(node)
        elif isinstance(node, nir.CubaLIF):
            layers[name] = _convert_nir_cubalif(node, dt)
        else:
            warnings.warn(
                f"Unsupported NIR node type '{type(node).__name__}' "
                f"for node '{name}'. Skipping.",
                stacklevel=2,
            )
            sorted_names = [n for n in sorted_names if n != name]

    return NIRSequential(layer_names=sorted_names, layers=layers)
