"""Export mlx-snn models to NIR (Neuromorphic Intermediate Representation).

Converts mlx-snn neuron layers and MLX linear layers into a ``nir.NIRGraph``
that can be saved to disk and loaded by any NIR-compatible framework.

Requires the ``nir`` package: ``pip install mlx-snn[nir]``

Examples:
    >>> import mlx.nn as nn
    >>> import mlxsnn
    >>> layers = [
    ...     ('fc1', nn.Linear(784, 128)),
    ...     ('lif1', mlxsnn.Leaky(beta=0.9)),
    ...     ('fc2', nn.Linear(128, 10)),
    ...     ('lif2', mlxsnn.Leaky(beta=0.9)),
    ... ]
    >>> graph = mlxsnn.export_to_nir(layers, input_shape=(784,))
    >>> import nir
    >>> nir.write('model.nir', graph)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

import mlx.nn as nn

try:
    import nir
except ImportError:
    raise ImportError(
        "nir package is required for NIR export. "
        "Install with: pip install mlx-snn[nir]"
    )

from mlxsnn.neurons.lif import Leaky
from mlxsnn.neurons.if_neuron import IF
from mlxsnn.neurons.synaptic import Synaptic
from mlxsnn.nir_utils import beta_to_tau, beta_to_r, mx_to_numpy, DEFAULT_DT


def _convert_linear(module: nn.Linear) -> nir.NIRNode:
    """Convert an MLX Linear layer to NIR Affine or Linear node.

    Args:
        module: An ``mlx.nn.Linear`` layer.

    Returns:
        ``nir.Affine`` if bias is present, ``nir.Linear`` otherwise.
    """
    weight = mx_to_numpy(module.weight)  # shape (out, in), same as NIR
    if hasattr(module, "bias") and module.bias is not None:
        bias = mx_to_numpy(module.bias)
        return nir.Affine(weight=weight, bias=bias)
    return nir.Linear(weight=weight)


def _convert_leaky(module: Leaky, num_neurons: int,
                   dt: float = DEFAULT_DT) -> nir.LIF:
    """Convert a Leaky (LIF) neuron to NIR LIF node.

    Args:
        module: A ``Leaky`` neuron layer.
        num_neurons: Number of neurons (inferred from preceding Linear).
        dt: Simulation timestep.

    Returns:
        A ``nir.LIF`` node with shape ``(num_neurons,)`` arrays.
    """
    beta = module._get_beta()
    if hasattr(beta, 'item'):
        beta = float(beta.item())
    else:
        beta = float(beta)

    tau = beta_to_tau(beta, dt)
    r = beta_to_r(beta)

    return nir.LIF(
        tau=np.full(num_neurons, tau, dtype=np.float32),
        r=np.full(num_neurons, r, dtype=np.float32),
        v_leak=np.zeros(num_neurons, dtype=np.float32),
        v_threshold=np.full(num_neurons, module.threshold, dtype=np.float32),
    )


def _convert_if(module: IF, num_neurons: int) -> nir.IF:
    """Convert an IF neuron to NIR IF node.

    Args:
        module: An ``IF`` neuron layer.
        num_neurons: Number of neurons.

    Returns:
        A ``nir.IF`` node with shape ``(num_neurons,)`` arrays.
    """
    return nir.IF(
        r=np.ones(num_neurons, dtype=np.float32),
        v_threshold=np.full(num_neurons, module.threshold, dtype=np.float32),
    )


def _convert_synaptic(module: Synaptic, num_neurons: int,
                      dt: float = DEFAULT_DT) -> nir.CubaLIF:
    """Convert a Synaptic neuron to NIR CubaLIF node.

    Args:
        module: A ``Synaptic`` neuron layer.
        num_neurons: Number of neurons.
        dt: Simulation timestep.

    Returns:
        A ``nir.CubaLIF`` node with shape ``(num_neurons,)`` arrays.
    """
    alpha = module._get_alpha()
    beta = module._get_beta()
    if hasattr(alpha, 'item'):
        alpha = float(alpha.item())
    else:
        alpha = float(alpha)
    if hasattr(beta, 'item'):
        beta = float(beta.item())
    else:
        beta = float(beta)

    tau_syn = beta_to_tau(alpha, dt)
    tau_mem = beta_to_tau(beta, dt)
    r = beta_to_r(beta)

    return nir.CubaLIF(
        tau_syn=np.full(num_neurons, tau_syn, dtype=np.float32),
        tau_mem=np.full(num_neurons, tau_mem, dtype=np.float32),
        r=np.full(num_neurons, r, dtype=np.float32),
        v_leak=np.zeros(num_neurons, dtype=np.float32),
        v_threshold=np.full(num_neurons, module.threshold, dtype=np.float32),
        w_in=np.ones(num_neurons, dtype=np.float32),
    )


def export_to_nir(
    layers: List[Tuple[str, nn.Module]],
    input_shape: tuple = None,
    dt: float = DEFAULT_DT,
) -> "nir.NIRGraph":
    """Export a list of mlx-snn layers to a NIR graph.

    Since mlx-snn models are typically custom ``nn.Module`` compositions
    (not a standard sequential container), this function takes an explicit
    list of ``(name, module)`` pairs defining the forward-pass order.

    Args:
        layers: Ordered list of ``(name, module)`` tuples. Supported module
            types: ``nn.Linear``, ``Leaky``, ``IF``, ``Synaptic``.
        input_shape: Shape of the input tensor (excluding batch dimension).
            If None, inferred from the first Linear layer's input features.
        dt: Simulation timestep for continuous-time conversion.

    Returns:
        A ``nir.NIRGraph`` representing the model.

    Raises:
        TypeError: If an unsupported module type is encountered.
        ValueError: If neuron feature count cannot be inferred.

    Examples:
        >>> import mlx.nn as nn, mlxsnn
        >>> layers = [
        ...     ('fc1', nn.Linear(784, 128)),
        ...     ('lif1', mlxsnn.Leaky(beta=0.9)),
        ...     ('fc2', nn.Linear(128, 10)),
        ...     ('lif2', mlxsnn.Leaky(beta=0.9)),
        ... ]
        >>> graph = mlxsnn.export_to_nir(layers)
    """
    nodes = {}
    edges = []
    last_linear_out = None

    # Infer input shape from first Linear layer
    if input_shape is None:
        for _, module in layers:
            if isinstance(module, nn.Linear):
                input_shape = (module.weight.shape[1],)
                break
    if input_shape is None:
        raise ValueError("Cannot infer input_shape; pass it explicitly.")

    nodes["input"] = nir.Input(input_type={"input": np.array(input_shape)})

    prev_name = "input"
    for name, module in layers:
        if isinstance(module, nn.Linear):
            nodes[name] = _convert_linear(module)
            last_linear_out = module.weight.shape[0]
        elif isinstance(module, Leaky):
            if last_linear_out is None:
                raise ValueError(
                    f"Cannot infer neuron count for '{name}'. "
                    "Place a Linear layer before each neuron layer."
                )
            nodes[name] = _convert_leaky(module, last_linear_out, dt)
        elif isinstance(module, IF):
            if last_linear_out is None:
                raise ValueError(
                    f"Cannot infer neuron count for '{name}'. "
                    "Place a Linear layer before each neuron layer."
                )
            nodes[name] = _convert_if(module, last_linear_out)
        elif isinstance(module, Synaptic):
            if last_linear_out is None:
                raise ValueError(
                    f"Cannot infer neuron count for '{name}'. "
                    "Place a Linear layer before each neuron layer."
                )
            nodes[name] = _convert_synaptic(module, last_linear_out, dt)
        else:
            raise TypeError(
                f"Unsupported module type for NIR export: {type(module).__name__}. "
                "Supported types: nn.Linear, Leaky, IF, Synaptic."
            )

        edges.append((prev_name, name))
        prev_name = name

    # Infer output shape from last node
    if last_linear_out is not None:
        output_shape = (last_linear_out,)
    elif input_shape is not None:
        output_shape = input_shape
    else:
        output_shape = (1,)

    nodes["output"] = nir.Output(
        output_type={"output": np.array(output_shape)}
    )
    edges.append((prev_name, "output"))

    return nir.NIRGraph(nodes=nodes, edges=edges)
