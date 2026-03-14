# NIR Interoperability

[NIR](https://github.com/neuromorphs/NIR) (Neuromorphic Intermediate Representation) enables cross-framework SNN model exchange between simulators and neuromorphic hardware platforms.

!!! note "Optional dependency"
    ```bash
    pip install mlx-snn[nir]
    ```

## Export

::: mlxsnn.nir_export.export_to_nir

## Import

::: mlxsnn.nir_import.import_from_nir

## NIR Sequential Model

::: mlxsnn.nir_import.NIRSequential

## Supported Conversions

| mlx-snn | NIR | Direction |
|---------|-----|-----------|
| `nn.Linear` | `nir.Affine` / `nir.Linear` | Export & Import |
| `Leaky` | `nir.LIF` | Export & Import |
| `IF` | `nir.IF` | Export & Import |
| `Synaptic` | `nir.CubaLIF` | Export & Import |

## Example

### Export

```python
import mlx.nn as nn
import mlxsnn, nir

layers = [
    ('fc1', nn.Linear(784, 128)),
    ('lif1', mlxsnn.Leaky(beta=0.9)),
    ('fc2', nn.Linear(128, 10)),
    ('lif2', mlxsnn.Leaky(beta=0.9)),
]
graph = mlxsnn.export_to_nir(layers)
nir.write('model.nir', graph)
```

### Import

```python
graph = nir.read('model.nir')
model = mlxsnn.import_from_nir(graph)
state = model.init_states(batch_size=32)
out, state = model(x, state)
```
