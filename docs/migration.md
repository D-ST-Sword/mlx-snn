# Migration from snnTorch

mlx-snn is designed to feel familiar to snnTorch users. This guide highlights the key differences.

## Side-by-Side Comparison

```python
# snnTorch                          # mlx-snn
import snntorch as snn              import mlxsnn
import torch                        import mlx.core as mx
import torch.nn as nn               import mlx.nn as nn

# Neuron creation
lif = snn.Leaky(beta=0.9)          lif = mlxsnn.Leaky(beta=0.9)

# State initialization
mem = lif.init_leaky()              state = lif.init_state(B, features)

# Forward pass
spk, mem = lif(x, mem)             spk, state = lif(x, state)
                                    # state["mem"] == mem
```

## Key Differences

### 1. State is a Dictionary

snnTorch returns separate tensors for each state variable. mlx-snn wraps everything in a `dict`:

```python
# snnTorch
spk, mem = lif(x, mem)              # 2 separate tensors
spk, syn, mem = synaptic(x, syn, mem)  # 3 tensors

# mlx-snn
spk, state = lif(x, state)          # state = {"mem": ...}
spk, state = synaptic(x, state)     # state = {"syn": ..., "mem": ...}
```

**Why?** Dict state works seamlessly with MLX's functional transforms (`mx.compile`, `mx.grad`) and is easier to serialize.

### 2. No Global Hidden State

snnTorch neurons can store state internally. mlx-snn always requires explicit state passing:

```python
# snnTorch (init_hidden=True)
lif = snn.Leaky(beta=0.9, init_hidden=True)
spk = lif(x)  # state stored internally

# mlx-snn (always explicit)
lif = mlxsnn.Leaky(beta=0.9)
state = lif.init_state(B, features)
spk, state = lif(x, state)
```

### 3. MLX Arrays Instead of Tensors

```python
# snnTorch
x = torch.randn(8, 784)
x = x.to("cuda")  # explicit device

# mlx-snn
x = mx.random.normal((8, 784))
# No device placement needed — unified memory
```

### 4. Time-First Format

Both libraries use time-first format `[T, B, ...]`, so this stays the same.

### 5. Optimizer API

```python
# snnTorch (PyTorch)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# mlx-snn (MLX)
optimizer = optim.Adam(learning_rate=1e-3)
loss, grads = loss_and_grad_fn(model, x, y)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)
```

## Feature Mapping

| snnTorch | mlx-snn | Notes |
|----------|---------|-------|
| `snn.Leaky` | `mlxsnn.Leaky` | Same API |
| `snn.Synaptic` | `mlxsnn.Synaptic` | State keys: `syn`, `mem` |
| `snn.Alpha` | `mlxsnn.Alpha` | State keys: `syn`, `mem` |
| `snn.RLeaky` | `mlxsnn.RLeaky` | Uses `V` param for recurrent weight |
| `snn.RSynaptic` | `mlxsnn.RSynaptic` | Uses `V` param |
| `snn.Lapicque` | — | Not yet implemented |
| `snn.SLSTM` | — | Not yet implemented |
| `spikegen.rate` | `mlxsnn.rate_encode` | Same behavior |
| `spikegen.latency` | `mlxsnn.latency_encode` | Same behavior |
| `snn.functional.ce_rate_loss` | `mlxsnn.ce_rate_loss` | Same behavior |
| `snn.functional.mse_count_loss` | `mlxsnn.mse_count_loss` | Same behavior |
| `snn.export_to_nir` | `mlxsnn.export_to_nir` | NIR interop |
| `snn.import_from_nir` | `mlxsnn.import_from_nir` | NIR interop |

## Learnable Parameters

```python
# snnTorch
lif = snn.Leaky(beta=0.9, learn_beta=True)

# mlx-snn (identical API)
lif = mlxsnn.Leaky(beta=0.9, learn_beta=True)

# Also supports learnable threshold
lif = mlxsnn.Leaky(beta=0.9, learn_threshold=True)
```
