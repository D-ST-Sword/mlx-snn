# Getting Started

This guide walks you through building and training your first SNN with mlx-snn.

## Installation

```bash
pip install mlx-snn
```

## Core Concepts

### 1. Neuron Models

Spiking neurons are the building blocks of SNNs. Each neuron maintains an internal **membrane potential** that integrates input current over time and emits a binary **spike** when it crosses a threshold.

```python
import mlx.core as mx
import mlxsnn

# Create a Leaky Integrate-and-Fire neuron
lif = mlxsnn.Leaky(beta=0.9, threshold=1.0)

# Initialize state for batch_size=4, features=10
state = lif.init_state(batch_size=4, features=10)

# One timestep forward pass
x = mx.random.normal((4, 10))
spk, state = lif(x, state)
# spk: binary spike tensor (4, 10)
# state["mem"]: membrane potential (4, 10)
```

### 2. Explicit State

Unlike some frameworks, mlx-snn uses **explicit state dictionaries**. State is always passed in and returned — no hidden global state.

```python
# State is a dict — easy to inspect and manipulate
state = lif.init_state(4, 10)
print(state.keys())  # dict_keys(['mem'])

# After forward pass, state is updated
spk, state = lif(x, state)
print(state["mem"].shape)  # (4, 10)
```

### 3. Spike Encoding

Static data must be encoded into spike trains before feeding to an SNN:

```python
# Rate encoding: higher values → higher spike probability
data = mx.random.uniform(shape=(4, 784))  # batch of 4, 784 features
spikes = mlxsnn.rate_encode(data, num_steps=25)  # (25, 4, 784)

# Direct encoding: repeat data across timesteps (no stochasticity)
spikes = mlxsnn.direct_encode(data, num_steps=25)  # (25, 4, 784)

# Latency encoding: higher values → earlier spikes
spikes = mlxsnn.latency_encode(data, num_steps=25)  # (25, 4, 784)
```

### 4. Surrogate Gradients

Spikes are non-differentiable (Heaviside step function). mlx-snn uses **surrogate gradients** to enable backpropagation:

```python
# Default: fast sigmoid surrogate
lif = mlxsnn.Leaky(beta=0.9, surrogate_fn="fast_sigmoid")

# Alternatives
lif = mlxsnn.Leaky(beta=0.9, surrogate_fn="arctan")
lif = mlxsnn.Leaky(beta=0.9, surrogate_fn="triangular")
```

## Building a Network

Combine standard MLX layers with spiking neurons:

```python
import mlx.nn as nn

class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.lif1 = mlxsnn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(128, 10)
        self.lif2 = mlxsnn.Leaky(beta=0.9)

    def __call__(self, x_seq):
        T, B = x_seq.shape[0], x_seq.shape[1]
        s1 = self.lif1.init_state(B, 128)
        s2 = self.lif2.init_state(B, 10)
        mems = []

        for t in range(T):
            h = self.fc1(x_seq[t])
            spk, s1 = self.lif1(h, s1)
            h = self.fc2(spk)
            spk, s2 = self.lif2(h, s2)
            mems.append(s2["mem"])

        return mx.stack(mems)  # (T, B, 10)
```

## Training

Use standard MLX training patterns with SNN-specific loss functions:

```python
import mlx.optimizers as optim

model = SimpleSNN()
optimizer = optim.Adam(learning_rate=1e-3)

def loss_fn(model, x, y):
    mem_out = model(x)  # (T, B, 10)
    return mlxsnn.mse_membrane_loss(mem_out, y)

loss_and_grad = nn.value_and_grad(model, loss_fn)

# Training step
x = mlxsnn.rate_encode(mx.random.uniform(shape=(32, 784)), num_steps=25)
y = mx.random.randint(0, 10, (32,))

loss, grads = loss_and_grad(model, x, y)
optimizer.update(model, grads)
mx.eval(model.parameters(), optimizer.state)
```

### Available Loss Functions

| Function | Use Case |
|----------|----------|
| `mse_membrane_loss` | MSE on membrane potentials vs one-hot targets |
| `membrane_loss` | Cross-entropy on final membrane potential |
| `ce_rate_loss` | Cross-entropy on spike rates |
| `ce_count_loss` | Cross-entropy on spike counts |
| `rate_coding_loss` | Rate coding classification loss |
| `activity_reg_loss` | Firing rate regularization |

## Convolutional SNNs

For spatiotemporal data (event cameras, video):

```python
# Conv SNN block
conv = mlxsnn.SpikingConv2d(2, 32, kernel_size=3, padding=1)
pool = mlxsnn.SpikingMaxPool2d(kernel_size=2, stride=2)

state = {"mem": mx.zeros((B, 64, 64, 32))}
x = mx.random.normal((B, 128, 128, 2))

spk, state = conv(x, state)
spk = pool(spk)  # (B, 64, 64, 32)
```

## Next Steps

- [API Reference](api/index.md) — Full documentation of all classes and functions
- [Migration from snnTorch](migration.md) — Guide for snnTorch users
- [Examples](examples.md) — Complete runnable examples
