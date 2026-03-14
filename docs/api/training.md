# Training

Training utilities for SNN models, including BPTT forward pass helpers.

## BPTT Forward Pass

::: mlxsnn.training.bptt.bptt_forward

## Training Pattern

The standard training pattern with mlx-snn:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlxsnn

model = MySNN()
optimizer = optim.Adam(learning_rate=1e-3)

def loss_fn(model, x_seq, y):
    mem_out = model(x_seq)
    return mlxsnn.mse_membrane_loss(mem_out, y)

loss_and_grad = nn.value_and_grad(model, loss_fn)

for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        x_seq = mx.array(x_batch).transpose(1, 0, 2)  # to time-first
        y = mx.array(y_batch)

        loss, grads = loss_and_grad(model, x_seq, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
```
