"""Custom Neuron: subclassing SpikingNeuron to create new models.

Demonstrates how to build a custom spiking neuron by subclassing
mlxsnn.neurons.base.SpikingNeuron. We implement an Exponential
Integrate-and-Fire (EIF) neuron and train it on a simple task.

Usage:
    python examples/06_custom_neuron.py

Requirements:
    pip install mlx-snn
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlxsnn.neurons.base import SpikingNeuron


# ---------------------------------------------------------------------------
# Custom Neuron: Exponential Integrate-and-Fire (EIF)
# ---------------------------------------------------------------------------

class EIF(SpikingNeuron):
    """Exponential Integrate-and-Fire neuron.

    Membrane dynamics (discrete):
        U[t+1] = beta * U[t] + delta_T * exp((U[t] - V_T) / delta_T) + X[t+1]

    The exponential term creates a sharp upstroke near threshold,
    modeling the sodium current activation in biological neurons.

    Args:
        beta: Membrane decay factor (0 < beta < 1).
        delta_T: Sharpness of exponential (larger = sharper).
        V_T: Soft threshold for exponential onset.
        threshold: Hard spike threshold.
    """

    def __init__(self, beta=0.9, delta_T=0.5, V_T=0.8, threshold=1.0,
                 reset_mechanism="subtract", surrogate_fn="fast_sigmoid",
                 surrogate_scale=25.0):
        super().__init__(
            threshold=threshold,
            reset_mechanism=reset_mechanism,
            surrogate_fn=surrogate_fn,
            surrogate_scale=surrogate_scale,
        )
        self._beta = beta
        self._delta_T = delta_T
        self._V_T = V_T

    def init_state(self, batch_size, features):
        """Initialize membrane potential to zero."""
        return {"mem": mx.zeros((batch_size, features))}

    def __call__(self, x, state):
        """Forward pass with exponential dynamics."""
        mem = state["mem"]
        beta = self._beta
        delta_T = self._delta_T
        V_T = self._V_T

        # Exponential term (clamped to prevent overflow)
        exp_term = delta_T * mx.exp(mx.clip((mem - V_T) / delta_T, a_min=-10, a_max=5))

        # Membrane update
        mem = beta * mem + exp_term + x

        # Spike generation (via surrogate gradient)
        spk = self.fire(mem)

        # Reset
        mem = self.reset(mem, spk)

        return spk, {"mem": mem}


# ---------------------------------------------------------------------------
# Simple training demo
# ---------------------------------------------------------------------------

class EIFNetwork(nn.Module):
    """Two-layer network using our custom EIF neuron."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.eif1 = EIF(beta=0.85, delta_T=0.3)
        self.fc2 = nn.Linear(32, 4)
        self.eif2 = EIF(beta=0.90, delta_T=0.5)

    def __call__(self, x_seq):
        T = x_seq.shape[0]
        B = x_seq.shape[1]
        state1 = self.eif1.init_state(B, 32)
        state2 = self.eif2.init_state(B, 4)
        mems = []

        for t in range(T):
            h = self.fc1(x_seq[t])
            spk, state1 = self.eif1(h, state1)
            h = self.fc2(spk)
            spk, state2 = self.eif2(h, state2)
            mems.append(state2["mem"])

        return mx.stack(mems)  # (T, B, 4)


def main():
    print("Custom Neuron Demo: Exponential Integrate-and-Fire (EIF)")
    print("=" * 55)

    model = EIFNetwork()
    optimizer = optim.Adam(learning_rate=1e-3)

    # Synthetic classification task: 4 classes, 8 features, 10 timesteps
    T, B, num_classes = 10, 32, 4

    def loss_fn(model, x, y):
        mem_out = model(x)
        # Use final membrane potential for classification
        logits = mem_out[-1]  # (B, 4)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(20):
        # Generate random data each epoch
        x = mx.random.normal((T, B, 8)) * 0.5
        y = mx.random.randint(0, num_classes, (B,))

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}  loss={loss.item():.4f}")

    # Verify custom neuron properties
    eif = EIF(beta=0.85, delta_T=0.3)
    state = eif.init_state(1, 4)
    print(f"\nEIF state keys: {list(state.keys())}")
    print(f"EIF threshold:  {eif._get_threshold()}")
    print(f"EIF reset:      {eif.reset_mechanism}")
    print(f"\nCustom neuron works correctly!")


if __name__ == "__main__":
    main()
