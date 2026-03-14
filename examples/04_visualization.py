"""Visualization: spike raster plots, membrane traces, and firing rates.

Demonstrates mlx-snn's built-in visualization utilities by running a
small LIF network and plotting the internal dynamics.

Usage:
    python examples/04_visualization.py

Requirements:
    pip install mlx-snn matplotlib
"""

import mlx.core as mx
import mlx.nn as nn

import mlxsnn
from mlxsnn.utils.visualization import plot_raster, plot_membrane, plot_firing_rate


# ---------------------------------------------------------------------------
# Build a small SNN
# ---------------------------------------------------------------------------

class SmallSNN(nn.Module):
    def __init__(self, input_size=20, hidden_size=50, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = mlxsnn.Leaky(beta=0.85)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = mlxsnn.Leaky(beta=0.90)

    def forward(self, x_seq):
        """Run forward pass, collecting spikes and membrane potentials."""
        T, B, _ = x_seq.shape
        state1 = self.lif1.init_state(B, 50)
        state2 = self.lif2.init_state(B, 10)

        spk1_list, mem1_list = [], []
        spk2_list, mem2_list = [], []

        for t in range(T):
            h = self.fc1(x_seq[t])
            spk1, state1 = self.lif1(h, state1)
            spk1_list.append(spk1)
            mem1_list.append(state1["mem"])

            h = self.fc2(spk1)
            spk2, state2 = self.lif2(h, state2)
            spk2_list.append(spk2)
            mem2_list.append(state2["mem"])

        return {
            "spk1": mx.stack(spk1_list),   # (T, B, 50)
            "mem1": mx.stack(mem1_list),
            "spk2": mx.stack(spk2_list),   # (T, B, 10)
            "mem2": mx.stack(mem2_list),
        }


# ---------------------------------------------------------------------------
# Run and visualize
# ---------------------------------------------------------------------------

def main():
    T, B = 50, 1  # 50 timesteps, 1 sample (for visualization clarity)
    model = SmallSNN()

    # Generate random input spike trains
    x_seq = mlxsnn.rate_encode(mx.random.uniform(shape=(B, 20)) * 0.8, num_steps=T)
    mx.eval(x_seq)

    # Forward pass
    out = model.forward(x_seq)
    for v in out.values():
        mx.eval(v)

    # --- Plot 1: Spike raster of hidden layer (first 20 neurons) ---
    print("Plotting spike raster (hidden layer)...")
    ax = plot_raster(
        out["spk1"][:, 0, :20],  # (T, 20)
        title="Hidden Layer Spike Raster (20 neurons)",
        xlabel="Time Step",
        ylabel="Neuron Index",
    )
    ax.figure.savefig("raster_hidden.png", dpi=150)
    print("  Saved raster_hidden.png")

    # --- Plot 2: Membrane potential traces of output layer ---
    print("Plotting membrane traces (output layer)...")
    ax = plot_membrane(
        out["mem2"][:, 0, :],    # (T, 10)
        title="Output Layer Membrane Potential",
        threshold=1.0,
    )
    ax.figure.savefig("membrane_output.png", dpi=150)
    print("  Saved membrane_output.png")

    # --- Plot 3: Firing rate bar chart ---
    print("Plotting firing rates (output layer)...")
    ax = plot_firing_rate(
        out["spk2"][:, 0, :],    # (T, 10)
        title="Output Layer Firing Rates",
    )
    ax.figure.savefig("firing_rate_output.png", dpi=150)
    print("  Saved firing_rate_output.png")

    print("\nDone! Three visualization files saved.")


if __name__ == "__main__":
    main()
