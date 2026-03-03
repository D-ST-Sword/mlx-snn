"""Benchmark: Neuron dynamics visualization for all 6 models.

Runs each neuron model for 200 timesteps with constant input current
and plots membrane traces + spike rasters.

Output:
    paper/figures/neuron_dynamics.pdf — 6-panel figure
"""

import sys
sys.path.insert(0, ".")

import mlx.core as mx
import numpy as np

from mlxsnn.neurons import Leaky, IF, Izhikevich, ALIF, Synaptic, Alpha


def simulate_neuron(neuron, x_input, num_steps, features=1):
    """Simulate a neuron for num_steps with given input.

    Returns:
        mem_trace: list of membrane potentials (scalar per step)
        spk_trace: list of spike values (0 or 1 per step)
    """
    state = neuron.init_state(batch_size=1, features=features)
    mem_trace = []
    spk_trace = []

    for t in range(num_steps):
        x = mx.array([[x_input]])
        spk, state = neuron(x, state)
        mx.eval(spk, *state.values())

        # Extract membrane potential (use 'v' for Izhikevich, 'mem' for others)
        if "v" in state:
            mem_val = state["v"][0, 0].item()
        else:
            mem_val = state["mem"][0, 0].item()
        spk_val = spk[0, 0].item()

        mem_trace.append(mem_val)
        spk_trace.append(spk_val)

    return mem_trace, spk_trace


def main():
    num_steps = 200

    # Define neurons and their input currents
    configs = [
        ("LIF (Leaky)", Leaky(beta=0.9, threshold=1.0), 0.3),
        ("IF", IF(threshold=1.0), 0.15),
        ("Izhikevich (RS)", Izhikevich(preset="RS"), 14.0),
        ("ALIF", ALIF(beta=0.9, rho=0.95, b=0.1, threshold=1.0), 0.3),
        ("Synaptic", Synaptic(alpha=0.8, beta=0.9, threshold=1.0), 0.08),
        ("Alpha", Alpha(alpha=0.85, beta=0.9, threshold=1.0), 0.008),
    ]

    results = {}
    for name, neuron, x_input in configs:
        print(f"Simulating {name}...")
        mem, spk = simulate_neuron(neuron, x_input, num_steps)
        results[name] = {"mem": mem, "spk": spk}
        n_spikes = sum(1 for s in spk if s > 0.5)
        print(f"  {n_spikes} spikes in {num_steps} steps")

    # Save raw data as npz for reproducibility
    np.savez(
        "paper/figures/neuron_dynamics_data.npz",
        **{f"{name}_mem": np.array(results[name]["mem"]) for name in results},
        **{f"{name}_spk": np.array(results[name]["spk"]) for name in results},
    )

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        axes = axes.flatten()
        t = np.arange(num_steps)

        for i, (name, neuron, x_input) in enumerate(configs):
            ax = axes[i]
            mem = np.array(results[name]["mem"])
            spk = np.array(results[name]["spk"])

            # Membrane trace
            ax.plot(t, mem, "b-", linewidth=0.8, label="Membrane")

            # Spike raster (vertical lines)
            spike_times = t[spk > 0.5]
            if len(spike_times) > 0:
                for st in spike_times:
                    ax.axvline(x=st, color="r", alpha=0.4, linewidth=0.5)

            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Membrane potential")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("paper/figures/neuron_dynamics.pdf", dpi=150, bbox_inches="tight")
        print("\nSaved: paper/figures/neuron_dynamics.pdf")

    except ImportError:
        print("matplotlib not installed. Data saved to .npz only.")


if __name__ == "__main__":
    main()
