from __future__ import annotations

"""Backpropagation Through Time (BPTT) helpers for SNN training.

Provides utilities to run a model over multiple timesteps and collect
spike outputs and membrane traces for loss computation.
"""

import mlx.core as mx


def bptt_forward(
    model,
    spikes: mx.array,
    state: dict,
    num_steps: int | None = None,
) -> tuple[mx.array, mx.array, dict]:
    """Run BPTT forward pass over time, collecting all outputs.

    Iterates the model over the time dimension, collecting output
    spikes and membrane potentials at each step.

    Args:
        model: An SNN model (callable with signature
            ``(x, state) -> (spk, new_state)``).
        spikes: Input spike tensor ``[T, batch, ...]`` (time-first).
        state: Initial hidden state dict.
        num_steps: Override time steps. If None, uses ``spikes.shape[0]``.

    Returns:
        all_spikes: Output spikes ``[T, batch, ...]``.
        all_mems: Membrane potentials ``[T, batch, ...]``.
        final_state: Final hidden state dict.

    Examples:
        >>> from mlxsnn.training import bptt_forward
        >>> all_spk, all_mem, final_state = bptt_forward(
        ...     model, input_spikes, init_state
        ... )
    """
    T = num_steps or spikes.shape[0]
    out_spikes = []
    out_mems = []

    for t in range(T):
        spk, state = model(spikes[t], state)
        out_spikes.append(spk)
        if "mem" in state:
            out_mems.append(state["mem"])

    all_spikes = mx.stack(out_spikes)
    all_mems = mx.stack(out_mems) if out_mems else mx.array(0.0)
    return all_spikes, all_mems, state
