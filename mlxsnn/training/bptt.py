from __future__ import annotations

"""Backpropagation Through Time (BPTT) helpers for SNN training.

Provides utilities to run a model over multiple timesteps and collect
spike outputs and membrane traces for loss computation.

Includes :func:`detach_state` and :func:`chunked_bptt_forward` for
truncated BPTT, which reduces memory from O(T) to O(chunk_size) by
periodically cutting the gradient graph.
"""

import mlx.core as mx


def detach_state(state: dict) -> dict:
    """Detach all arrays in a state dict from the computation graph.

    Applies :func:`mx.stop_gradient` to every ``mx.array`` in *state*
    (including nested dicts) and forces evaluation so that the gradient
    graph accumulated during prior timesteps can be freed.

    This is the key primitive for truncated BPTT: call it at chunk
    boundaries to bound peak memory usage.

    Args:
        state: Neuron hidden state dict (may be nested).

    Returns:
        A new dict with the same structure where every array has been
        detached from the gradient tape and materialized.

    Examples:
        >>> state = {"mem": mx.array([1.0, 2.0])}
        >>> detached = detach_state(state)
    """
    detached = {}
    arrays_to_eval = []
    for k, v in state.items():
        if isinstance(v, mx.array):
            d = mx.stop_gradient(v)
            detached[k] = d
            arrays_to_eval.append(d)
        elif isinstance(v, dict):
            detached[k] = detach_state(v)
        else:
            detached[k] = v
    if arrays_to_eval:
        mx.eval(*arrays_to_eval)
    return detached


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


def chunked_bptt_forward(
    model,
    spikes: mx.array,
    state: dict,
    chunk_size: int = 16,
    num_steps: int | None = None,
) -> tuple[mx.array, mx.array, dict]:
    """Chunked BPTT: truncate the gradient graph every *chunk_size* steps.

    This is functionally equivalent to :func:`bptt_forward` in the
    forward direction, but it calls :func:`detach_state` between
    chunks so that the gradient tape never grows beyond *chunk_size*
    steps.  Peak memory drops from ``O(T * model_size)`` to
    ``O(chunk_size * model_size)``.

    Args:
        model: An SNN model (callable with signature
            ``(x, state) -> (spk, new_state)``).
        spikes: Input spike tensor ``[T, batch, ...]`` (time-first).
        state: Initial hidden state dict.
        chunk_size: Number of timesteps per gradient chunk. Gradients
            are truncated at chunk boundaries.  Smaller values save
            more memory but may reduce learning quality for long-range
            temporal dependencies.
        num_steps: Override time steps. If None, uses ``spikes.shape[0]``.

    Returns:
        all_spikes: Output spikes ``[T, batch, ...]``.
        all_mems: Membrane potentials ``[T, batch, ...]``.
        final_state: Final hidden state dict.

    Examples:
        >>> from mlxsnn.training import chunked_bptt_forward
        >>> # Memory-efficient BPTT with 16-step chunks
        >>> all_spk, all_mem, state = chunked_bptt_forward(
        ...     model, input_spikes, init_state, chunk_size=16
        ... )
    """
    T = num_steps or spikes.shape[0]
    all_spikes = []
    all_mems = []

    for chunk_start in range(0, T, chunk_size):
        chunk_end = min(chunk_start + chunk_size, T)
        for t in range(chunk_start, chunk_end):
            spk, state = model(spikes[t], state)
            all_spikes.append(spk)
            if "mem" in state:
                all_mems.append(state["mem"])
        # Detach at chunk boundary (skip after the last chunk)
        if chunk_end < T:
            state = detach_state(state)

    return (
        mx.stack(all_spikes),
        mx.stack(all_mems) if all_mems else mx.array(0.0),
        state,
    )
