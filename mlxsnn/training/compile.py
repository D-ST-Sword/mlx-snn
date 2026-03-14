"""``mx.compile`` optimization utilities for SNN training.

Provides helpers to compile per-timestep computations for faster
execution. The temporal loop itself remains in Python, but each
timestep's forward pass is compiled into a fused graph.

Note:
    ``mx.compile`` requires pure functions with no Python side effects
    or control flow that depends on array values.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def compiled_step(model: nn.Module) -> callable:
    """Create a compiled single-timestep forward function.

    Wraps the model's ``__call__`` with ``mx.compile`` so that each
    timestep's forward pass is optimized as a fused computation graph.

    Args:
        model: An SNN model or neuron with signature
            ``(x, state) -> (spk, new_state)``.

    Returns:
        A compiled callable with the same signature as the model.

    Examples:
        >>> lif = mlxsnn.Leaky(beta=0.9)
        >>> step_fn = compiled_step(lif)
        >>> state = lif.init_state(8, 128)
        >>> x = mx.random.normal((8, 128))
        >>> spk, state = step_fn(x, state)
    """
    @mx.compile
    def step(x, state):
        return model(x, state)
    return step


def compiled_forward(
    model: nn.Module,
    spikes: mx.array,
    state: dict,
    num_steps: int | None = None,
) -> tuple[mx.array, mx.array, dict]:
    """BPTT forward pass with per-timestep compilation.

    Like :func:`~mlxsnn.training.bptt.bptt_forward` but uses
    ``mx.compile`` to optimize each timestep's computation.

    Args:
        model: An SNN model with ``(x, state) -> (spk, new_state)``.
        spikes: Input spike tensor ``[T, batch, ...]`` (time-first).
        state: Initial hidden state dict.
        num_steps: Override time steps. If None, uses ``spikes.shape[0]``.

    Returns:
        all_spikes: Output spikes ``[T, batch, ...]``.
        all_mems: Membrane potentials ``[T, batch, ...]``.
        final_state: Final hidden state dict.

    Note:
        Compilation is most beneficial for models with expensive
        per-timestep computation (e.g., conv layers). For simple
        FC networks the overhead may not be worth it.
    """
    T = num_steps or spikes.shape[0]
    step_fn = compiled_step(model)

    out_spikes = []
    out_mems = []

    for t in range(T):
        spk, state = step_fn(spikes[t], state)
        out_spikes.append(spk)
        if "mem" in state:
            out_mems.append(state["mem"])

    all_spikes = mx.stack(out_spikes)
    all_mems = mx.stack(out_mems) if out_mems else mx.array(0.0)
    return all_spikes, all_mems, state
