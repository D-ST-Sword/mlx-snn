"""Liquid State Machine: reservoir + readout.

Complete LSM system that combines a :class:`LiquidReservoir` with
a trainable readout layer.  The reservoir is frozen by default
(only the readout is trained).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlxsnn.liquid.reservoir import LiquidReservoir


class LSM(nn.Module):
    """Liquid State Machine with reservoir and trainable readout.

    The reservoir provides a high-dimensional, temporally-rich
    representation of the input, and the readout maps reservoir
    states to task outputs.

    Args:
        input_size: Input feature dimension.
        reservoir_size: Number of neurons in the reservoir.
        output_size: Number of output classes/features.
        readout: Readout type ('linear' or 'mlp').
        train_reservoir: If True, reservoir weights are trainable.
            Default False (standard LSM approach).
        **reservoir_kwargs: Additional arguments passed to
            :class:`LiquidReservoir`.

    Examples:
        >>> lsm = LSM(input_size=64, reservoir_size=200, output_size=10)
        >>> state = lsm.init_state(batch_size=8)
        >>> x = mx.random.normal((8, 64))
        >>> out, state = lsm(x, state)
        >>> out.shape
        [8, 10]
    """

    def __init__(
        self,
        input_size: int,
        reservoir_size: int = 500,
        output_size: int = 10,
        readout: str = "linear",
        train_reservoir: bool = False,
        **reservoir_kwargs,
    ):
        super().__init__()
        self.reservoir = LiquidReservoir(
            input_size=input_size,
            reservoir_size=reservoir_size,
            **reservoir_kwargs,
        )

        if readout == "linear":
            self.readout = nn.Linear(reservoir_size, output_size)
        elif readout == "mlp":
            self.readout = nn.Sequential(
                nn.Linear(reservoir_size, reservoir_size // 2),
                nn.ReLU(),
                nn.Linear(reservoir_size // 2, output_size),
            )
        else:
            raise ValueError(f"Unknown readout type: {readout}")

        if not train_reservoir:
            self.reservoir.freeze()

    def init_state(self, batch_size: int) -> dict:
        """Initialize LSM state.

        Args:
            batch_size: Number of samples.

        Returns:
            Reservoir state dictionary.
        """
        return self.reservoir.init_state(batch_size)

    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        """Process one timestep.

        Args:
            x: Input of shape ``[batch, input_size]``.
            state: Reservoir state from previous timestep.

        Returns:
            Tuple of ``(output, new_state)`` where output has shape
            ``[batch, output_size]``.
        """
        spk, state = self.reservoir(x, state)
        out = self.readout(spk)
        return out, state

    def forward_sequence(
        self,
        x_seq: mx.array,
        state: dict | None = None,
    ) -> tuple[mx.array, dict]:
        """Process a full input sequence.

        Args:
            x_seq: Input sequence of shape ``[T, batch, input_size]``.
            state: Initial state. If None, initialized internally.

        Returns:
            Tuple of ``(outputs, final_state)`` where outputs has
            shape ``[T, batch, output_size]``.
        """
        T, B = x_seq.shape[0], x_seq.shape[1]
        if state is None:
            state = self.init_state(B)

        outputs = []
        for t in range(T):
            out, state = self(x_seq[t], state)
            outputs.append(out)

        return mx.stack(outputs), state
