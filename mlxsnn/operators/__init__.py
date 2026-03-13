"""Mathematically-principled Conv SNN operators.

Five operators that exploit specific mathematical properties of spiking
neural networks to improve efficiency and/or expressiveness:

- **TemporalAggregatedConv (TAC)**: Exploits linearity of convolution
  to reduce T conv calls to T/K by aggregating K timesteps.
- **TACTemporalPreserve (TAC-TP)**: Like TAC but preserves full temporal
  dimension T by running LIF dynamics K times per chunk with shared conv output.
- **FourierTemporalConv (FTC)**: Replaces fixed LIF dynamics with
  learnable biquad IIR temporal filters per channel.
- **InfoMaxSpikeConv (IMC)**: Information-theoretic channel gating
  with entropy-based regularization.
- **TemporalCollapseConv (TCC)**: Collapses consecutive no-spike
  timesteps into single conv calls using cached spike patterns.
"""

from mlxsnn.operators.temporal_aggregated_conv import TemporalAggregatedConv
from mlxsnn.operators.tac_temporal_preserve import TACTemporalPreserve
from mlxsnn.operators.learnable_tac import LearnableTAC
from mlxsnn.operators.fourier_temporal_conv import FourierTemporalConv
from mlxsnn.operators.infomax_spike_conv import InfoMaxSpikeConv
from mlxsnn.operators.temporal_collapse_conv import TemporalCollapseConv
