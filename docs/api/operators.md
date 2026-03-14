# Operators

Mathematically-principled operators that optimize convolutional SNN processing. These operators exploit temporal structure to reduce computation while preserving gradient flow.

## Temporal Aggregated Convolution (TAC)

Exploits the linearity of convolution to aggregate K timesteps into a single conv call, achieving near-linear speedup.

::: mlxsnn.operators.temporal_aggregated_conv.TemporalAggregatedConv

## TAC with Temporal Preservation (TAC-TP)

TAC variant that preserves the full temporal dimension in the output.

::: mlxsnn.operators.tac_temporal_preserve.TACTemporalPreserve

## Learnable TAC (L-TAC)

TAC with learnable aggregation weights instead of uniform averaging.

::: mlxsnn.operators.learnable_tac.LearnableTAC

## Fourier Temporal Convolution (FTC)

Learnable biquad IIR filters per channel for temporal processing.

::: mlxsnn.operators.fourier_temporal_conv.FourierTemporalConv

## Information-Max Spike Convolution (IMC)

Information-theoretic channel gating that selectively activates channels based on input information content.

::: mlxsnn.operators.infomax_spike_conv.InfoMaxSpikeConv

## Temporal Collapse Convolution (TCC)

Sparsity-aware operator that collapses consecutive silent timesteps.

::: mlxsnn.operators.temporal_collapse_conv.TemporalCollapseConv
