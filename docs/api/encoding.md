# Encoding

Spike encoding methods convert continuous-valued data into binary spike trains for SNN processing.

## Rate Encoding

Poisson rate coding — higher values produce higher spike probability.

::: mlxsnn.encoding.rate.rate_encode

## Latency Encoding

Time-to-first-spike encoding — higher values spike earlier.

::: mlxsnn.encoding.latency.latency_encode

## Delta Encoding

Change detection — spikes when input changes exceed a threshold.

::: mlxsnn.encoding.delta.delta_encode

## Direct Encoding

Repeats static data across timesteps without stochastic conversion.

::: mlxsnn.encoding.direct.direct_encode

## Repeat Encoding

Tiles an existing spike pattern N times along the temporal dimension.

::: mlxsnn.encoding.direct.repeat_encode

## EEG Encoder

Specialized encoder for EEG biomedical signals.

::: mlxsnn.encoding.medical.eeg.EEGEncoder
