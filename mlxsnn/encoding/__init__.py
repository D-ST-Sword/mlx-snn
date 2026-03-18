"""Spike encoding methods.

Convert continuous-valued signals into spike trains for processing
by spiking neural networks.
"""

from mlxsnn.encoding.delta import delta_encode
from mlxsnn.encoding.direct import direct_encode, repeat_encode
from mlxsnn.encoding.frequency_band import frequency_band_encode
from mlxsnn.encoding.latency import latency_encode
from mlxsnn.encoding.medical import EEGEncoder
from mlxsnn.encoding.rate import rate_encode
from mlxsnn.encoding.threshold_crossing import threshold_crossing_encode
