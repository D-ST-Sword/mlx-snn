"""Composite spiking layers for building convolutional SNNs.

Provides convolutional, pooling, and flatten layers that integrate
spiking neuron dynamics with standard spatial operations.  All spatial
layers use MLX's NHWC format: ``(B, H, W, C)``.
"""

from mlxsnn.layers._factory import create_neuron
from mlxsnn.layers.conv import SpikingConv2d
from mlxsnn.layers.flatten import SpikingFlatten
from mlxsnn.layers.pooling import SpikingAvgPool2d, SpikingMaxPool2d
