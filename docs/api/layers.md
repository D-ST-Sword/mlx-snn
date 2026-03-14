# Layers

Composite spiking layers that combine standard operations (convolution, pooling) with spiking neuron dynamics.

## SpikingConv2d

Convolutional layer with integrated LIF neuron — performs `Conv2d → LIF` in a single module.

::: mlxsnn.layers.conv.SpikingConv2d

## SpikingMaxPool2d

Max pooling that preserves spike semantics.

::: mlxsnn.layers.pooling.SpikingMaxPool2d

## SpikingAvgPool2d

Average pooling for spike feature maps.

::: mlxsnn.layers.pooling.SpikingAvgPool2d

## SpikingFlatten

Flatten spatial dimensions for transition from conv to FC layers.

::: mlxsnn.layers.flatten.SpikingFlatten

## SpikeDropout

Dropout specialized for binary spike trains. Drops spikes with probability `p` during training (no rescaling, since spikes are binary).

::: mlxsnn.layers.dropout.SpikeDropout

## Neuron Factory

::: mlxsnn.layers._factory.create_neuron
