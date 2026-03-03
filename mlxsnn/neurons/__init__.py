"""Spiking neuron models.

All neuron models accept input current and previous state, returning
output spikes and updated state. State is represented as an explicit
dictionary for compatibility with MLX functional transforms.
"""

from mlxsnn.neurons.adaptive_lif import ALIF
from mlxsnn.neurons.alpha import Alpha
from mlxsnn.neurons.base import SpikingNeuron
from mlxsnn.neurons.if_neuron import IF
from mlxsnn.neurons.izhikevich import Izhikevich
from mlxsnn.neurons.lif import Leaky
from mlxsnn.neurons.synaptic import Synaptic
