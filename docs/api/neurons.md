# Neurons

Spiking neuron models are the core building blocks of SNNs. All neurons inherit from `SpikingNeuron` and follow the same interface: `(x, state) → (spk, new_state)`.

## Base Class

::: mlxsnn.neurons.base.SpikingNeuron
    options:
      members:
        - __init__
        - init_state
        - __call__
        - fire
        - reset

## Leaky Integrate-and-Fire (LIF)

The most commonly used spiking neuron. Membrane dynamics:

$$U[t+1] = \beta \cdot U[t] + X[t+1] - S[t] \cdot V_{thr}$$

::: mlxsnn.neurons.lif.Leaky

## Integrate-and-Fire (IF)

Non-leaky variant — no membrane decay ($\beta = 1$).

::: mlxsnn.neurons.if_neuron.IF

## Izhikevich

2D dynamical system with biologically realistic spiking patterns.

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$$
$$\frac{du}{dt} = a(bv - u)$$

Supports presets: Regular Spiking (RS), Intrinsically Bursting (IB), Chattering (CH), Fast Spiking (FS).

::: mlxsnn.neurons.izhikevich.Izhikevich

## Adaptive LIF (ALIF)

LIF with adaptive threshold that increases after each spike.

::: mlxsnn.neurons.adaptive_lif.ALIF

## Synaptic

Conductance-based LIF with dual-state dynamics (synaptic current + membrane potential).

::: mlxsnn.neurons.synaptic.Synaptic

## Alpha

Dual-exponential synaptic model with alpha-function shaped PSPs.

::: mlxsnn.neurons.alpha.Alpha

## Recurrent LIF (RLeaky)

LIF with learnable recurrent feedback weight.

::: mlxsnn.neurons.rleaky.RLeaky

## Recurrent Synaptic (RSynaptic)

Synaptic neuron with learnable recurrent feedback weight.

::: mlxsnn.neurons.rsynaptic.RSynaptic
