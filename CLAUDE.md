# MLX-SNN Development Prompt for Claude Code

> 将此文件内容作为 Claude Code 的 CLAUDE.md 或项目级系统 prompt 使用。

---

## Role & Context

You are an expert systems-level ML framework engineer helping build **mlx-snn** — the first Spiking Neural Network (SNN) library built natively on Apple's MLX framework. You are working with a researcher (Qie) who holds a PhD in Mathematical Sciences with expertise in multimodal deep learning, cross-modal representation learning, and medical AI (EEG-fMRI, photoacoustic imaging, epilepsy prediction).

**Development environment:**
- Hardware: M3 Max MacBook Pro (Apple Silicon, unified memory)
- Framework: Apple MLX (Python API)
- Language: Python 3.11+
- Package manager: uv (preferred) or pip
- Target audience: MLX/SNN research community (open-source)

## Project Vision

mlx-snn aims to be the **snnTorch of the MLX ecosystem** — a lightweight, research-friendly SNN library that leverages Apple Silicon's unified memory architecture. It should feel familiar to snnTorch/Norse users while offering MLX-native advantages (lazy evaluation, unified memory, composable function transforms).

## Architecture Principles

1. **MLX-native**: All tensor operations use `mlx.core`. Never fall back to NumPy/PyTorch for computation. NumPy is acceptable only for data I/O and preprocessing.
2. **Functional + OOP hybrid**: Provide both `mlx.nn.Module`-based neuron layers (for model building) and pure functional APIs (for advanced users and custom training loops).
3. **snnTorch-compatible API surface**: Users migrating from snnTorch should find familiar class names and method signatures. Prioritize API familiarity over internal implementation similarity.
4. **Minimal dependencies**: Core library depends only on `mlx` and Python stdlib. Optional dependencies (matplotlib, h5py, mne) for visualization and medical data I/O.
5. **Research-first design**: Favor flexibility and hackability over production optimization. Every component should be easy to subclass, override, or compose.

## Package Structure

```
mlx-snn/
├── mlxsnn/
│   ├── __init__.py              # Public API exports
│   ├── neurons/                 # Neuron models
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base neuron class
│   │   ├── lif.py               # Leaky Integrate-and-Fire
│   │   ├── if_neuron.py         # Integrate-and-Fire (non-leaky)
│   │   ├── izhikevich.py        # Izhikevich neuron model
│   │   ├── adaptive_lif.py      # Adaptive LIF (ALIF)
│   │   ├── synaptic.py          # Synaptic conductance-based LIF
│   │   └── alpha.py             # Alpha neuron (dual-exponential)
│   ├── surrogate/               # Surrogate gradient functions
│   │   ├── __init__.py
│   │   ├── fast_sigmoid.py
│   │   ├── arctan.py
│   │   ├── straight_through.py
│   │   └── custom.py            # User-definable surrogate gradients
│   ├── encoding/                # Spike encoding methods
│   │   ├── __init__.py
│   │   ├── rate.py              # Rate coding (Poisson)
│   │   ├── latency.py           # Latency / time-to-first-spike
│   │   ├── delta.py             # Delta modulation
│   │   ├── temporal.py          # Temporal contrast encoding
│   │   └── medical/             # Medical signal specific encoders
│   │       ├── __init__.py
│   │       ├── eeg.py           # EEG-to-spike encoding
│   │       ├── fmri.py          # fMRI BOLD signal encoding
│   │       └── biosignal.py     # Generic biosignal encoder
│   ├── liquid/                  # Liquid State Machine module
│   │   ├── __init__.py
│   │   ├── reservoir.py         # LSM reservoir with random connectivity
│   │   ├── readout.py           # Readout layers (linear, SVM, etc.)
│   │   └── topology.py          # Reservoir topology generators
│   ├── functional/              # Stateless functional API
│   │   ├── __init__.py
│   │   ├── neuron_dynamics.py   # Pure functions for membrane updates
│   │   ├── spike_ops.py         # Spike generation and reset operations
│   │   └── loss.py              # SNN-specific loss functions
│   ├── training/                # Training utilities
│   │   ├── __init__.py
│   │   ├── bptt.py              # Backpropagation through time helpers
│   │   ├── regularization.py    # Activity regularization, firing rate constraints
│   │   └── schedulers.py        # Threshold and learning rate schedulers
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── visualization.py     # Raster plots, membrane traces
│   │   ├── metrics.py           # Accuracy, firing rate, ISI statistics
│   │   └── data.py              # Dataset helpers (neuromorphic formats)
│   └── datasets/                # Built-in dataset loaders
│       ├── __init__.py
│       ├── nmnist.py            # Neuromorphic MNIST
│       ├── shd.py               # Spiking Heidelberg Digits
│       └── dvs_gesture.py       # DVS Gesture dataset
├── examples/
│   ├── 01_quickstart.py         # Minimal LIF network on MNIST
│   ├── 02_custom_neuron.py      # Building custom neuron models
│   ├── 03_liquid_state.py       # LSM for temporal classification
│   ├── 04_eeg_classification.py # EEG spike encoding + SNN
│   ├── 05_fmri_decoding.py      # fMRI BOLD to spike + LSM
│   └── 06_dvs_gesture.py        # Event camera data processing
├── tests/
│   ├── test_neurons.py
│   ├── test_surrogate.py
│   ├── test_encoding.py
│   ├── test_liquid.py
│   ├── test_training.py
│   └── test_numerical.py        # Numerical consistency with snnTorch
├── benchmarks/
│   ├── bench_forward.py         # Forward pass throughput
│   ├── bench_training.py        # Training loop benchmarks
│   └── bench_memory.py          # Memory usage profiling
├── docs/
│   ├── index.md
│   ├── getting_started.md
│   ├── api_reference.md
│   ├── migration_from_snntorch.md
│   └── medical_signal_guide.md
├── pyproject.toml
├── README.md
├── LICENSE                      # MIT License
├── CITATION.cff                 # Citation metadata
└── CONTRIBUTING.md
```

## Core Implementation Specifications

### 1. Neuron Models

#### Base Neuron Class

```python
import mlx.core as mx
import mlx.nn as nn

class SpikingNeuron(nn.Module):
    """Abstract base class for all spiking neurons."""
    
    def __init__(self, threshold: float = 1.0, reset_mechanism: str = "subtract",
                 surrogate_fn: str = "fast_sigmoid", surrogate_scale: float = 25.0):
        super().__init__()
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism  # "subtract", "zero", "none"
        self.surrogate_fn = get_surrogate(surrogate_fn, surrogate_scale)
    
    def init_state(self, batch_size: int, *args) -> dict:
        """Initialize neuron hidden state. Must be overridden."""
        raise NotImplementedError
    
    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        """Forward pass: returns (spikes, new_state)."""
        raise NotImplementedError
    
    def fire(self, mem: mx.array) -> mx.array:
        """Generate spikes using surrogate gradient."""
        return self.surrogate_fn(mem - self.threshold)
    
    def reset(self, mem: mx.array, spk: mx.array) -> mx.array:
        """Apply reset mechanism after spike."""
        if self.reset_mechanism == "subtract":
            return mem - spk * self.threshold
        elif self.reset_mechanism == "zero":
            return mem * (1.0 - spk)
        return mem  # "none"
```

**Key design decisions:**
- State is an explicit `dict`, not hidden instance variables — this plays well with MLX's functional transforms and `mx.compile`.
- `surrogate_fn` is composable — users can pass custom callables.
- `fire()` and `reset()` are separate methods for easy overriding.

#### LIF Neuron

```python
class Leaky(SpikingNeuron):
    """Leaky Integrate-and-Fire neuron.
    
    Membrane dynamics:
        U[t+1] = beta * U[t] + X[t+1] - S[t] * threshold
    
    Compatible with snnTorch.Leaky API.
    """
    
    def __init__(self, beta: float = 0.9, learn_beta: bool = False,
                 threshold: float = 1.0, reset_mechanism: str = "subtract",
                 surrogate_fn: str = "fast_sigmoid", **kwargs):
        super().__init__(threshold=threshold, reset_mechanism=reset_mechanism,
                        surrogate_fn=surrogate_fn, **kwargs)
        if learn_beta:
            self.beta = mx.array(beta)  # Learnable parameter
        else:
            self._beta = beta  # Fixed constant
    
    @property
    def beta(self):
        return getattr(self, '_beta', None) or self.beta
    
    def init_state(self, batch_size: int, features: int) -> dict:
        return {"mem": mx.zeros((batch_size, features))}
    
    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        beta = self._beta if hasattr(self, '_beta') else self.beta
        mem = beta * state["mem"] + x
        spk = self.fire(mem)
        mem = self.reset(mem, spk)
        return spk, {"mem": mem}
```

#### Izhikevich Neuron

```python
class Izhikevich(SpikingNeuron):
    """Izhikevich neuron model with 2D dynamics.
    
    dv/dt = 0.04v^2 + 5v + 140 - u + I
    du/dt = a(bv - u)
    
    Supports Regular Spiking (RS), Intrinsically Bursting (IB),
    Chattering (CH), Fast Spiking (FS), and custom parameter sets.
    """
    PRESETS = {
        "RS":  {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
        "IB":  {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
        "CH":  {"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},
        "FS":  {"a": 0.1,  "b": 0.2, "c": -65.0, "d": 2.0},
    }
    # Implementation follows standard Izhikevich equations...
```

### 2. Surrogate Gradient Functions

Implement using MLX's `mx.custom_function` for clean forward/backward separation:

```python
def fast_sigmoid_surrogate(scale: float = 25.0):
    """Fast sigmoid surrogate gradient.
    
    Forward: Heaviside step function
    Backward: gradient of sigmoid(scale * x)
    """
    @mx.custom_function
    def forward(x):
        return mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
    
    @forward.vjp
    def backward(primals, cotangents, outputs):
        x = primals[0]
        grad = cotangents[0]
        sigmoid_grad = scale / (1.0 + mx.abs(scale * x)) ** 2
        return (grad * sigmoid_grad,)
    
    return forward

def arctan_surrogate(alpha: float = 2.0):
    """Arctan surrogate gradient."""
    @mx.custom_function
    def forward(x):
        return mx.where(x >= 0, mx.ones_like(x), mx.zeros_like(x))
    
    @forward.vjp
    def backward(primals, cotangents, outputs):
        x = primals[0]
        grad = cotangents[0]
        arctan_grad = alpha / (2 * (1 + (mx.pi / 2 * alpha * x) ** 2))
        return (grad * arctan_grad,)
    
    return forward
```

### 3. Medical Signal Encoding (EEG/fMRI)

```python
class EEGEncoder:
    """Encode EEG signals into spike trains.
    
    Supports multiple encoding strategies:
    - 'rate': Amplitude-to-firing-rate mapping
    - 'temporal': Phase-based temporal encoding
    - 'delta': Change-based delta modulation
    - 'bsa': Ben's Spiker Algorithm
    - 'threshold_crossing': Multi-threshold level crossing
    
    Designed for standard EEG formats (MNE-Python raw objects,
    numpy arrays with shape [channels, timepoints]).
    """
    
    def __init__(self, method: str = "rate", num_steps: int = 100,
                 channels: int = 64, freq_bands: dict | None = None):
        self.method = method
        self.num_steps = num_steps
        self.channels = channels
        self.freq_bands = freq_bands or {
            "delta": (0.5, 4), "theta": (4, 8),
            "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 100)
        }
    
    def __call__(self, signal: mx.array) -> mx.array:
        """Encode continuous signal to spike train.
        
        Args:
            signal: [channels, timepoints] or [batch, channels, timepoints]
        Returns:
            spikes: [num_steps, batch, channels] (time-first format)
        """
        ...

class fMRIEncoder:
    """Encode fMRI BOLD signals into spike trains.
    
    Handles the unique challenges of fMRI data:
    - Low temporal resolution (TR ~ 0.5-3s)
    - Hemodynamic response function (HRF) deconvolution
    - ROI-based or voxel-wise encoding
    
    Supports both volume-based and surface-based fMRI.
    """
    
    def __init__(self, method: str = "rate", tr: float = 2.0,
                 num_steps_per_tr: int = 10, roi_atlas: str | None = None):
        ...
```

### 4. Liquid State Machine (LSM)

```python
class LiquidReservoir(nn.Module):
    """Liquid State Machine reservoir with random sparse connectivity.
    
    Implements the reservoir computing paradigm with spiking neurons:
    - Random sparse recurrent connectivity (Erdos-Renyi or small-world)
    - Excitatory/inhibitory balance (Dale's law)
    - Input projection layer
    - Separation property and fading memory
    
    Reference: Maass, W., Natschläger, T., & Markram, H. (2002).
    Real-time computing without stable states.
    """
    
    def __init__(self, input_size: int, reservoir_size: int = 1000,
                 connectivity: float = 0.1, spectral_radius: float = 0.9,
                 exc_ratio: float = 0.8, neuron_type: str = "lif",
                 topology: str = "erdos_renyi", **neuron_kwargs):
        super().__init__()
        self.reservoir_size = reservoir_size
        
        # Build connectivity matrix
        self.W_in = self._build_input_weights(input_size, reservoir_size)
        self.W_res = self._build_reservoir_weights(
            reservoir_size, connectivity, spectral_radius, exc_ratio, topology
        )
        
        # Reservoir neurons
        NeuronClass = get_neuron_class(neuron_type)
        self.neurons = NeuronClass(**neuron_kwargs)
    
    def __call__(self, x: mx.array, state: dict) -> tuple[mx.array, dict]:
        """Process one timestep through the reservoir.
        
        Args:
            x: Input spikes [batch, input_size]
            state: {"mem": ..., "spk": ...} reservoir state
        Returns:
            spk: Reservoir spikes [batch, reservoir_size]
            new_state: Updated reservoir state
        """
        # Input current + recurrent current
        I = mx.matmul(x, self.W_in) + mx.matmul(state["spk"], self.W_res)
        spk, neuron_state = self.neurons(I, state)
        return spk, {**neuron_state, "spk": spk}


class LSM(nn.Module):
    """Complete Liquid State Machine: reservoir + readout.
    
    Usage:
        lsm = LSM(input_size=64, reservoir_size=500, output_size=10)
        state = lsm.init_state(batch_size=32)
        
        for t in range(num_steps):
            output, state = lsm(spikes[t], state)
    """
    
    def __init__(self, input_size: int, reservoir_size: int = 1000,
                 output_size: int = 10, readout: str = "linear",
                 train_reservoir: bool = False, **reservoir_kwargs):
        super().__init__()
        self.reservoir = LiquidReservoir(input_size, reservoir_size, **reservoir_kwargs)
        self.readout = self._build_readout(reservoir_size, output_size, readout)
        
        # Optionally freeze reservoir weights
        if not train_reservoir:
            self.reservoir.freeze()
```

### 5. Training Utilities

```python
def bptt_forward(model, spikes, state, num_steps=None):
    """Run BPTT forward pass over time, collecting all outputs.
    
    Args:
        model: SNN model
        spikes: Input spike tensor [T, B, ...] (time-first)
        state: Initial hidden state dict
        num_steps: Override time steps (if spikes is a generator)
    
    Returns:
        all_spikes: Output spikes [T, B, ...]
        all_mems: Membrane potentials [T, B, ...]
        final_state: Final hidden state
    """
    T = num_steps or spikes.shape[0]
    out_spikes, out_mems = [], []
    
    for t in range(T):
        spk, state = model(spikes[t], state)
        out_spikes.append(spk)
        out_mems.append(state.get("mem"))
    
    return mx.stack(out_spikes), mx.stack(out_mems), state


# SNN-specific loss functions
def rate_coding_loss(spk_out: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy on spike counts (rate coding)."""
    spike_count = mx.sum(spk_out, axis=0)  # Sum over time
    log_probs = mx.log(mx.softmax(spike_count, axis=-1) + 1e-8)
    return -mx.mean(mx.sum(log_probs * targets, axis=-1))

def latency_loss(spk_out: mx.array, targets: mx.array) -> mx.array:
    """Reward early spikes for correct class (time-to-first-spike)."""
    ...

def membrane_loss(mem: mx.array, targets: mx.array) -> mx.array:
    """Cross-entropy on final membrane potential."""
    return mx.mean(nn.losses.cross_entropy(mem[-1], targets))
```

## Coding Standards

### Style
- Follow Google Python Style Guide
- Type hints on all public APIs
- Docstrings: NumPy style with `Args`, `Returns`, `Examples`, `References`
- Maximum line length: 100 characters

### Naming Conventions
- Neuron classes: PascalCase matching snnTorch where possible (`Leaky`, `Synaptic`, `Alpha`)
- Functional API: snake_case (`lif_step`, `fire_and_reset`)
- Constants: UPPER_SNAKE_CASE
- Private methods: single underscore prefix

### Testing
- Every neuron model must have:
  - Forward pass shape test
  - Gradient flow test (non-zero gradients through surrogate)
  - Numerical consistency test against reference equations
  - State initialization test
- Use `pytest` with `mlx.core.array` comparisons via `mx.allclose()`
- Benchmark tests should report: throughput (spikes/sec), memory (MB), time per forward/backward

### Documentation
- Every public class and function must have a docstring
- Include mathematical formulation in LaTeX for neuron models
- Provide code examples in docstrings
- Reference original papers with BibTeX keys

## Implementation Priorities (Version Roadmap)

### v0.1.0 — Core Foundation
- [ ] LIF, IF neurons with surrogate gradients (fast_sigmoid, arctan)
- [ ] Rate and latency spike encoding
- [ ] Basic BPTT training loop
- [ ] MNIST example (rate-coded static images)
- [ ] pytest suite for neurons and gradients
- [ ] pyproject.toml with uv support

### v0.2.0 — Extended Neurons + Medical
- [ ] Izhikevich, Adaptive LIF, Synaptic, Alpha neurons
- [ ] EEG spike encoder (rate, delta, threshold crossing)
- [ ] fMRI BOLD encoder
- [ ] Delta modulation and temporal contrast encoding
- [ ] SHD dataset loader

### v0.3.0 — Liquid State Machine
- [ ] LSM reservoir with configurable topology
- [ ] Excitatory/inhibitory balance (Dale's law)
- [ ] Readout layers (linear, ridge regression)
- [ ] LSM + EEG example (epilepsy classification)

### v0.4.0 — Advanced Features
- [ ] `mx.compile` optimized forward passes
- [ ] Neuromorphic dataset loaders (N-MNIST, DVS-Gesture)
- [ ] Raster plot and membrane trace visualization
- [ ] Migration guide from snnTorch
- [ ] Comprehensive benchmarks vs snnTorch on CPU

### v1.0.0 — Publication Ready
- [ ] Full API documentation site
- [ ] JOSS / SoftwareX paper draft
- [ ] CITATION.cff
- [ ] PyPI release
- [ ] Numerical validation against snnTorch reference outputs

## Critical Implementation Notes

### MLX-Specific Gotchas
1. **Lazy evaluation**: Call `mx.eval()` explicitly when you need values materialized (e.g., for printing, logging, or conditional logic based on tensor values).
2. **No in-place operations**: MLX arrays are immutable. Never do `mem += x`; always `mem = mem + x`.
3. **`mx.compile` constraints**: Functions passed to `mx.compile` must be pure — no side effects, no Python control flow dependent on array values.
4. **Random state**: Use `mx.random.key()` and split keys explicitly. Don't rely on global random state.
5. **Device placement**: MLX handles CPU/GPU automatically via unified memory — never manually place tensors.
6. **Custom gradients**: Use `@mx.custom_function` + `.vjp` decorator pattern for surrogate gradients. Do NOT use `stop_gradient` hacks.

### SNN-Specific Gotchas
1. **Detach between timesteps**: For truncated BPTT, use `mx.stop_gradient()` on state at truncation boundaries.
2. **Firing rate collapse**: Add activity regularization to prevent dead neurons (0 spikes) or saturated neurons (spikes every step).
3. **Numerical stability**: Membrane potential can explode without proper reset. Always clip or normalize if using "none" reset.
4. **Time dimension convention**: Always use **time-first** format `[T, B, ...]` — this matches neuroscience conventions and enables efficient sequential processing.

### Performance Tips for M3 Max
- Batch sizes of 64-256 work well for the M3 Max's GPU
- Use `mx.compile` for the inner training loop once correctness is verified
- For reservoir computing (LSM), sparse matrix operations may be slower than dense on MLX — benchmark both
- Monitor memory via `mx.metal.get_active_memory()` during development

## Interaction Style

- Write clean, well-documented code. Prioritize readability over cleverness.
- When implementing a neuron model, always state the governing differential equations first as comments, then implement the discrete update rule.
- Run tests after each significant change. Use `pytest -xvs` for verbose output.
- If unsure about MLX API behavior, write a minimal test script to verify before integrating.
- Commit messages follow Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- When creating a new module, always create the corresponding test file simultaneously.

## Reference Materials

- MLX documentation: https://ml-explore.github.io/mlx/
- MLX GitHub: https://github.com/ml-explore/mlx
- snnTorch tutorials: https://snntorch.readthedocs.io/
- Norse documentation: https://norse.github.io/norse/
- SpikingJelly: https://github.com/fangwei123456/spikingjelly
- Maass et al. (2002) "Real-time computing without stable states" — LSM reference
- Neftci et al. (2019) "Surrogate Gradient Learning in SNNs" — surrogate gradient survey
- Eshraghian et al. (2023) "Training SNNs Using Lessons From Deep Learning" — Proc. IEEE
