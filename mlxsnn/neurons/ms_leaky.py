"""Multi-Scale Leaky Integrate-and-Fire (MS-LIF) neuron model.

Uses K parallel LIF branches with learnable decay rates, each
capturing a different temporal scale.  Outputs are combined via
learnable mixing weights.

Membrane dynamics per scale k:
    U_k[t+1] = beta_k * U_k[t] + X[t+1]

Two firing modes are supported:

**mixed** (default, recommended):
    U_mixed[t+1] = sum_k alpha_k * U_k[t+1]
    S[t+1] = H(U_mixed - threshold)
    U_k[t+1] = reset(U_k[t+1], S[t+1])   (shared spike resets all scales)

    Mix membrane potentials first, fire once.  This produces clean
    binary spikes and lets multi-scale dynamics influence a single
    firing decision.

**per_scale** (legacy):
    S_k[t+1] = H(U_k[t+1] - threshold)
    U_k[t+1] = reset(U_k[t+1], S_k[t+1])
    S[t+1] = sum_k alpha_k * S_k[t+1]

    Fire each scale independently and mix spikes.  Output is
    fractional (weighted sum of binary spikes), which can degrade
    downstream SNN layers that expect binary inputs.

where alpha_k = softmax(mix_logits)_k.

Unlike the dual-compartment TS-LIF (ICLR 2025) which uses two
hard-coded compartments with cross-coupling, MS-LIF uses N
independent scales with data-driven beta learning — simpler,
more flexible, and with fewer parameters per neuron.

References:
    Eshraghian, J. K., et al. (2023). Training Spiking Neural Networks
    Using Lessons From Deep Learning. Proceedings of the IEEE.
"""

from __future__ import annotations

import mlx.core as mx

from mlxsnn.neurons.base import SpikingNeuron


class MSLeaky(SpikingNeuron):
    """Multi-Scale Leaky Integrate-and-Fire neuron.

    Maintains K parallel membrane potentials, each with its own
    learnable decay rate (beta).  Outputs are combined via learnable
    softmax weights.

    Args:
        num_scales: Number of parallel temporal scales (K).
        beta_min: Smallest initial beta (fastest decay / shortest memory).
        beta_max: Largest initial beta (slowest decay / longest memory).
        learn_beta: If True, betas are learnable parameters.
        learn_mix: If True, mixing weights are learnable parameters.
        fire_mode: Firing strategy. ``'mixed'`` (default) mixes membrane
            potentials then fires once (binary output). ``'per_scale'``
            fires each scale independently then mixes spikes (fractional
            output, legacy mode).
        threshold: Spike threshold voltage.
        learn_threshold: If True, threshold becomes learnable.
        reset_mechanism: Reset method ('subtract', 'zero', 'none').
        surrogate_fn: Surrogate gradient function name or callable.
        surrogate_scale: Scale parameter for surrogate gradient.

    Examples:
        >>> import mlx.core as mx
        >>> from mlxsnn.neurons import MSLeaky
        >>> ms = MSLeaky(num_scales=4, fire_mode='mixed')
        >>> state = ms.init_state(4, 128)
        >>> x = mx.ones((4, 128))
        >>> spk, state = ms(x, state)
        >>> spk.shape
        [4, 128]
    """

    def __init__(
        self,
        num_scales: int = 4,
        beta_min: float = 0.5,
        beta_max: float = 0.99,
        learn_beta: bool = True,
        learn_mix: bool = True,
        fire_mode: str = "mixed",
        threshold: float = 1.0,
        learn_threshold: bool = False,
        reset_mechanism: str = "subtract",
        surrogate_fn: str = "arctan",
        surrogate_scale: float = 2.0,
    ):
        if fire_mode not in ("mixed", "per_scale"):
            raise ValueError(
                f"fire_mode must be 'mixed' or 'per_scale', got '{fire_mode}'"
            )
        super().__init__(
            threshold=threshold,
            learn_threshold=learn_threshold,
            reset_mechanism=reset_mechanism,
            surrogate_fn=surrogate_fn,
            surrogate_scale=surrogate_scale,
        )
        self.num_scales = num_scales
        self.fire_mode = fire_mode

        # Initialize betas evenly spaced in [beta_min, beta_max].
        # When learnable, store as logits so sigmoid(logit) gives the
        # desired beta range.
        if num_scales == 1:
            target_betas = [(beta_min + beta_max) / 2.0]
        else:
            target_betas = [
                beta_min + i * (beta_max - beta_min) / (num_scales - 1)
                for i in range(num_scales)
            ]

        if learn_beta:
            # Convert to logit space: logit = log(beta / (1 - beta))
            import math
            logits = [math.log(b / (1.0 - b)) for b in target_betas]
            self.betas = mx.array(logits)
        else:
            self._betas_const = mx.array(target_betas)

        # Mixing weights (logits, softmax applied in forward)
        if learn_mix:
            self.mix_logits = mx.zeros((num_scales,))
        else:
            self._mix_const = mx.ones((num_scales,)) / num_scales

    def _get_betas(self):
        """Return betas in (0, 1). Applies sigmoid when learnable."""
        if hasattr(self, "_betas_const"):
            return self._betas_const
        return mx.sigmoid(self.betas)

    def _get_mix_weights(self):
        """Return normalized mixing weights."""
        if hasattr(self, "_mix_const"):
            return self._mix_const
        return mx.softmax(self.mix_logits)

    def init_state(self, batch_size: int, *shape) -> dict:
        """Initialize multi-scale membrane state.

        Args:
            batch_size: Number of samples.
            *shape: Feature dimensions.

        Returns:
            State dict with 'mem' of shape ``[B, K, *shape]``.
        """
        return {"mem": mx.zeros((batch_size, self.num_scales, *shape))}

    def __call__(self, x: mx.array, state: dict) -> tuple:
        """Forward one timestep.

        Args:
            x: Input current ``[B, D]`` (or ``[B, H, W, C]`` for conv).
            state: Dict with 'mem' of shape ``[B, K, ...]``.

        Returns:
            Tuple of ``(spikes [B, D], new_state)``.  In ``'mixed'`` mode
            the state also contains ``'mem_mixed'`` of shape ``[B, D]``
            for membrane-potential readout.
        """
        mem = state["mem"]  # [B, K, *features]
        betas = self._get_betas()  # [K]

        # Reshape betas for broadcasting: [1, K, 1, 1, ...]
        n_feat_dims = mem.ndim - 2  # dims beyond B and K
        beta_shape = [1, self.num_scales] + [1] * n_feat_dims
        betas_bc = betas.reshape(beta_shape)

        # Expand x: [B, 1, *features]
        x_expanded = mx.expand_dims(x, axis=1)

        # Leaky integration per scale
        mem = betas_bc * mem + x_expanded  # [B, K, *features]

        # Mixing weights
        mix_w = self._get_mix_weights()  # [K]
        mix_shape = [1, self.num_scales] + [1] * n_feat_dims
        mix_bc = mix_w.reshape(mix_shape)

        if self.fire_mode == "mixed":
            # Mix membrane potentials, then fire once → binary output
            mixed_mem = mx.sum(mem * mix_bc, axis=1)  # [B, *features]
            spk = self.fire(mixed_mem)  # [B, *features] — binary

            # Reset all scales using the shared firing decision
            spk_expanded = mx.expand_dims(spk, axis=1)  # [B, 1, *features]
            mem = self.reset(mem, spk_expanded)

            return spk, {"mem": mem, "mem_mixed": mixed_mem}
        else:
            # Legacy per_scale: fire each scale, mix spikes (fractional)
            spk = self.fire(mem)  # [B, K, *features]
            mem = self.reset(mem, spk)
            out = mx.sum(spk * mix_bc, axis=1)  # [B, *features]

            return out, {"mem": mem}
