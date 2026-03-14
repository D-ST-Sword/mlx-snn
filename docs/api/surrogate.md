# Surrogate Gradients

Surrogate gradient functions enable backpropagation through the non-differentiable Heaviside spike function. The forward pass uses a hard threshold; the backward pass uses a smooth approximation.

## Factory Function

::: mlxsnn.surrogate.get_surrogate

## Available Surrogates

### Fast Sigmoid

Default surrogate. Good balance of speed and gradient quality.

$$\frac{\partial S}{\partial U} \approx \frac{\alpha}{(1 + \alpha|U - V_{thr}|)^2}$$

::: mlxsnn.surrogate.fast_sigmoid.fast_sigmoid_surrogate

### Arctan

Smoother gradient landscape, useful when training is unstable.

::: mlxsnn.surrogate.arctan.arctan_surrogate

### Sigmoid

Standard logistic sigmoid derivative as surrogate.

::: mlxsnn.surrogate.sigmoid.sigmoid_surrogate

### Triangular (Tent)

Localized gradient with compact support near threshold.

::: mlxsnn.surrogate.triangular.triangular_surrogate

### Straight-Through Estimator

Simplest surrogate — unit gradient everywhere.

::: mlxsnn.surrogate.straight_through.straight_through_surrogate
