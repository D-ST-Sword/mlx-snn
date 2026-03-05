"""Tests for surrogate gradient functions."""

import mlx.core as mx
import numpy as np
import pytest

from mlxsnn.surrogate import get_surrogate
from mlxsnn.surrogate.fast_sigmoid import fast_sigmoid_surrogate
from mlxsnn.surrogate.arctan import arctan_surrogate
from mlxsnn.surrogate.straight_through import straight_through_surrogate
from mlxsnn.surrogate.custom import custom_surrogate


class TestFastSigmoid:
    """Tests for fast sigmoid surrogate gradient."""

    def test_forward_heaviside(self):
        """Forward pass should be Heaviside step function."""
        fn = fast_sigmoid_surrogate(scale=25.0)
        x = mx.array([-1.0, -0.1, 0.0, 0.1, 1.0])
        out = fn(x)
        mx.eval(out)
        expected = mx.array([0.0, 0.0, 1.0, 1.0, 1.0])
        assert mx.allclose(out, expected).item()

    def test_backward_nonzero(self):
        """Backward pass should produce non-zero gradients."""
        fn = fast_sigmoid_surrogate(scale=25.0)
        grad_fn = mx.grad(lambda x: mx.sum(fn(x)))
        x = mx.array([0.0, 0.1, -0.1])
        grads = grad_fn(x)
        mx.eval(grads)
        # Gradients should be positive (surrogate is symmetric positive)
        assert mx.all(grads > 0).item()

    def test_gradient_peak_at_zero(self):
        """Surrogate gradient should peak near x=0."""
        fn = fast_sigmoid_surrogate(scale=25.0)
        grad_fn = mx.grad(lambda x: mx.sum(fn(x)))
        x_zero = mx.array([0.0])
        x_far = mx.array([1.0])
        g_zero = grad_fn(x_zero)
        g_far = grad_fn(x_far)
        mx.eval(g_zero, g_far)
        assert g_zero.item() > g_far.item()

    def test_scale_parameter(self):
        """Higher scale should produce sharper gradients."""
        fn_sharp = fast_sigmoid_surrogate(scale=100.0)
        fn_wide = fast_sigmoid_surrogate(scale=5.0)
        grad_sharp = mx.grad(lambda x: mx.sum(fn_sharp(x)))
        grad_wide = mx.grad(lambda x: mx.sum(fn_wide(x)))

        x = mx.array([0.0])
        g_sharp = grad_sharp(x)
        g_wide = grad_wide(x)
        mx.eval(g_sharp, g_wide)
        # Sharper gradient should be larger at x=0
        assert g_sharp.item() > g_wide.item()


class TestArctan:
    """Tests for arctan surrogate gradient."""

    def test_forward_heaviside(self):
        fn = arctan_surrogate(alpha=2.0)
        x = mx.array([-1.0, -0.1, 0.0, 0.1, 1.0])
        out = fn(x)
        mx.eval(out)
        expected = mx.array([0.0, 0.0, 1.0, 1.0, 1.0])
        assert mx.allclose(out, expected).item()

    def test_backward_nonzero(self):
        fn = arctan_surrogate(alpha=2.0)
        grad_fn = mx.grad(lambda x: mx.sum(fn(x)))
        x = mx.array([0.0, 0.1, -0.1])
        grads = grad_fn(x)
        mx.eval(grads)
        assert mx.all(grads > 0).item()


class TestStraightThrough:
    """Tests for straight-through estimator."""

    def test_forward_heaviside(self):
        fn = straight_through_surrogate(scale=1.0)
        x = mx.array([-1.0, 0.0, 1.0])
        out = fn(x)
        mx.eval(out)
        expected = mx.array([0.0, 1.0, 1.0])
        assert mx.allclose(out, expected).item()

    def test_gradient_everywhere(self):
        """Gradient should be 1 everywhere (identity pass-through)."""
        fn = straight_through_surrogate(scale=1.0)
        grad_fn = mx.grad(lambda x: mx.sum(fn(x)))
        # Gradient at x=0
        g_zero = grad_fn(mx.array([0.0]))
        mx.eval(g_zero)
        assert np.isclose(g_zero.item(), 1.0, atol=1e-5)
        # Gradient at x=1.0 (far from threshold) should also be 1.0
        g_far = grad_fn(mx.array([1.0]))
        mx.eval(g_far)
        assert np.isclose(g_far.item(), 1.0, atol=1e-5)
        # Gradient at x=-2.0
        g_neg = grad_fn(mx.array([-2.0]))
        mx.eval(g_neg)
        assert np.isclose(g_neg.item(), 1.0, atol=1e-5)


class TestCustomSurrogate:
    """Tests for custom user-defined surrogate gradients."""

    def test_custom_sigmoid(self):
        """Custom sigmoid surrogate should work."""
        fn = custom_surrogate(lambda x: mx.sigmoid(50.0 * x))
        x = mx.array([-1.0, 0.0, 1.0])
        out = fn(x)
        mx.eval(out)
        expected = mx.array([0.0, 1.0, 1.0])
        assert mx.allclose(out, expected).item()

        grad_fn = mx.grad(lambda x: mx.sum(fn(x)))
        grads = grad_fn(mx.array([0.0]))
        mx.eval(grads)
        assert grads.item() > 0


class TestGetSurrogate:
    """Tests for the surrogate registry."""

    def test_get_by_name(self):
        fn = get_surrogate("fast_sigmoid", scale=25.0)
        assert callable(fn)

    def test_get_arctan(self):
        fn = get_surrogate("arctan", scale=2.0)
        assert callable(fn)

    def test_get_straight_through(self):
        fn = get_surrogate("straight_through", scale=1.0)
        assert callable(fn)

    def test_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown surrogate"):
            get_surrogate("nonexistent")

    def test_callable_passthrough(self):
        """Passing a callable should return it directly."""
        my_fn = lambda x: x
        result = get_surrogate(my_fn)
        assert result is my_fn


class TestGradientFlow:
    """Test that gradients flow through neuron models using surrogates."""

    def test_lif_gradient_flow(self):
        """Gradients should flow through LIF neuron via surrogate."""
        from mlxsnn.neurons import Leaky

        lif = Leaky(beta=0.9, threshold=1.0)
        state = lif.init_state(batch_size=1, features=4)

        def forward(x):
            spk, _ = lif(x, state)
            return mx.sum(spk)

        x = mx.array([[0.5, 1.5, 0.8, 1.2]])
        grad_fn = mx.grad(forward)
        grads = grad_fn(x)
        mx.eval(grads)
        # Should have non-zero gradients
        assert mx.any(grads != 0).item()

    def test_if_gradient_flow(self):
        """Gradients should flow through IF neuron via surrogate."""
        from mlxsnn.neurons import IF

        neuron = IF(threshold=1.0)
        state = neuron.init_state(batch_size=1, features=4)

        def forward(x):
            spk, _ = neuron(x, state)
            return mx.sum(spk)

        x = mx.array([[0.5, 1.5, 0.8, 1.2]])
        grad_fn = mx.grad(forward)
        grads = grad_fn(x)
        mx.eval(grads)
        assert mx.any(grads != 0).item()

    def test_multi_step_gradient(self):
        """Gradients should flow through multiple timesteps (BPTT)."""
        from mlxsnn.neurons import Leaky

        lif = Leaky(beta=0.9, threshold=1.0)

        def forward(x_seq):
            state = lif.init_state(batch_size=1, features=2)
            total = mx.array(0.0)
            for t in range(x_seq.shape[0]):
                spk, state = lif(x_seq[t:t+1], state)
                total = total + mx.sum(spk)
            return total

        x_seq = mx.ones((5, 1, 2)) * 0.5
        grad_fn = mx.grad(forward)
        grads = grad_fn(x_seq)
        mx.eval(grads)
        assert grads.shape == (5, 1, 2)
        # At least some timesteps should have nonzero gradients
        assert mx.any(grads != 0).item()
