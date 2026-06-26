"""test_normalization.py - Unit tests for model/normalization.py, model/rms_normalization.py,
and model/batch_normalization.py.

Tests the forward pass of all three normalization variants: output shape,
numerical values, rounding precision, and training vs. inference branching
for batch normalization.
"""

import numpy as np
import pytest
from model.normalization import Solution as LayerNormSolution
from model.rms_normalization import Solution as RMSNormSolution
from model.batch_normalization import Solution as BatchNormSolution


# ---------------------------------------------------------------------------
# Layer Normalization
# ---------------------------------------------------------------------------

class TestLayerNorm:
    """Tests for LayerNormSolution.forward."""

    @pytest.fixture
    def ln(self):
        """Return a LayerNormSolution instance."""
        return LayerNormSolution()

    def test_output_shape(self, ln):
        """Output has the same shape as input x."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        gamma = np.ones(4)
        beta = np.zeros(4)
        out = ln.forward(x, gamma, beta)
        assert out.shape == x.shape

    def test_zero_mean_after_norm(self, ln):
        """With gamma=1 and beta=0, the output mean is approximately zero."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = ln.forward(x, np.ones(4), np.zeros(4))
        assert abs(float(np.mean(out))) < 1e-4

    def test_unit_variance_after_norm(self, ln):
        """With gamma=1 and beta=0, the output variance is approximately one."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = ln.forward(x, np.ones(4), np.zeros(4))
        assert abs(float(np.var(out)) - 1.0) < 1e-3

    def test_beta_shifts_output(self, ln):
        """A nonzero beta shifts every element by that constant."""
        x = np.array([1.0, 2.0, 3.0])
        out_no_shift = ln.forward(x, np.ones(3), np.zeros(3))
        out_shifted = ln.forward(x, np.ones(3), np.full(3, 5.0))
        assert np.allclose(out_shifted, out_no_shift + 5.0, atol=1e-4)

    def test_gamma_scales_output(self, ln):
        """A uniform gamma scales the normalized output by that factor."""
        x = np.array([1.0, 2.0, 3.0])
        out_base = ln.forward(x, np.ones(3), np.zeros(3))
        out_scaled = ln.forward(x, np.full(3, 2.0), np.zeros(3))
        assert np.allclose(out_scaled, out_base * 2.0, atol=1e-4)

    def test_rounded_to_five_decimals(self, ln):
        """Output values are rounded to exactly 5 decimal places."""
        x = np.array([1.1, 2.2, 3.3])
        out = ln.forward(x, np.ones(3), np.zeros(3))
        assert np.all(out == np.round(out, 5))

    def test_uniform_input_numerical_stability(self, ln):
        """Uniform input (zero variance) does not raise a division by zero error."""
        x = np.array([5.0, 5.0, 5.0])
        out = ln.forward(x, np.ones(3), np.zeros(3))
        assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# RMS Normalization
# ---------------------------------------------------------------------------

class TestRMSNorm:
    """Tests for RMSNormSolution.rms_norm."""

    @pytest.fixture
    def rms(self):
        """Return an RMSNormSolution instance."""
        return RMSNormSolution()

    def test_output_length(self, rms):
        """Output has the same length as input x."""
        x = [1.0, 2.0, 3.0]
        gamma = [1.0, 1.0, 1.0]
        out = rms.rms_norm(x, gamma, eps=1e-8)
        assert len(out) == len(x)

    def test_no_mean_centering(self, rms):
        """Positive input with gamma=1 should still have a positive mean (no centering)."""
        x = [1.0, 2.0, 3.0]
        out = rms.rms_norm(x, [1.0, 1.0, 1.0], eps=1e-8)
        assert float(np.mean(out)) > 0

    def test_gamma_scaling(self, rms):
        """Doubling gamma doubles every output value."""
        x = [1.0, 2.0, 3.0]
        out1 = np.array(rms.rms_norm(x, [1.0, 1.0, 1.0], eps=1e-8))
        out2 = np.array(rms.rms_norm(x, [2.0, 2.0, 2.0], eps=1e-8))
        assert np.allclose(out2, out1 * 2.0, atol=1e-3)

    def test_rounded_to_four_decimals(self, rms):
        """Output values are rounded to exactly 4 decimal places."""
        x = [1.1, 2.2, 3.3]
        out = np.array(rms.rms_norm(x, [1.0, 1.0, 1.0], eps=1e-8))
        assert np.all(out == np.round(out, 4))

    def test_no_nan_on_small_eps(self, rms):
        """Output does not contain NaN values for a valid positive eps."""
        x = [0.1, 0.2, 0.3]
        out = rms.rms_norm(x, [1.0, 1.0, 1.0], eps=1e-8)
        assert not np.any(np.isnan(out))

    def test_numerical_values(self, rms):
        """RMS norm of [3, 4] with gamma=1 approximates [3/5, 4/5]."""
        x = [3.0, 4.0]
        out = np.array(rms.rms_norm(x, [1.0, 1.0], eps=0.0))
        rms_val = np.sqrt((9.0 + 16.0) / 2.0)
        expected = np.round(np.array([3.0 / rms_val, 4.0 / rms_val]), 4)
        assert np.allclose(out, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Batch Normalization
# ---------------------------------------------------------------------------

class TestBatchNorm:
    """Tests for BatchNormSolution.batch_norm."""

    @pytest.fixture
    def bn(self):
        """Return a BatchNormSolution instance."""
        return BatchNormSolution()

    @pytest.fixture
    def simple_input(self):
        """A 4x2 input batch with known statistics."""
        return [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

    def test_returns_three_tuple(self, bn, simple_input):
        """batch_norm returns a 3-tuple (output, running_mean, running_var)."""
        result = bn.batch_norm(
            simple_input, [1.0, 1.0], [0.0, 0.0],
            [0.0, 0.0], [1.0, 1.0], 0.1, 1e-5, True
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_output_shape_training(self, bn, simple_input):
        """Training mode output has the same shape as the input."""
        out, _, _ = bn.batch_norm(
            simple_input, [1.0, 1.0], [0.0, 0.0],
            [0.0, 0.0], [1.0, 1.0], 0.1, 1e-5, True
        )
        assert np.array(out).shape == np.array(simple_input).shape

    def test_running_stats_updated_during_training(self, bn, simple_input):
        """Running mean and variance are updated (not equal to the initial values) in training mode."""
        _, rm, rv = bn.batch_norm(
            simple_input, [1.0, 1.0], [0.0, 0.0],
            [0.0, 0.0], [1.0, 1.0], 0.1, 1e-5, True
        )
        assert not np.allclose(rm, [0.0, 0.0])

    def test_running_stats_unchanged_during_inference(self, bn, simple_input):
        """Running statistics are not modified in inference mode."""
        init_mean = [0.5, 0.5]
        init_var = [1.0, 1.0]
        _, rm, rv = bn.batch_norm(
            simple_input, [1.0, 1.0], [0.0, 0.0],
            list(init_mean), list(init_var), 0.1, 1e-5, False
        )
        assert np.allclose(rm, np.round(init_mean, 4))
        assert np.allclose(rv, np.round(init_var, 4))

    def test_beta_shifts_output(self, bn, simple_input):
        """A nonzero beta shifts every output element by that constant."""
        out_zero, _, _ = bn.batch_norm(
            simple_input, [1.0, 1.0], [0.0, 0.0],
            [0.0, 0.0], [1.0, 1.0], 0.1, 1e-5, True
        )
        out_shifted, _, _ = bn.batch_norm(
            simple_input, [1.0, 1.0], [3.0, 3.0],
            [0.0, 0.0], [1.0, 1.0], 0.1, 1e-5, True
        )
        assert np.allclose(np.array(out_shifted), np.array(out_zero) + 3.0, atol=1e-3)

    def test_rounded_to_four_decimals(self, bn, simple_input):
        """All three returned arrays are rounded to 4 decimal places."""
        out, rm, rv = bn.batch_norm(
            simple_input, [1.0, 1.0], [0.0, 0.0],
            [0.0, 0.0], [1.0, 1.0], 0.1, 1e-5, True
        )
        for arr in [out, rm, rv]:
            arr_np = np.array(arr)
            assert np.all(arr_np == np.round(arr_np, 4))
