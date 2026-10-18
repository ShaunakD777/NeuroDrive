"""Tests for core/activations.py: hand-computed values, output ranges, numerical safety."""

import math
import warnings

import numpy as np
import pytest

from neurodrive.core.activations import ACTIVATIONS, get_activation, linear, relu, sigmoid, tanh

SWEEP = np.linspace(-50.0, 50.0, 2001)


def test_hand_computed_values_at_z_0_2():
    # tanh(0.2) = (e^0.2 - e^-0.2) / (e^0.2 + e^-0.2); sigmoid(0.2) = 1 / (1 + e^-0.2)
    assert tanh(0.2) == pytest.approx(0.19737532022490401, abs=1e-9)
    assert sigmoid(0.2) == pytest.approx(0.54983399731247795, abs=1e-9)
    assert relu(0.2) == pytest.approx(0.2, abs=1e-9)
    assert linear(0.2) == pytest.approx(0.2, abs=1e-9)


def test_hand_computed_values_at_z_minus_1():
    assert tanh(-1.0) == pytest.approx(-0.7615941559557649, abs=1e-9)
    assert sigmoid(-1.0) == pytest.approx(0.2689414213699951, abs=1e-9)
    assert relu(-1.0) == 0.0
    assert linear(-1.0) == -1.0


def test_special_points():
    assert tanh(0.0) == 0.0
    assert sigmoid(0.0) == 0.5
    assert relu(0.0) == 0.0


def test_output_ranges_over_a_wide_sweep():
    assert np.all(np.abs(tanh(SWEEP)) <= 1.0)
    assert np.all((sigmoid(SWEEP) >= 0.0) & (sigmoid(SWEEP) <= 1.0))
    assert np.all(relu(SWEEP) >= 0.0)
    assert np.array_equal(linear(SWEEP), SWEEP)
    assert tanh(SWEEP).min() < -0.99 and tanh(SWEEP).max() > 0.99  # tanh does go both ways
    assert sigmoid(SWEEP).min() >= 0.0  # sigmoid never steers "negative"


def test_tanh_is_odd_and_sigmoid_is_symmetric():
    assert np.allclose(tanh(-SWEEP), -tanh(SWEEP), atol=1e-12)
    assert np.allclose(sigmoid(SWEEP) + sigmoid(-SWEEP), 1.0, atol=1e-12)


def test_sigmoid_matches_naive_formula_where_that_is_safe():
    z = np.linspace(-30.0, 30.0, 601)
    assert np.allclose(sigmoid(z), 1.0 / (1.0 + np.exp(-z)), atol=1e-12)


def test_sigmoid_extremes_do_not_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning becomes a failure
        out = sigmoid(np.array([-1000.0, 1000.0]))
    assert out[0] == 0.0 and out[1] == 1.0


def test_shape_is_preserved_and_dtype_is_float64():
    z = np.arange(-3, 3).reshape(2, 3)  # integers in, floats out
    for f in ACTIVATIONS.values():
        out = f(z)
        assert out.shape == (2, 3)
        assert out.dtype == np.float64


def test_linear_returns_a_copy():
    z = np.array([1.0, 2.0])
    out = linear(z)
    out[0] = 99.0
    assert z[0] == 1.0


def test_lookup_by_name():
    for name, f in ACTIVATIONS.items():
        assert get_activation(name) is f
    assert math.isclose(float(get_activation("tanh")(0.2)), 0.19737532022490401, abs_tol=1e-9)


def test_unknown_name_lists_valid_names():
    with pytest.raises(ValueError, match="unknown activation 'gelu'.*linear.*relu.*sigmoid.*tanh"):
        get_activation("gelu")
