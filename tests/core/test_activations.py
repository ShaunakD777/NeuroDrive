"""Tests for core/activations.py: hand-computed values, output ranges, numerical safety."""

import math
import warnings

import numpy as np
import pytest

from neurodrive.core.activations import (
    ACTIVATIONS,
    DERIVATIVES,
    get_activation,
    get_derivative,
    linear,
    linear_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)

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


# --- Derivatives (task 3.2) ---

FD_POINTS = np.linspace(-10.0, 10.0, 2000)  # step about 0.01; does not contain 0 exactly


def test_derivative_hand_values_at_z_0_and_2():
    # tanh'(z) = 1 - tanh(z)^2; sigmoid'(z) = s(1 - s); relu' is 1 for z > 0, 0 at the corner
    assert tanh_derivative(0.0) == 1.0
    assert tanh_derivative(2.0) == pytest.approx(0.07065082485316443, abs=1e-9)
    assert sigmoid_derivative(0.0) == 0.25
    assert sigmoid_derivative(2.0) == pytest.approx(0.10499358540350652, abs=1e-9)
    assert relu_derivative(2.0) == 1.0 and relu_derivative(0.0) == 0.0
    assert relu_derivative(-1.0) == 0.0
    assert linear_derivative(0.0) == 1.0 and linear_derivative(2.0) == 1.0


@pytest.mark.parametrize("name", sorted(ACTIVATIONS))
def test_derivative_matches_central_finite_differences(name):
    f, df = ACTIVATIONS[name], DERIVATIVES[name]
    z = FD_POINTS
    if name == "relu":
        z = z[np.abs(z) > 1e-3]  # the corner at 0 has no slope to estimate
    h = 1e-5
    numeric = (f(z + h) - f(z - h)) / (2.0 * h)
    assert np.allclose(df(z), numeric, rtol=1e-6, atol=1e-9)


def test_sigmoid_maximum_slope_is_0_25_at_zero():
    z = np.linspace(-10.0, 10.0, 20001)  # contains 0 exactly
    slopes = sigmoid_derivative(z)
    assert slopes.max() == 0.25
    assert z[np.argmax(slopes)] == 0.0
    assert np.all(slopes <= 0.25)


def test_tanh_maximum_slope_is_1_at_zero_and_saturates():
    z = np.linspace(-10.0, 10.0, 20001)
    slopes = tanh_derivative(z)
    assert slopes.max() == 1.0 and z[np.argmax(slopes)] == 0.0
    assert tanh_derivative(5.0) < 2e-4 and sigmoid_derivative(10.0) < 5e-5  # saturation


def test_derivatives_are_non_negative_and_symmetric():
    for name in ("tanh", "sigmoid"):
        df = DERIVATIVES[name]
        assert np.all(df(SWEEP) >= 0.0)
        assert np.allclose(df(-SWEEP), df(SWEEP), atol=1e-15)
    assert set(np.unique(relu_derivative(SWEEP))) <= {0.0, 1.0}


def test_derivative_extremes_do_not_overflow():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        z = np.array([-1000.0, 1000.0])
        assert np.array_equal(tanh_derivative(z), [0.0, 0.0])
        assert np.array_equal(sigmoid_derivative(z), [0.0, 0.0])


def test_derivative_shape_and_dtype_are_kept():
    z = np.arange(-3, 3).reshape(2, 3)
    for df in DERIVATIVES.values():
        out = df(z)
        assert out.shape == (2, 3) and out.dtype == np.float64


def test_derivative_lookup_by_name():
    assert set(DERIVATIVES) == set(ACTIVATIONS)
    for name, df in DERIVATIVES.items():
        assert get_derivative(name) is df
    with pytest.raises(ValueError, match="unknown activation 'gelu'.*linear.*relu.*sigmoid.*tanh"):
        get_derivative("gelu")
