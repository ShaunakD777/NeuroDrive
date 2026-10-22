"""Tests for core/losses.py: hand values, derivatives against finite differences, guards."""

import numpy as np
import pytest

from neurodrive.core.losses import huber, huber_grad, mse, mse_grad
from neurodrive.core.rng import make_rng

# Worked example from the Brief: residuals r = [1, 0, 3], M = 3.
Y_HAT = np.array([2.0, 0.0, 3.0])
Y = np.array([1.0, 0.0, 0.0])


def _numeric_grad(loss, y_hat: np.ndarray, y: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central finite differences: nudge each prediction by ±h and divide the loss change by 2h."""
    grad = np.zeros_like(y_hat)
    for i in np.ndindex(y_hat.shape):
        up, down = y_hat.copy(), y_hat.copy()
        up[i] += h
        down[i] -= h
        grad[i] = (loss(up, y) - loss(down, y)) / (2.0 * h)
    return grad


def test_mse_hand_example():
    # (1 + 0 + 9) / 3 and 2 * [1, 0, 3] / 3
    assert mse(Y_HAT, Y) == pytest.approx(10.0 / 3.0, abs=1e-12)
    assert np.allclose(mse_grad(Y_HAT, Y), [2.0 / 3.0, 0.0, 2.0], atol=1e-12)


def test_huber_hand_example():
    # h = [0.5, 0, 1 * (3 - 0.5)] -> mean 1.0; clip([1, 0, 3], -1, 1) / 3
    assert huber(Y_HAT, Y) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(huber_grad(Y_HAT, Y), [1.0 / 3.0, 0.0, 1.0 / 3.0], atol=1e-12)


def test_huber_with_other_delta_by_hand():
    # delta = 2: r = [1, 0, 3] -> h = [0.5, 0, 2 * (3 - 1)] = [0.5, 0, 4] -> mean 1.5
    assert huber(Y_HAT, Y, delta=2.0) == pytest.approx(1.5, abs=1e-12)
    assert np.allclose(huber_grad(Y_HAT, Y, delta=2.0), [1.0 / 3.0, 0.0, 2.0 / 3.0], atol=1e-12)


def test_perfect_predictions_give_zero_loss_and_zero_gradient():
    y = np.array([[0.3, -0.7], [1.2, 0.0]])
    assert mse(y, y) == 0.0 and huber(y, y) == 0.0
    assert not np.any(mse_grad(y, y)) and not np.any(huber_grad(y, y))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shape", [(7,), (5, 2), (3, 4)])
def test_mse_grad_matches_finite_differences(seed, shape):
    rng = make_rng(seed)
    y_hat, y = rng.normal(size=shape), rng.normal(size=shape)
    assert np.allclose(mse_grad(y_hat, y), _numeric_grad(mse, y_hat, y), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("seed", range(5))
def test_huber_grad_matches_finite_differences(seed, delta):
    rng = make_rng(seed)
    # residuals spread evenly over -4..4 (plus jitter), so both pieces of Huber are always used
    y = rng.normal(size=(6, 2))
    y_hat = y + (np.linspace(-4.0, 4.0, 12) + rng.uniform(-0.1, 0.1, size=12)).reshape(6, 2)
    r = np.abs(y_hat - y)
    assert np.any(r < delta) and np.any(r > delta)

    def loss(a, b):
        return huber(a, b, delta)

    numeric = _numeric_grad(loss, y_hat, y)
    assert np.allclose(huber_grad(y_hat, y, delta), numeric, rtol=1e-6, atol=1e-9)


def test_huber_derivative_is_continuous_at_the_threshold():
    # exactly at |r| = delta and a hair either side, the slope is delta / M
    for r in (1.0 - 1e-9, 1.0, 1.0 + 1e-9):
        assert huber_grad([r], [0.0])[0] == pytest.approx(1.0, abs=1e-8)
        assert huber_grad([-r], [0.0])[0] == pytest.approx(-1.0, abs=1e-8)
    assert huber([1.0], [0.0]) == pytest.approx(0.5)  # both pieces give ½ δ² there


def test_huber_is_half_mse_for_small_misses_and_linear_for_large():
    small, zeros = np.linspace(-0.9, 0.9, 19), np.zeros(19)
    assert huber(small, zeros) == pytest.approx(0.5 * mse(small, zeros))
    # beyond delta, each extra unit of miss adds exactly delta to h
    for r in (5.0, 50.0, 500.0):
        assert huber([r + 1.0], [0.0]) - huber([r], [0.0]) == pytest.approx(1.0)


def test_outlier_pushes_much_harder_under_mse_than_huber():
    y_hat, y = np.array([0.1, 0.2, 100.0]), np.zeros(3)
    assert mse_grad(y_hat, y)[2] / mse_grad(y_hat, y)[1] == pytest.approx(500.0)
    assert huber_grad(y_hat, y)[2] / huber_grad(y_hat, y)[1] == pytest.approx(5.0)


def test_gradient_has_the_prediction_shape_and_scales_with_batch_size():
    rng = make_rng(0)
    y_hat, y = rng.normal(size=(4, 2)), rng.normal(size=(4, 2))
    assert mse_grad(y_hat, y).shape == (4, 2) and huber_grad(y_hat, y).shape == (4, 2)
    # duplicating the batch leaves the loss unchanged and halves each entry's gradient
    y_hat2, y2 = np.vstack([y_hat, y_hat]), np.vstack([y, y])
    assert mse(y_hat2, y2) == pytest.approx(mse(y_hat, y))
    assert np.allclose(mse_grad(y_hat2, y2)[:4], mse_grad(y_hat, y) / 2.0)


def test_losses_return_python_floats():
    assert type(mse(Y_HAT, Y)) is float and type(huber(Y_HAT, Y)) is float


@pytest.mark.parametrize("fn", [mse, mse_grad, huber, huber_grad])
def test_mismatched_shapes_are_refused(fn):
    # (3, 1) against (3,) would broadcast into a 3 x 3 table of misses
    with pytest.raises(ValueError, match="shape"):
        fn(np.zeros((3, 1)), np.zeros(3))


@pytest.mark.parametrize("fn", [mse, mse_grad, huber, huber_grad])
def test_empty_input_is_refused(fn):
    with pytest.raises(ValueError, match="zero predictions"):
        fn(np.zeros((0, 2)), np.zeros((0, 2)))


@pytest.mark.parametrize("delta", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("fn", [huber, huber_grad])
def test_bad_delta_is_refused(fn, delta):
    with pytest.raises(ValueError, match="delta"):
        fn(Y_HAT, Y, delta=delta)
