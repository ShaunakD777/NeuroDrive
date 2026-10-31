"""Tests for core/optim/adam.py: hand values for 3 steps, bias correction, state handling."""

import numpy as np
import pytest

from neurodrive.core.optim.adam import Adam
from neurodrive.core.optim.sgd import SGD


def bowl_walk(opt, steps: int, scale: float = 1.0) -> list[float]:
    """Minimise f(w) = scale * (w - 3)^2 from w = 0; returns w after each step."""
    w, out = [np.array([0.0])], []
    for _ in range(steps):
        w = opt.step(w, [scale * 2.0 * (w[0] - 3.0)])
        out.append(float(w[0][0]))
    return out


def test_hand_values_three_steps():
    w1, w2, w3 = bowl_walk(Adam(0.1), 3)
    # step 1: g = -6, m_hat = -6, v_hat = 36, step 0.1
    assert w1 == pytest.approx(0.1, abs=1e-8)
    # step 2: g = -5.8, m = -1.12, v = 0.069604, m_hat = -5.89474, v_hat = 34.8194
    assert w2 == pytest.approx(0.19990, abs=1e-5)
    # step 3: g = -5.60021, m = -1.56802, v = 0.100897, m_hat = -5.78605, v_hat = 33.6659
    assert w3 == pytest.approx(0.29962, abs=1e-5)


def test_moments_after_two_steps_match_hand_values():
    opt = Adam(0.1)
    bowl_walk(opt, 2)
    assert opt.t == 2
    assert opt.first_moments is not None and opt.second_moments is not None
    assert opt.first_moments[0] == pytest.approx([-1.12], abs=1e-8)
    assert opt.second_moments[0] == pytest.approx([0.069604], abs=1e-8)


@pytest.mark.parametrize("g", [1e-3, 0.5, 3.0, 1e4, -7.0])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.5, 0.9), (0.0, 0.99)])
def test_first_step_is_eta_in_size(g, betas):
    (new,) = Adam(0.01, *betas).step([np.zeros(1)], [np.array([g])])
    assert abs(new[0]) == pytest.approx(0.01, rel=1e-4)
    assert np.sign(new[0]) == -np.sign(g)


def test_bias_correction_on_step_one_and_its_fading():
    opt = Adam(0.1)
    opt.step([np.zeros(1)], [np.array([-6.0])])
    assert opt.first_moments is not None
    assert opt.first_moments[0] / (1.0 - 0.9) == pytest.approx([-6.0])  # m_hat = g exactly
    assert 1.0 - 0.9**100 == pytest.approx(1.0, abs=1e-4)  # correction 1 has faded by t = 100
    # Without correction the first step would be eta * 0.1 g / sqrt(0.001 g^2) = 3.16 eta.
    uncorrected = 0.1 * 0.1 * 6.0 / np.sqrt(0.001 * 36.0)
    assert uncorrected == pytest.approx(0.1 * 0.1 / np.sqrt(0.001))
    assert uncorrected == pytest.approx(0.31623, abs=1e-5)


def test_beta1_zero_matches_bias_corrected_rmsprop_by_hand():
    eta, b2, eps = 0.05, 0.9, 1e-8
    opt, w, v = Adam(eta, 0.0, b2, eps), np.array([0.0]), 0.0
    p = [w.copy()]
    for t in range(1, 6):
        g = 2.0 * (w - 3.0)
        p = opt.step(p, [g])
        v = b2 * v + (1.0 - b2) * g * g
        w = w - eta * g / (np.sqrt(v / (1.0 - b2**t)) + eps)
        assert p[0] == pytest.approx(w, abs=1e-12)


def test_converges_on_the_bowl():
    assert bowl_walk(Adam(0.1), 500)[-1] == pytest.approx(3.0, abs=1e-3)


def test_scaling_the_loss_barely_changes_the_walk():
    assert np.allclose(bowl_walk(Adam(0.1), 50), bowl_walk(Adam(0.1), 50, scale=1000.0), atol=1e-6)


def test_zero_gradient_leaves_parameter_unchanged():
    (new,) = Adam(0.1).step([np.array([1.0, 2.0])], [np.zeros(2)])
    assert np.array_equal(new, [1.0, 2.0])


def test_separate_state_per_parameter():
    opt = Adam(0.1)
    opt.step([np.zeros(1), np.zeros(2)], [np.array([2.0]), np.array([0.0, -4.0])])
    assert opt.first_moments is not None and opt.second_moments is not None
    assert opt.first_moments[0] == pytest.approx([0.2])
    assert opt.first_moments[1] == pytest.approx([0.0, -0.4])
    assert opt.second_moments[1] == pytest.approx([0.0, 0.016])


def test_inputs_not_modified():
    p, g = np.array([1.0, 2.0]), np.array([0.5, -0.5])
    p0, g0 = p.copy(), g.copy()
    opt = Adam(0.1)
    opt.step([p], [g])
    opt.step([p], [g])
    assert np.array_equal(p, p0) and np.array_equal(g, g0)


def test_reset_and_independent_repeatable_runs():
    opt = Adam(0.1)
    first = bowl_walk(opt, 5)
    opt.reset()
    assert opt.t == 0
    assert bowl_walk(opt, 5) == first
    fresh = Adam(0.1)
    assert fresh.first_moments is None and fresh.t == 0
    assert bowl_walk(fresh, 5) == first


def test_shape_change_requires_reset():
    opt = Adam(0.1)
    opt.step([np.zeros(2)], [np.ones(2)])
    with pytest.raises(ValueError):
        opt.step([np.zeros(3)], [np.ones(3)])
    with pytest.raises(ValueError):
        opt.step([np.zeros(2), np.zeros(1)], [np.ones(2), np.ones(1)])


def test_bad_arguments_rejected():
    for lr in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            Adam(lr)
    for beta in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError):
            Adam(0.1, beta1=beta)
        with pytest.raises(ValueError):
            Adam(0.1, beta2=beta)
    for eps in (0.0, -1e-8, float("inf")):
        with pytest.raises(ValueError):
            Adam(0.1, eps=eps)
    with pytest.raises(ValueError):
        Adam(0.1).step([np.zeros(2)], [np.zeros(3)])


def test_closer_than_sgd_in_a_narrow_valley():
    """f = 0.5 (w1^2 + 100 w2^2), the same valley as the momentum and RMSProp tests."""
    scale = np.array([1.0, 100.0])

    def distance(opt) -> float:
        w = [np.array([1.0, 1.0])]
        for _ in range(100):
            w = opt.step(w, [scale * w[0]])
        return float(np.linalg.norm(w[0]))

    assert distance(Adam(0.05)) < 0.5 * distance(SGD(0.009))
