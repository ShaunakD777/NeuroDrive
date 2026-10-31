"""Tests for core/optim/rmsprop.py: hand values, per-parameter scaling, state handling."""

import numpy as np
import pytest

from neurodrive.core.optim.rmsprop import RMSProp
from neurodrive.core.optim.sgd import SGD


def bowl_walk(opt, steps: int, scale: float = 1.0) -> list[float]:
    """Minimise f(w) = scale * (w - 3)^2 from w = 0; returns w after each step."""
    w, out = [np.array([0.0])], []
    for _ in range(steps):
        w = opt.step(w, [scale * 2.0 * (w[0] - 3.0)])
        out.append(float(w[0][0]))
    return out


def test_hand_example_equal_steps_for_different_gradients():
    (new,) = RMSProp(0.1, 0.9).step([np.zeros(2)], [np.array([10.0, 0.1])])
    # s = [10, 0.001], sqrt(s) = [3.1623, 0.031623], step = 0.1 * [3.1623, 3.1623] = 0.31623 each
    assert new == pytest.approx([-0.31623, -0.31623], abs=1e-5)


def test_bowl_hand_values_three_steps():
    w1, w2, w3 = bowl_walk(RMSProp(0.1, 0.9), 3)
    assert w1 == pytest.approx(0.31623, abs=1e-5)
    assert w2 == pytest.approx(0.53318, abs=1e-5)
    # step 3: g = 2(0.53318 - 3) = -4.93364, s = 0.9 * 6.12105 + 0.1 * 24.3408 = 7.94303,
    # sqrt(s) = 2.81834, w = 0.53318 + 0.1 * 4.93364 / 2.81834 = 0.70823
    assert w3 == pytest.approx(0.70823, abs=1e-5)


@pytest.mark.parametrize("g", [1e-4, 0.5, 3.0, 1e4, -7.0])
@pytest.mark.parametrize("rho", [0.5, 0.9, 0.99])
def test_first_step_size_is_eta_over_sqrt_one_minus_rho(g, rho):
    """|step| = eta |g| / (sqrt(1 - rho) |g| + eps): eta / sqrt(1 - rho) unless g is near eps."""
    eta, eps = 0.01, 1e-8
    (new,) = RMSProp(eta, rho, eps).step([np.zeros(1)], [np.array([g])])
    exact = eta * abs(g) / (np.sqrt(1.0 - rho) * abs(g) + eps)
    assert abs(new[0]) == pytest.approx(exact, rel=1e-12)
    assert abs(new[0]) == pytest.approx(eta / np.sqrt(1.0 - rho), rel=1e-3)
    assert np.sign(new[0]) == -np.sign(g)


def test_scaling_the_loss_barely_changes_the_walk():
    plain, scaled = bowl_walk(RMSProp(0.1), 50), bowl_walk(RMSProp(0.1), 50, scale=1000.0)
    assert np.allclose(plain, scaled, atol=1e-6)
    sgd_plain, sgd_scaled = bowl_walk(SGD(1e-4), 5), bowl_walk(SGD(1e-4), 5, scale=1000.0)
    assert abs(sgd_scaled[-1] - sgd_plain[-1]) > 1.0  # SGD's step is 1000 times bigger


def test_zero_gradient_leaves_parameter_unchanged():
    (new,) = RMSProp(0.1).step([np.array([1.0, 2.0])], [np.zeros(2)])
    assert np.array_equal(new, [1.0, 2.0])


def test_separate_averages_per_parameter():
    opt = RMSProp(0.1, 0.5)
    opt.step([np.zeros(1), np.zeros(1)], [np.array([2.0]), np.array([0.0])])
    assert opt.square_averages is not None
    assert opt.square_averages[0] == pytest.approx([2.0])  # 0.5 * 4
    assert opt.square_averages[1] == pytest.approx([0.0])


def test_inputs_not_modified():
    p, g = np.array([1.0, 2.0]), np.array([0.5, -0.5])
    p0, g0 = p.copy(), g.copy()
    opt = RMSProp(0.1)
    opt.step([p], [g])
    opt.step([p], [g])
    assert np.array_equal(p, p0) and np.array_equal(g, g0)


def test_reset_and_independent_repeatable_runs():
    opt = RMSProp(0.1)
    first = bowl_walk(opt, 5)
    opt.reset()
    assert bowl_walk(opt, 5) == first
    fresh = RMSProp(0.1)
    assert fresh.square_averages is None
    assert bowl_walk(fresh, 5) == first


def test_shape_change_requires_reset():
    opt = RMSProp(0.1)
    opt.step([np.zeros(2)], [np.ones(2)])
    with pytest.raises(ValueError):
        opt.step([np.zeros(3)], [np.ones(3)])
    with pytest.raises(ValueError):
        opt.step([np.zeros(2), np.zeros(1)], [np.ones(2), np.ones(1)])


def test_bad_arguments_rejected():
    for lr in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            RMSProp(lr)
    for rho in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError):
            RMSProp(0.1, rho=rho)
    for eps in (0.0, -1e-8, float("inf")):
        with pytest.raises(ValueError):
            RMSProp(0.1, eps=eps)
    with pytest.raises(ValueError):
        RMSProp(0.1).step([np.zeros(2)], [np.zeros(3)])


def test_closer_than_sgd_in_a_narrow_valley():
    """f = 0.5 (w1^2 + 100 w2^2), the same valley as the momentum test.

    With a fixed learning rate RMSProp keeps stepping about eta in each direction, so it hovers
    near the minimum instead of settling: after 100 steps 0.124 away, against 0.405 for SGD.
    """
    scale = np.array([1.0, 100.0])

    def distance(opt) -> float:
        w = [np.array([1.0, 1.0])]
        for _ in range(100):
            w = opt.step(w, [scale * w[0]])
        return float(np.linalg.norm(w[0]))

    assert distance(RMSProp(0.01)) < 0.5 * distance(SGD(0.009))
