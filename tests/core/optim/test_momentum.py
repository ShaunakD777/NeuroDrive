"""Tests for core/optim/momentum.py: hand values, beta = 0 equals SGD, state handling."""

import numpy as np
import pytest

from neurodrive.core.optim.momentum import Momentum
from neurodrive.core.optim.sgd import SGD


def bowl_walk(opt, steps: int) -> list[float]:
    """Minimise f(w) = (w - 3)^2 from w = 0; returns w after each step."""
    w, out = [np.array([0.0])], []
    for _ in range(steps):
        w = opt.step(w, [2.0 * (w[0] - 3.0)])
        out.append(float(w[0][0]))
    return out


def test_hand_values_three_steps():
    w1, w2, w3 = bowl_walk(Momentum(0.1, 0.9), 3)
    assert w1 == pytest.approx(0.6, abs=1e-12)
    assert w2 == pytest.approx(1.62, abs=1e-12)
    # step 3: g = 2(1.62 - 3) = -2.76; v = 0.9 * -10.2 - 2.76 = -11.94; w = 1.62 + 1.194
    assert w3 == pytest.approx(2.814, abs=1e-12)


def test_first_step_equals_sgd_then_pulls_ahead():
    sgd, mom = bowl_walk(SGD(0.1), 2), bowl_walk(Momentum(0.1, 0.9), 2)
    assert mom[0] == pytest.approx(sgd[0])
    assert mom[1] > sgd[1]


def test_overshoots_then_settles():
    walk = bowl_walk(Momentum(0.1, 0.9), 400)
    assert max(walk) > 3.0
    assert walk[-1] == pytest.approx(3.0, abs=1e-6)


def test_beta_zero_is_exactly_sgd():
    assert bowl_walk(Momentum(0.1, 0.0), 30) == bowl_walk(SGD(0.1), 30)


def test_constant_gradient_velocity_tends_to_g_over_one_minus_beta():
    opt, p, g = Momentum(0.01, 0.9), [np.zeros(2)], [np.array([1.0, -2.0])]
    for _ in range(200):
        p = opt.step(p, g)
    assert opt.velocities is not None
    assert np.allclose(opt.velocities[0], [10.0, -20.0], rtol=1e-6)


def test_separate_velocity_per_parameter_and_hand_update():
    opt = Momentum(0.5, 0.5)
    params = [np.array([1.0, 2.0]), np.array([[0.0]])]
    grads = [np.array([1.0, 0.0]), np.array([[2.0]])]
    new = opt.step(params, grads)
    assert np.array_equal(new[0], [0.5, 2.0]) and np.array_equal(new[1], [[-1.0]])
    new2 = opt.step(new, grads)  # v = 0.5 * g + g = 1.5 g
    assert np.array_equal(new2[0], [0.5 - 0.75, 2.0]) and np.array_equal(new2[1], [[-1.0 - 1.5]])


def test_inputs_not_modified():
    p, g = np.array([1.0, 2.0]), np.array([0.5, 0.5])
    p0, g0 = p.copy(), g.copy()
    opt = Momentum(0.1)
    opt.step([p], [g])
    opt.step([p], [g])
    assert np.array_equal(p, p0) and np.array_equal(g, g0)


def test_reset_restores_first_step_behaviour():
    opt = Momentum(0.1, 0.9)
    first = bowl_walk(opt, 5)
    opt.reset()
    assert bowl_walk(opt, 5) == first


def test_optimizers_do_not_share_state_and_are_deterministic():
    a, b = Momentum(0.1, 0.9), Momentum(0.1, 0.9)
    bowl_walk(a, 3)
    assert b.velocities is None
    assert bowl_walk(b, 10) == bowl_walk(Momentum(0.1, 0.9), 10)


def test_shape_change_requires_reset():
    opt = Momentum(0.1)
    opt.step([np.zeros(2)], [np.ones(2)])
    with pytest.raises(ValueError):
        opt.step([np.zeros(3)], [np.ones(3)])
    with pytest.raises(ValueError):
        opt.step([np.zeros(2), np.zeros(1)], [np.ones(2), np.ones(1)])


def test_bad_arguments_rejected():
    for lr in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            Momentum(lr)
    for beta in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            Momentum(0.1, beta)
    with pytest.raises(ValueError):
        Momentum(0.1).step([np.zeros(2)], [np.zeros(3)])


def test_faster_than_sgd_in_a_narrow_valley():
    """f = 0.5 (w1^2 + 100 w2^2): steep in w2, shallow in w1, so SGD needs a small step."""
    scale = np.array([1.0, 100.0])

    def distance(opt) -> float:
        w = [np.array([1.0, 1.0])]
        for _ in range(100):
            w = opt.step(w, [scale * w[0]])
        return float(np.linalg.norm(w[0]))

    assert distance(Momentum(0.009, 0.9)) < 0.1 * distance(SGD(0.009))
