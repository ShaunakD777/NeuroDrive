"""Tests for learn/supervised/toy_tasks.py: shapes, exact targets, seeded determinism."""

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.learn.supervised.toy_tasks import (
    LINE_INTERCEPT,
    LINE_SLOPE,
    TOY_TASKS,
    line,
    make_toy_dataset,
    noisy_sine,
    sine,
    two_spirals,
    xor,
)


@pytest.mark.parametrize("name", sorted(TOY_TASKS))
def test_shapes_and_float_arrays(name):
    d = make_toy_dataset(name, make_rng(0))
    assert d.name == name
    assert d.x.ndim == 2 and d.y.ndim == 2 and d.x.shape[0] == d.y.shape[0]
    assert d.x.dtype == np.float64 and d.y.dtype == np.float64
    assert np.all(np.isfinite(d.x)) and np.all(np.isfinite(d.y))


@pytest.mark.parametrize("name", sorted(TOY_TASKS))
def test_same_seed_same_data(name):
    a, b = make_toy_dataset(name, make_rng(3)), make_toy_dataset(name, make_rng(3))
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)


@pytest.mark.parametrize("name", ["line", "sine", "two_spirals", "noisy_sine"])
def test_different_seed_different_data(name):
    a, b = make_toy_dataset(name, make_rng(0)), make_toy_dataset(name, make_rng(1))
    assert not np.array_equal(a.x, b.x)


def test_line_targets_are_exactly_on_the_line():
    d = line(make_rng(0), n=50)
    assert d.x.shape == (50, 1) and np.all(np.abs(d.x) <= 1.0)
    np.testing.assert_array_equal(d.y, LINE_SLOPE * d.x + LINE_INTERCEPT)
    assert (LINE_SLOPE, LINE_INTERCEPT) == (2.0, -0.5)


def test_sine_targets_and_range():
    d = sine(make_rng(0))
    assert d.x.shape == (128, 1) and np.all(np.abs(d.x) <= np.pi)
    np.testing.assert_allclose(d.y, np.sin(d.x), atol=0)


def test_xor_is_the_four_corners():
    d = xor()
    np.testing.assert_array_equal(d.x, [[0, 0], [0, 1], [1, 0], [1, 1]])
    np.testing.assert_array_equal(d.y[:, 0], [0, 1, 1, 0])
    np.testing.assert_array_equal(d.y[:, 0], np.logical_xor(d.x[:, 0], d.x[:, 1]))


def test_two_spirals_balanced_and_mirrored():
    d = two_spirals(make_rng(0), n_per_class=40, noise=0.0)
    assert d.x.shape == (80, 2) and d.y.shape == (80, 1)
    assert d.y.sum() == 40  # half are class 1
    class0, class1 = d.x[d.y[:, 0] == 0], d.x[d.y[:, 0] == 1]
    np.testing.assert_allclose(class1, -class0)  # class 1 is class 0 turned by 180°
    radius = np.linalg.norm(class0, axis=1)
    assert np.all(np.diff(radius) > 0) and radius.max() == pytest.approx(1.0)


def test_unknown_task_lists_valid_names():
    with pytest.raises(ValueError, match="unknown toy task 'circle'.*line.*sine"):
        make_toy_dataset("circle", make_rng(0))


def test_noisy_sine_is_sine_plus_noise_of_the_stated_size():
    d = noisy_sine(make_rng(3), n=4000)
    assert d.x.shape == (4000, 1) and d.y.shape == (4000, 1)
    assert np.all(np.abs(d.x) <= np.pi)
    residual = d.y - np.sin(d.x)
    assert abs(float(residual.mean())) < 0.02
    assert float(residual.std()) == pytest.approx(0.2, rel=0.05)
    clean = noisy_sine(make_rng(3), n=10, noise=0.0)
    np.testing.assert_allclose(clean.y, np.sin(clean.x))
    assert noisy_sine(make_rng(0)).x.shape == (64, 1)  # the default size
