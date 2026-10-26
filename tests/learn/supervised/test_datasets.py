"""Tests for learn/supervised/datasets.py: each example once per epoch, seeded order, splits."""

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.learn.supervised.datasets import MinibatchLoader, train_val_split


def numbered(n: int):
    """x = 0..n-1 as a column and y = 10 * x, so each row's pairing is easy to check."""
    x = np.arange(n, dtype=float).reshape(n, 1)
    return x, 10.0 * x


def epoch_rows(loader):
    return [b[0][:, 0] for b in loader]


@pytest.mark.parametrize("n,b", [(10, 4), (10, 1), (10, 10), (10, 25), (7, 3), (1, 1), (64, 8)])
def test_every_example_once_per_epoch(n, b):
    x, y = numbered(n)
    loader = MinibatchLoader(x, y, b, make_rng(0))
    for _ in range(3):
        seen = np.concatenate(epoch_rows(loader))
        assert sorted(seen) == list(range(n))


def test_batch_sizes_and_length():
    x, y = numbered(10)
    loader = MinibatchLoader(x, y, 4, make_rng(0))
    assert len(loader) == 3
    assert [len(b) for b in epoch_rows(loader)] == [4, 4, 2]


def test_drop_last():
    x, y = numbered(10)
    loader = MinibatchLoader(x, y, 4, make_rng(0), drop_last=True)
    assert len(loader) == 2
    assert [len(b) for b in epoch_rows(loader)] == [4, 4]
    seen = np.concatenate(epoch_rows(loader))
    assert len(set(seen)) == 8  # no repeats; two examples are left out this epoch


def test_x_and_y_stay_paired():
    x, y = numbered(20)
    for xb, yb in MinibatchLoader(x, y, 6, make_rng(3)):
        assert np.array_equal(yb, 10.0 * xb)


def test_same_seed_same_orders_across_epochs():
    x, y = numbered(30)
    a, b = MinibatchLoader(x, y, 7, make_rng(5)), MinibatchLoader(x, y, 7, make_rng(5))
    for _ in range(4):
        for ra, rb in zip(epoch_rows(a), epoch_rows(b), strict=True):
            assert np.array_equal(ra, rb)


def test_different_seed_different_order():
    x, y = numbered(30)
    a = np.concatenate(epoch_rows(MinibatchLoader(x, y, 7, make_rng(1))))
    b = np.concatenate(epoch_rows(MinibatchLoader(x, y, 7, make_rng(2))))
    assert not np.array_equal(a, b)


def test_epochs_differ_from_each_other():
    x, y = numbered(30)
    loader = MinibatchLoader(x, y, 7, make_rng(1))
    first = np.concatenate(epoch_rows(loader))
    second = np.concatenate(epoch_rows(loader))
    assert not np.array_equal(first, second)


def test_no_shuffle_keeps_order():
    x, y = numbered(9)
    loader = MinibatchLoader(x, y, 4, make_rng(0), shuffle=False)
    for _ in range(2):
        assert np.array_equal(np.concatenate(epoch_rows(loader)), np.arange(9))


def test_loader_does_not_modify_data():
    x, y = numbered(12)
    x0, y0 = x.copy(), y.copy()
    list(MinibatchLoader(x, y, 5, make_rng(0)))
    assert np.array_equal(x, x0) and np.array_equal(y, y0)


@pytest.mark.parametrize("bad", [0, -2])
def test_bad_batch_size(bad):
    x, y = numbered(5)
    with pytest.raises(ValueError):
        MinibatchLoader(x, y, bad, make_rng(0))


def test_bad_batch_size_type_and_shapes():
    x, y = numbered(5)
    with pytest.raises(TypeError):
        MinibatchLoader(x, y, 2.0, make_rng(0))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MinibatchLoader(x, y[:4], 2, make_rng(0))
    with pytest.raises(ValueError):
        MinibatchLoader(x[:, 0], y, 2, make_rng(0))
    with pytest.raises(ValueError):
        MinibatchLoader(x[:0], y[:0], 2, make_rng(0))
    with pytest.raises(ValueError):
        MinibatchLoader(x, y, 9, make_rng(0), drop_last=True)


def test_split_sizes_cover_and_disjoint():
    x, y = numbered(10)
    s = train_val_split(x, y, 0.3, make_rng(0))
    assert len(s.val_idx) == 3 and len(s.train_idx) == 7
    assert s.x_val.shape == (3, 1) and s.x_train.shape == (7, 1)
    assert set(s.val_idx).isdisjoint(s.train_idx)
    assert sorted(np.concatenate([s.train_idx, s.val_idx])) == list(range(10))
    assert np.array_equal(s.x_train[:, 0], s.train_idx)
    assert np.array_equal(s.y_val, 10.0 * s.x_val)


def test_split_reproducible_and_seed_dependent():
    x, y = numbered(50)
    a = train_val_split(x, y, 0.2, make_rng(4))
    b = train_val_split(x, y, 0.2, make_rng(4))
    c = train_val_split(x, y, 0.2, make_rng(5))
    assert np.array_equal(a.val_idx, b.val_idx)
    assert not np.array_equal(a.val_idx, c.val_idx)


def test_split_zero_fraction_and_input_untouched():
    x, y = numbered(8)
    x0 = x.copy()
    s = train_val_split(x, y, 0.0, make_rng(0))
    assert s.x_val.shape == (0, 1) and len(s.train_idx) == 8
    assert np.array_equal(x, x0)


@pytest.mark.parametrize("f", [-0.1, 1.0, 1.5])
def test_split_bad_fraction(f):
    x, y = numbered(10)
    with pytest.raises(ValueError):
        train_val_split(x, y, f, make_rng(0))


def test_split_fraction_rounding_to_zero_rejected():
    x, y = numbered(10)
    with pytest.raises(ValueError):
        train_val_split(x, y, 0.01, make_rng(0))
