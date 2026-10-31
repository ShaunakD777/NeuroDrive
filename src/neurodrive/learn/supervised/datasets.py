"""Minibatches and train/validation splits: how a learner reads its data a few examples at a time.

Words used here:

* An *epoch* is one pass through all the training examples.
* A *minibatch* is a small group of examples used for one learning step. With N examples and a
  batch size B, one epoch has K = ceil(N / B) minibatches (the last one is smaller when B does not
  divide N, unless ``drop_last`` is set).
* A *validation set* is a part of the data the network never learns from. Its loss shows whether
  the network has learned the pattern or only memorised the training examples.

Shuffling uses the generator passed in, so the same seed always gives the same orders (CLAUDE.md
§4). Callers pass a child stream (for example ``child_rng(rng, "batches")``) so that the batch
order never shifts the numbers used for the data or the starting weights.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrive.core.rng import Rng

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _check_xy(x: Array, y: Array) -> tuple[Array, Array]:
    """Turn inputs and targets into float arrays and check they have one row per example."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be 2-D (examples, values); got {x.shape} and {y.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x has {x.shape[0]} rows but y has {y.shape[0]}")
    if x.shape[0] == 0:
        raise ValueError("there are no examples")
    return x, y


@dataclass(frozen=True)
class DataSplit:
    """A dataset cut in two: training and validation arrays, and the original row numbers."""

    x_train: Array
    y_train: Array
    x_val: Array
    y_val: Array
    train_idx: IntArray
    val_idx: IntArray


def train_val_split(x: Array, y: Array, val_fraction: float, rng: Rng) -> DataSplit:
    """Randomly set aside a fraction of the examples for validation.

    With N examples and ``val_fraction`` f, the validation set has n_val = round(f * N) examples
    and the training set has N - n_val (for N = 10 and f = 0.3: 3 and 7).

    First a random shuffle of the row numbers 0..N-1 is drawn; the first n_val of them become the
    validation rows and the rest the training rows. ``x`` and ``y`` are not modified. f must lie
    in [0, 1); f = 0 gives an empty validation set. A positive f that would round to zero
    validation examples, or leave no training example, is an error.
    """
    x, y = _check_xy(x, y)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    n = x.shape[0]
    n_val = int(round(val_fraction * n))
    if val_fraction > 0.0 and n_val == 0:
        raise ValueError(f"val_fraction {val_fraction} of {n} examples rounds to zero")
    if n_val >= n:
        raise ValueError(f"val_fraction {val_fraction} leaves no training example out of {n}")
    order = rng.permutation(n)
    val_idx, train_idx = order[:n_val], order[n_val:]
    return DataSplit(x[train_idx], y[train_idx], x[val_idx], y[val_idx], train_idx, val_idx)


class MinibatchLoader:
    """Deals the examples out in minibatches, in a new random order every epoch.

    Iterating over the loader once is one epoch and yields pairs ``(x_batch, y_batch)``. Every
    example appears exactly once per epoch (or, with ``drop_last``, all but the few left over from
    a short final batch). Each epoch draws a fresh shuffle from the loader's generator, so two
    loaders built from the same seed give the same sequence of orders, while epoch 2 differs from
    epoch 1. With ``shuffle=False`` the original order is kept every epoch.
    """

    def __init__(
        self,
        x: Array,
        y: Array,
        batch_size: int,
        rng: Rng,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.x, self.y = _check_xy(x, y)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int | np.integer):
            raise TypeError(f"batch_size must be an int, got {type(batch_size).__name__}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        if drop_last and batch_size > self.x.shape[0]:
            raise ValueError(f"drop_last with batch_size {batch_size} > {self.x.shape[0]} examples")
        self.batch_size = int(batch_size)
        self.rng = rng
        self.shuffle = shuffle
        self.drop_last = drop_last

    @property
    def n_examples(self) -> int:
        return int(self.x.shape[0])

    def __len__(self) -> int:
        """K, the number of minibatches in one epoch."""
        n, b = self.n_examples, self.batch_size
        return n // b if self.drop_last else -(-n // b)

    def epoch_order(self) -> IntArray:
        """The row order for the next epoch: a fresh shuffle (uses up random numbers) or 0..N-1."""
        if self.shuffle:
            return self.rng.permutation(self.n_examples)
        return np.arange(self.n_examples)

    def __iter__(self) -> Iterator[tuple[Array, Array]]:
        order = self.epoch_order()
        for k in range(len(self)):
            rows = order[k * self.batch_size : (k + 1) * self.batch_size]
            yield self.x[rows], self.y[rows]
