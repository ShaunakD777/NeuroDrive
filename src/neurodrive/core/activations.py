"""Activation functions: the squashing step that turns a neuron's weighted sum into its output.

A neuron first adds up its weighted inputs into one number, the weighted sum ``z``. The
*activation function* ``f`` then turns ``z`` into the neuron's output ``a = f(z)``. Which function
is used decides the *range* of possible outputs, whatever the weights are:

* ``tanh``    : smooth S-curve, outputs between -1 and +1, zero when z is zero.
* ``sigmoid`` : smooth S-curve, outputs between 0 and 1, never negative.
* ``relu``    : 0 for negative z, z itself otherwise; outputs from 0 upwards.
* ``linear``  : leaves z unchanged; outputs are unlimited in both directions.

Every function takes an array of any shape and returns an array of the same shape (float64).
Derivatives are added in Phase 3, when learning needs them.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Activation = Callable[[ArrayLike], NDArray[np.float64]]


def tanh(z: ArrayLike) -> NDArray[np.float64]:
    """Hyperbolic tangent: (e^z - e^-z) / (e^z + e^-z), where e is Euler's number (about 2.718).

    ``z`` is the weighted sum. The result lies in [-1, 1]; it is -1 for very negative z, 0 at
    z = 0 and +1 for very positive z.
    """
    return np.tanh(np.asarray(z, dtype=np.float64))


def sigmoid(z: ArrayLike) -> NDArray[np.float64]:
    """Logistic sigmoid: 1 / (1 + e^-z), where ``z`` is the weighted sum.

    The result lies in [0, 1] and is 0.5 at z = 0. It is computed separately for positive and
    negative z (using e^-|z|, which is never huge) so that very large |z| cannot overflow.
    """
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))  # in (0, 1], so it never overflows
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def relu(z: ArrayLike) -> NDArray[np.float64]:
    """Rectified linear unit: the larger of 0 and ``z`` (the weighted sum).

    The result is 0 for negative z and equals z otherwise, so it lies in [0, infinity).
    """
    return np.maximum(np.asarray(z, dtype=np.float64), 0.0)


def linear(z: ArrayLike) -> NDArray[np.float64]:
    """Identity: returns ``z`` (the weighted sum) unchanged, so the range is unlimited."""
    return np.array(z, dtype=np.float64)


ACTIVATIONS: dict[str, Activation] = {
    "tanh": tanh,
    "sigmoid": sigmoid,
    "relu": relu,
    "linear": linear,
}


def get_activation(name: str) -> Activation:
    """Look an activation function up by its name ("tanh", "sigmoid", "relu" or "linear").

    Names (not function objects) are used so that a network spec can be saved in a file.
    Raises ``ValueError`` listing the valid names if ``name`` is unknown.
    """
    try:
        return ACTIVATIONS[name]
    except (KeyError, TypeError):
        valid = ", ".join(sorted(ACTIVATIONS))
        raise ValueError(f"unknown activation {name!r}; valid names: {valid}") from None
