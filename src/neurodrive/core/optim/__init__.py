"""Optimizers: the rules that turn gradients into changes of the parameters.

Like the rest of ``core``, this package imports only NumPy and the standard library (and the
network types of ``core.nn``).

``make_optimizer`` builds an optimizer from its name, so configs can choose one by writing
``sgd``, ``momentum``, ``rmsprop`` or ``adam``. ``OPTIMIZER_SETTINGS`` lists, for each name, the
settings besides the learning rate that it accepts.
"""

from collections.abc import Mapping

from neurodrive.core.optim.adam import Adam
from neurodrive.core.optim.base import Optimizer
from neurodrive.core.optim.momentum import Momentum
from neurodrive.core.optim.rmsprop import RMSProp
from neurodrive.core.optim.sgd import SGD

OPTIMIZERS: dict[str, type] = {"sgd": SGD, "momentum": Momentum, "rmsprop": RMSProp, "adam": Adam}

OPTIMIZER_SETTINGS: dict[str, tuple[str, ...]] = {
    "sgd": (),
    "momentum": ("beta",),
    "rmsprop": ("rho", "eps"),
    "adam": ("beta1", "beta2", "eps"),
}


def make_optimizer(
    name: str, learning_rate: float, settings: Mapping[str, float] | None = None
) -> Optimizer:
    """A new optimizer called ``name`` with learning rate η = ``learning_rate``.

    ``settings`` holds that optimizer's other settings (for example ``{"beta1": 0.8}`` for Adam);
    missing ones take their defaults. An unknown name, a setting the optimizer does not have, or
    an out-of-range value raises ``ValueError``.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"unknown optimizer {name!r}; valid names: {', '.join(OPTIMIZERS)}")
    settings = dict(settings or {})
    unknown = sorted(set(settings) - set(OPTIMIZER_SETTINGS[name]))
    if unknown:
        allowed = ", ".join(OPTIMIZER_SETTINGS[name]) or "none"
        raise ValueError(f"{name} has no setting(s) {', '.join(unknown)}; allowed: {allowed}")
    return OPTIMIZERS[name](learning_rate, **settings)
