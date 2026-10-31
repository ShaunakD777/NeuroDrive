"""Tests for core/optim/__init__.py: building optimizers by name."""

import pytest

from neurodrive.core.optim import OPTIMIZER_SETTINGS, OPTIMIZERS, make_optimizer
from neurodrive.core.optim.adam import Adam
from neurodrive.core.optim.momentum import Momentum
from neurodrive.core.optim.sgd import SGD


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_every_name_builds_its_optimizer(name):
    opt = make_optimizer(name, 0.01)
    assert isinstance(opt, OPTIMIZERS[name])
    assert opt.learning_rate == 0.01


def test_settings_are_passed_on():
    opt = make_optimizer("adam", 0.002, {"beta1": 0.8, "eps": 1e-6})
    assert isinstance(opt, Adam)
    assert (opt.beta1, opt.beta2, opt.eps) == (0.8, 0.999, 1e-6)
    mom = make_optimizer("momentum", 0.1, {"beta": 0.5})
    assert isinstance(mom, Momentum) and mom.beta == 0.5
    assert isinstance(make_optimizer("sgd", 0.1, {}), SGD)


def test_settings_list_matches_the_constructors():
    assert set(OPTIMIZER_SETTINGS) == set(OPTIMIZERS)


def test_unknown_name_and_setting_refused():
    with pytest.raises(ValueError, match="unknown optimizer"):
        make_optimizer("adma", 0.1)
    with pytest.raises(ValueError, match="no setting"):
        make_optimizer("sgd", 0.1, {"beta": 0.9})
    with pytest.raises(ValueError, match="no setting"):
        make_optimizer("adam", 0.1, {"rho": 0.9})
    with pytest.raises(ValueError):
        make_optimizer("adam", 0.1, {"beta1": 1.5})
    with pytest.raises(ValueError):
        make_optimizer("rmsprop", -0.1)
