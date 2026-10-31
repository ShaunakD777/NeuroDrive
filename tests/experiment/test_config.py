"""Tests for experiment/config.py: the toy config file (config v1, ADR-032)."""

import inspect
from pathlib import Path

import pytest
import yaml

from neurodrive.core.optim import OPTIMIZER_SETTINGS, OPTIMIZERS
from neurodrive.experiment.config import (
    OPTIMIZER_SECTIONS,
    ConfigError,
    load_config,
    parse_config,
    resolved,
    resolved_from_toy_config,
    to_toy_config,
    toy_config_from_run,
)
from neurodrive.learn.supervised.train_toy import DEFAULTS, ToyConfig

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def minimal(**changes) -> dict:
    """The smallest valid toy config (XOR), with top-level ``changes``."""
    data = {
        "method": "toy",
        "task": "xor",
        "network": {"layer_sizes": [2, 4, 1]},
        "training": {"epochs": 10},
        "optimizer": {"name": "sgd", "learning_rate": 1.0},
    }
    data.update(changes)
    return data


def problems(data: dict) -> list[str]:
    with pytest.raises(ConfigError) as info:
        parse_config(data)
    return info.value.problems


@pytest.mark.parametrize("task", sorted(DEFAULTS))
def test_shipped_configs_load_and_equal_the_task_defaults(task):
    config = load_config(CONFIGS / f"toy_{task}.yaml")
    assert to_toy_config(config) == ToyConfig.for_task(task)


def test_minimal_config_fills_defaults():
    toy = to_toy_config(parse_config(minimal()))
    assert (toy.seed, toy.tag, toy.loss, toy.init_scheme) == (0, "default", "mse", "xavier")
    assert (toy.batch_size, toy.val_fraction) == (None, 0.0)
    assert (toy.hidden_activation, toy.output_activation) == ("tanh", "linear")


def test_optimizer_settings_reach_the_learner_with_defaults_filled():
    config = parse_config(minimal(optimizer={"name": "adam", "learning_rate": 0.01, "beta1": 0.8}))
    toy = to_toy_config(config)
    assert toy.optimizer == "adam" and toy.learning_rate == 0.01
    assert toy.optimizer_settings == {"beta1": 0.8, "beta2": 0.999, "eps": 1e-8}


def test_optimizer_sections_match_the_optimizers():
    """Each section's settings and defaults are exactly the optimizer constructor's."""
    assert set(OPTIMIZER_SECTIONS) == set(OPTIMIZERS)
    for name, section in OPTIMIZER_SECTIONS.items():
        fields = section.model_fields
        settings = set(fields) - {"name", "learning_rate"}
        assert settings == set(OPTIMIZER_SETTINGS[name])
        params = inspect.signature(OPTIMIZERS[name]).parameters
        for s in settings:
            assert fields[s].default == params[s].default


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"learning_rat": 0.1}, "learning_rat: unknown field"),
        (
            {"optimizer": {"name": "adma", "learning_rate": 0.1}},
            "optimizer.name: unknown optimizer",
        ),
        ({"optimizer": {"learning_rate": 0.1}}, "optimizer.name: Field required"),
        ({"optimizer": {"name": "sgd", "learning_rate": -0.1}}, "optimizer.learning_rate"),
        ({"optimizer": {"name": "sgd", "learning_rate": 0.1, "beta1": 0.9}}, "optimizer.beta1"),
        ({"optimizer": {"name": "adam", "learning_rate": 0.1, "beta2": 1.0}}, "optimizer.beta2"),
        ({"network": {"layer_sizes": [2, 4, 1], "hidden_activation": "tahn"}}, "tahn"),
        ({"network": {"layer_sizes": [3, 4, 1]}}, "do not fit the xor data"),
        ({"network": {"layer_sizes": [2]}}, "network.layer_sizes"),
        ({"training": {"epochs": 0}}, "training.epochs"),
        ({"training": {"epochs": 2.5}}, "training.epochs"),
        ({"training": {"epochs": 10, "batch_size": 0}}, "training.batch_size"),
        ({"training": {"epochs": 10, "val_fraction": 1.0}}, "training.val_fraction"),
        ({"training": {"epochs": 10, "loss": "hinge"}}, "unknown loss"),
        ({"task": "circle"}, "unknown toy task"),
        ({"method": "evolution"}, "method"),
        ({"seed": -1}, "seed"),
        ({"tag": "my run"}, "tag"),
    ],
)
def test_bad_configs_name_the_field(change, expected):
    found = problems(minimal(**change))
    assert any(expected in p for p in found), found


def test_every_problem_is_listed_at_once():
    found = problems(minimal(seed=-1, tag="a b", training={"epochs": 0}))
    assert len(found) == 3


def test_missing_sections_are_reported():
    found = problems({"method": "toy", "task": "xor"})
    assert {p.split(":")[0] for p in found} == {"network", "training", "optimizer"}


def test_overrides():
    config = parse_config(minimal(), seed=7, tag="sweep_a", epochs=99)
    assert (config.seed, config.tag, config.training.epochs) == (7, "sweep_a", 99)
    with pytest.raises(ConfigError, match="training.epochs"):
        parse_config(minimal(), epochs=0)


def test_resolved_round_trips_through_yaml():
    config = parse_config(minimal(optimizer={"name": "rmsprop", "learning_rate": 0.01}))
    data = yaml.safe_load(yaml.safe_dump(resolved(config), sort_keys=False))
    assert data["optimizer"] == {"name": "rmsprop", "learning_rate": 0.01, "rho": 0.9, "eps": 1e-8}
    assert parse_config(data) == config
    assert list(data) == ["method", "task", "seed", "tag", "network", "training", "optimizer"]


@pytest.mark.parametrize("task", sorted(DEFAULTS))
def test_resolved_from_toy_config_is_the_inverse(task):
    toy = ToyConfig.for_task(task, optimizer="momentum", optimizer_settings={"beta": 0.5})
    assert to_toy_config(parse_config(resolved_from_toy_config(toy))) == toy


def test_phase3_flat_run_configs_still_read():
    flat = ToyConfig.for_task("sine", learning_rate=0.1, tag="old").to_dict()
    for key in ("optimizer", "optimizer_settings", "batch_size", "val_fraction"):
        del flat[key]
    expected = ToyConfig.for_task("sine", learning_rate=0.1, tag="old")
    assert toy_config_from_run(flat) == expected
    assert to_toy_config(parse_config(flat)) == expected  # nd train accepts it too
    assert toy_config_from_run(resolved_from_toy_config(expected)) == expected


def test_file_errors(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("method: toy\ntask: [xor\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML at line"):
        load_config(bad_yaml)
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(empty)
    listing = tmp_path / "list.yaml"
    listing.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(listing)
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
