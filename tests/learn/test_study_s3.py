"""Guards for Study S3 (docs/studies/S3.md): its measurements mean what the report says, and its
failure case still behaves as described."""

import importlib.util
import math
from pathlib import Path

import pytest

from neurodrive.core.rng import make_rng
from neurodrive.learn.supervised.train_toy import ToyLearner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_s3", ROOT / "scripts" / "study_s3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_study_run_is_a_plain_toy_learner_run(study, monkeypatch):
    monkeypatch.setattr(study, "EPOCHS", 40)
    result = study.run_sine(("adam", 0.01, 3))
    learner = ToyLearner(study.sine_config("adam", 0.01, 3), make_rng(3))
    losses = [learner.train_unit().metrics["loss"] for _ in range(40)]
    assert result["curve"][1:] == losses and result["final"] == losses[-1]
    assert result["curve"][0] == pytest.approx(
        ToyLearner(study.sine_config("adam", 0.01, 3), make_rng(3)).loss()
    )


def test_a_diverged_run_counts_as_infinity(study):
    result = study.run_sine(("sgd", 3.0, 0))
    assert result["final"] == math.inf and result["diverged_at"] is not None
    assert result["first_below"] is None


def test_best_rate_and_edges_by_hand(study):
    def cell(median, diverged):
        return {"final": {"median": median}, "diverged": diverged}

    cells = {}
    for o in study.OPTIMIZERS:
        for lr in study.LEARNING_RATES:
            cells[f"{o}|{lr}"] = cell(1.0, 0)
    cells["sgd|0.03"] = cell(0.005, 0)
    cells["sgd|0.1"] = cell(0.005, 0)  # a tie: the smaller rate wins
    cells["sgd|0.3"] = cell(math.inf, 4)
    cells["sgd|1.0"] = cell(math.inf, 5)
    assert study.best_learning_rates(cells)["sgd"] == 0.03
    edges = study.edges(cells)["sgd"]
    assert edges == {"first_any_diverged": 0.3, "first_all_diverged": 1.0, "usable": [0.03, 0.1]}


def test_overfit_measures(study, monkeypatch):
    monkeypatch.setattr(study, "OVERFIT_EPOCHS", 60)
    r = study.run_overfit(("small", 1))
    assert len(r["train"]) == len(r["val"]) == 61
    assert r["best_val"] == min(r["val"]) and r["val"][r["best_val_epoch"]] == r["best_val"]
    assert r["ratio"] == pytest.approx(r["final_val"] / r["best_val"])
    assert r["ratio"] >= 1.0


def test_the_failure_case_still_spikes_at_the_end(study):
    """Momentum at 0.1, test seed 5: learns to about 1e-4, then blows up near epoch 3000."""
    curve = study.run_sine(("momentum", 0.1, 5))["curve"]
    assert min(curve[2900:2985]) < 2e-4
    assert curve[-1] > 0.01
    assert all(b > a for a, b in zip(curve[2986:], curve[2987:], strict=False))  # rising to the end
