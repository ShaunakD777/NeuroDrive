"""Tests for GA runs (task 7.5, ADR-056): the config, a short run folder with one line of
``extra/population.jsonl`` per generation, an exact resume (extra lines included), and the
command line."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from neurodrive.cli import app
from neurodrive.env.track import Track
from neurodrive.experiment.config import (
    ConfigError,
    GARunConfig,
    load_config,
    parse_config,
    to_ga_settings,
)
from neurodrive.experiment.runner import learner_from_config, sweep_seeds, train_ga_run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.evolution.ga import (
    BreedingSettings,
    GALearner,
    GASettings,
    evaluate_genomes,
)

REPO = Path(__file__).resolve().parents[2]
TRAIN = Track.from_json(REPO / "tracks" / "train_loop_01.json")
runner = CliRunner()


def config_data(**overrides) -> dict:
    """A small GA config: 5 networks (1 elite), 1 start, 1 lap, 3 generations, exams off."""
    data = {
        "method": "ga",
        "tag": "small",
        "population": {"size": 5, "elite": 1},
        "variation": {"mutation_rate": 0.1, "mutation_strength": 0.2},
        "fitness": {"start_fractions": [0.0], "lateral_offsets": [0.0], "target_laps": 1},
        "budget": {"units": 3},
        "exams": False,
    }
    return {**data, **overrides}


def metrics_of(store: RunStore) -> list[dict]:
    return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]


# ── the config ──────────────────────────────────────────────────────────────────────────────


def test_the_shipped_config_is_the_briefs_choice():
    config = load_config(REPO / "configs" / "ga.yaml")
    assert isinstance(config, GARunConfig) and config.budget.units == 100
    settings = to_ga_settings(config)
    assert settings.population == 50 and settings.starts_per_network == 10
    assert settings.breeding == BreedingSettings(
        elite=2,
        selection="tournament",
        tournament_size=3,
        crossover_rate=0.5,
        mutation_rate=0.02,
        mutation_strength=0.05,
    )
    assert settings.breeding == BreedingSettings()  # the learner's defaults are the same values
    assert load_config(REPO / "configs" / "ga.yaml", units=7).budget.units == 7


def test_a_ga_config_gets_its_defaults():
    config = parse_config({"method": "ga", "budget": {"units": 2}})
    settings = to_ga_settings(config)
    assert settings == GASettings()  # population 50, the exam's starts, 2 laps, D3B breeding
    assert config.fitness.track == "train_loop_01" and config.exams is True


@pytest.mark.parametrize(
    "change, message",
    [
        ({"population": {"size": 4, "elite": 4}}, "must be smaller than size"),
        ({"population": {"size": 1}}, "population.size"),
        ({"selection": {"method": "roulette"}}, "unknown selection 'roulette'"),
        ({"selection": {"rank_pressure": 3.0}}, "selection.rank_pressure"),
        ({"variation": {"mutation_rate": 1.5}}, "variation.mutation_rate"),
        ({"variation": {"mutation_strength": -0.1}}, "variation.mutation_strength"),
        ({"variation": {"sigma": 0.1}}, "unknown field"),
        ({"fitness": {"track": "hairpins_01"}}, "held-out track"),
        ({"budget": {}}, "give at least one of"),
    ],
)
def test_bad_ga_configs_are_refused(change, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(config_data(**change))


def test_epochs_are_not_for_the_ga():
    with pytest.raises(ConfigError, match="--epochs is for toy and imitation configs"):
        parse_config(config_data(), epochs=3)
    assert parse_config(config_data(), units=9).budget.units == 9


# ── a short run ─────────────────────────────────────────────────────────────────────────────


def test_a_short_run_writes_one_population_line_per_generation(tmp_path):
    result = train_ga_run(parse_config(config_data(exams=True, budget={"units": 2})), tmp_path)
    store = result.store
    assert result.status == "completed" and result.units_done == 2
    assert store.path.name.endswith("-ga-small")
    metrics, lines = store.read_metrics(), store.read_extra_lines("population")
    assert [m["unit"] for m in metrics] == [1, 2] == [line["unit"] for line in lines]
    assert all(len(line["fitness"]) == 5 for line in lines)
    assert [line["elite"] for line in lines] == [[], [0]]
    assert metrics[1]["elite_drift"] == 0.0
    assert 0 < metrics[0]["env_steps_total"] < metrics[1]["env_steps_total"]
    assert metrics[0]["generation_best_fitness"] == max(lines[0]["fitness"])
    assert result.best_fitness == metrics[-1]["best_fitness"]
    # unit 0, unit 1 (a new best: the first), and unit 2 (the last); each with an exam card
    assert store.checkpoint_units() == [0, 1, 2] == store.exam_units()
    learner = learner_from_config(store.read_config())
    assert isinstance(learner, GALearner) and learner.settings.population == 5


def test_every_scored_generation_is_saved(tmp_path):
    """Task 7.6 (ADR-057): extra/generations/unit_<g>.npz is the matrix generation g was scored
    with, so scoring it again gives that unit's population line exactly."""
    config = parse_config(config_data())
    store = train_ga_run(config, tmp_path).store
    settings = to_ga_settings(config)
    lines = store.read_extra_lines("population")
    assert store.unit_array_names() == ["generations"]
    assert store.unit_array_units("generations") == [1, 2, 3]
    for line in lines:
        genomes = store.load_unit_arrays("generations", line["unit"])["genomes"]
        assert genomes.shape == (5, settings.spec.n_parameters) and genomes.dtype == np.float64
        scored = evaluate_genomes(genomes, TRAIN, settings)
        assert scored.fitness.tolist() == line["fitness"]  # bit for bit
        assert scored.best == line["champion"]
    # with E = 1 row 0 of generation g + 1 is generation g's champion, copied exactly
    for before, after in zip(lines, lines[1:], strict=False):
        old = store.load_unit_arrays("generations", before["unit"])["genomes"]
        new = store.load_unit_arrays("generations", after["unit"])["genomes"]
        assert np.array_equal(new[0], old[before["champion"]])
    # and generation 1's file is the learner's first draw (same seed, same matrix)
    first = learner_from_config(store.read_config()).genomes
    assert np.array_equal(store.load_unit_arrays("generations", 1)["genomes"], first)


def test_a_failed_run_resumes_exactly_extra_lines_included(tmp_path, monkeypatch):
    config = parse_config(config_data(budget={"units": 4}))
    reference = train_ga_run(config, tmp_path / "a").store
    original = GALearner.train_unit

    def fails_in_generation_3(self):
        if self.units == 2:
            raise RuntimeError("the process died")
        return original(self)

    monkeypatch.setattr(GALearner, "train_unit", fails_in_generation_3)
    with pytest.raises(RuntimeError):
        train_ga_run(config, tmp_path / "b")
    monkeypatch.undo()
    (folder,) = [p for p in (tmp_path / "b").iterdir() if p.is_dir()]
    store = RunStore.open(folder)
    assert store.read_meta()["status"] == "failed" and len(store.read_metrics()) == 2
    # a line that a kill left behind after the saved state must be cut away on resume
    store.append_extra_line("population", {"unit": 3, "fitness": [9.9]})
    store.save_unit_arrays("generations", 3, {"genomes": np.zeros((5, 3))})
    answer = runner.invoke(app, ["resume", "--run", str(folder)])
    assert answer.exit_code == 0, answer.output
    assert "status: completed after 4 units (2 were kept)" in answer.output
    assert "best fitness:" in answer.output
    assert metrics_of(store) == metrics_of(reference)
    assert store.read_extra_lines("population") == reference.read_extra_lines("population")
    assert store.unit_array_units("generations") == [1, 2, 3, 4]
    for unit in range(1, 5):
        mine = store.load_unit_arrays("generations", unit)["genomes"]
        assert np.array_equal(mine, reference.load_unit_arrays("generations", unit)["genomes"])
    assert store.checkpoint_units() == reference.checkpoint_units()
    for unit in store.checkpoint_units():
        assert store.load_checkpoint(unit).equals(reference.load_checkpoint(unit))
    (state_a, _), (state_b, _) = store.load_state(), reference.load_state()
    assert state_a.info == state_b.info
    assert all(np.array_equal(state_a.arrays[k], state_b.arrays[k]) for k in state_b.arrays)


def test_the_same_config_twice_gives_the_same_run(tmp_path):
    config = parse_config(config_data())
    a = train_ga_run(config, tmp_path / "a").store
    b = train_ga_run(config, tmp_path / "b").store
    assert metrics_of(a) == metrics_of(b)
    assert a.read_extra_lines("population") == b.read_extra_lines("population")
    other = train_ga_run(parse_config(config_data(seed=1)), tmp_path / "c").store
    assert other.read_extra_lines("population") != a.read_extra_lines("population")


# ── the command line ────────────────────────────────────────────────────────────────────────


def test_nd_train_and_sweep(tmp_path):
    config_file = tmp_path / "small.yaml"
    config_file.write_text(yaml.safe_dump(config_data()), encoding="utf-8")
    answer = runner.invoke(
        app,
        ["train", "--config", str(config_file), "--units", "2", "--runs-dir", str(tmp_path / "r")],
    )
    assert answer.exit_code == 0, answer.output
    assert "-ga-small" in answer.output and "status: completed after 2 generations" in answer.output
    assert "best fitness:" in answer.output
    refused = runner.invoke(app, ["train", "--config", str(config_file), "--epochs", "2"])
    assert refused.exit_code == 2 and "--epochs is for toy and imitation" in refused.output
    sweep = runner.invoke(
        app,
        ["sweep", "--config", str(config_file), "--seeds", "0-1", "--units", "1"]
        + ["--runs-dir", str(tmp_path / "s")],
    )
    assert sweep.exit_code == 0, sweep.output
    assert "-ga-small" in sweep.output and "best fitness: median" in sweep.output


def test_a_sweep_tags_every_seed(tmp_path):
    _, results = sweep_seeds(parse_config(config_data(budget={"units": 1})), [0, 1], tmp_path)
    assert [r.store.read_config()["tag"] for r in results] == ["small-s0", "small-s1"]
    assert all(len(r.store.read_extra_lines("population")) == 1 for r in results)
