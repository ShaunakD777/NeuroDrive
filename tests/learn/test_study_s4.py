"""Guards for Study S4 (docs/studies/S4.md): the budget rule, B*, the hypothesis rules on a
hand-made set of run folders, and that a study run is a plain random-search run."""

import importlib.util
import json
from pathlib import Path

import pytest

from neurodrive.experiment.config import load_config, parse_config, resolved
from neurodrive.experiment.runner import train_random_search_run
from neurodrive.experiment.store import RunStore

ROOT = Path(__file__).resolve().parents[2]
HELDOUT = ("hairpins_01", "sweeper_01")


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_s4", ROOT / "scripts" / "study_s4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_budget_rule_on_the_briefs_worked_example(study):
    # E(6) = 140,000 ... E(9) = 205,000; the best held after unit 8 was found in unit 7.
    steps = [0, 20_000, 40_000, 60_000, 80_000, 100_000, 140_000, 160_000, 185_000, 205_000]
    best = [0, 1, 1, 1, 1, 1, 1, 7, 7, 7]
    assert study.best_unit_at(steps, best, 200_000) == 7  # u = 8, card of unit 7
    assert study.best_unit_at(steps, best, 150_000) == 1  # u = 6, an older best
    assert study.best_unit_at(steps, best, 185_000) == 7  # a budget equal to E(8) includes unit 8
    assert study.best_unit_at(steps, best, 19_999) == 0  # before unit 1: the unit-0 network
    with pytest.raises(ValueError):
        study.best_unit_at(steps, best, -1)


def test_b_star_skips_budget_0_and_needs_a_value_strictly_above(study):
    budgets, values = [0, 10, 20, 40], [0.3, 0.1, 0.2, 0.5]
    assert study.first_budget_above(budgets, values, 0.2) == 40  # 0.2 is not above 0.2
    assert study.first_budget_above(budgets, values, 0.05) == 10  # budget 0 never counts
    assert study.first_budget_above(budgets, values, 0.5) is None


def test_the_grid_runs_from_the_first_unit_to_the_smallest_final_total(study):
    records = [{"steps": [0, 120, 500, 900]}, {"steps": [0, 100, 300, 800]}]
    grid = study.budget_grid(records)
    assert grid[0] == 0 and grid[1] == 100 and grid[-1] == 800
    assert grid == sorted(set(grid)) and len(grid) <= study.GRID_POINTS + 1


# --- a hand-made study ----------------------------------------------------------------------


def card(progress: float, completion: float = 0.0, crash: float = 1.0) -> dict:
    """An exam card with only what the study reads: every held-out track scores alike."""
    aggregate = {"progress_frac_median": progress, "completion_rate": completion}
    summary = {**aggregate, "crash_rate": crash}
    tracks = [{"track": t, "split": "heldout", "aggregate": aggregate} for t in HELDOUT]
    return {"training": summary, "heldout": summary, "tracks": tracks, "policy": {}}


def fake_run(runs_dir: Path, method: str, seed, cards: dict, lines=(), tag="s4") -> None:
    config = {"method": method} if seed is None else {"method": method, "seed": seed}
    tag = f"{tag}-s{seed}" if seed is not None else tag
    store = RunStore.create(runs_dir, method, tag, config, extra_meta={"study": "S4"})
    for line in lines:
        store.append_metrics(line)
    (store.path / "exams").mkdir(exist_ok=True)
    for unit, c in cards.items():
        (store.path / "exams" / f"unit_{unit}.json").write_text(json.dumps(c), encoding="utf-8")
    store.update_meta(sweep="sw" if method == "random_search" else None, exam_wall_clock_s=1.0)
    store.finish("completed", "units")


def lines_of(best_units: list[int], fitness: list[float]) -> list[dict]:
    """Metrics lines of a 3-unit run: 100 environment steps per unit."""
    return [
        {
            "unit": u,
            "env_steps_total": 100 * u,
            "wall_clock_s": float(u),
            "best_fitness": f,
            "best_unit": b,
            "improved": b == u,
        }
        for u, (b, f) in enumerate(zip(best_units, fitness, strict=True), start=1)
    ]


@pytest.fixture(scope="module")
def made(tmp_path_factory, study) -> dict:
    """Seeds 0 and 1 overfit (their unit-3 best is 0.3 and 0.2 below their unit-1 best on
    held-out progress), seeds 2-4 improve. Final held-out progress: 0.3, 0.4, 0.5, 0.6, 0.7.
    Random networks score 0; the neuron 0.234 progress, 0.25 completion; the expert 1, 1."""
    runs = tmp_path_factory.mktemp("s4")
    first = [0.6, 0.6, 0.1, 0.1, 0.1]
    final = [0.3, 0.4, 0.5, 0.6, 0.7]
    for seed in study.SEEDS:
        cards = {0: card(0.0), 1: card(first[seed]), 3: card(final[seed], completion=0.1)}
        fake_run(runs, "random_search", seed, cards, lines_of([1, 1, 3], [0.2, 0.2, 0.5]))
    for seed in study.RANDOM_SEEDS:
        fake_run(runs, "random_network", seed, {0: card(0.0)}, tag="baseline")
    fake_run(runs, "hand_neuron", None, {0: card(0.234, 0.25, 0.75)}, tag="baseline")
    fake_run(runs, "scripted_expert", None, {0: card(1.0, 1.0, 0.0)}, tag="baseline")
    return study.analyse(runs)


def test_the_analysis_reads_the_hand_made_runs(made, study):
    assert made["final_budget"] == 300 and made["grid"][0] == 0
    assert made["curves"]["heldout_progress"][-1]["seeds"] == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert made["curves"]["heldout_progress"][-1]["median"] == 0.5
    assert made["table"]["heldout_progress"][0]["seeds"] == [0.0] * 5  # budget 0: unit 0
    # After unit 1 the seeds hold [0.6, 0.6, 0.1, 0.1, 0.1] (median 0.1): above the random
    # networks' 0 but not the neuron's 0.234; after unit 3 the median is 0.5.
    assert made["b_star"]["random_network"]["heldout_progress"] == 100
    assert made["b_star"]["hand_neuron"]["heldout_progress"] == 300
    assert made["b_star"]["scripted_expert"]["heldout_progress"] is None
    assert len(made["run_ids"]) == 17


def test_the_hypothesis_rules_on_the_hand_made_runs(made):
    h = made["hypotheses"]
    assert h["H1"]["supported"]  # every budget after unit 1 is above the random networks' 0
    assert h["H2"]["progress_above_neuron"] and h["H2"]["completion_below_neuron"]
    assert h["H3"]["seeds_with_drop"] == [0, 1] and h["H3"]["supported"]
    assert h["H4"]["supported"]
    c = made["comparisons"]["heldout_progress"]
    assert c["vs_scripted_expert"]["verdict"] == "random search clearly lower"
    assert c["minus_random_network_paired"]["verdict"] == "random search clearly higher"


def test_a_study_run_is_a_plain_random_search_run(study, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "UNITS", 2)
    made = RunStore.open(tmp_path / "study" / study._train_seed((3, "sw", str(tmp_path / "study"))))
    config = load_config(study.CONFIG, units=2)
    plain_config = parse_config(resolved(config), seed=3, tag="s4-s3")
    plain = train_random_search_run(plain_config, tmp_path / "plain").store

    assert made.read_meta()["study"] == "S4" and "study" not in plain.read_meta()
    assert made.read_config() == plain.read_config()

    def without_clock(store):
        return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]

    assert without_clock(made) == without_clock(plain)
    assert made.checkpoint_units() == plain.checkpoint_units() == [0, 1, 2]
    for unit in made.checkpoint_units():
        assert made.load_checkpoint(unit).equals(plain.load_checkpoint(unit))
        a, b = study.read_card(made, unit), study.read_card(plain, unit)
        a.pop("policy"), b.pop("policy")
        assert a == b
