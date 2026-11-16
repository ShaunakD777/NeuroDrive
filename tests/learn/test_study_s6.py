"""Guards for Study S6 (docs/studies/S6.md): the paired difference on the Brief's three-seed
example, the checkpoint choice reading no held-out number, every hypothesis rule on hand-made
results (including cases that must fail), the exploit flags, and that every arm is its shipped
config with exactly one change and the same budget."""

import importlib.util
import json
from pathlib import Path

import pytest

from neurodrive.experiment.runner import train_ga_run

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_s6", ROOT / "scripts" / "study_s6.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── the statistics ──────────────────────────────────────────────────────────────────────────


def test_paired_difference_on_the_briefs_three_seed_example(study):
    d = study.paired_difference([0.80, 0.60, 0.90], [0.50, 0.60, 0.40])
    assert d["per_seed"] == pytest.approx([0.30, 0.0, 0.50])
    assert d["estimate"] == pytest.approx(0.30)
    assert d["ci"] == pytest.approx([0.0, 0.50])  # holds 0: no clear difference
    assert d["clear"] is False
    with pytest.raises(ValueError, match="one value per seed"):
        study.paired_difference([1.0], [1.0, 2.0])


def test_paired_by_seed_pairs_only_seeds_with_both_values_and_needs_three(study):
    a = {0: 0.8, 1: 0.6, 2: 0.9, 3: None, 4: 0.7}
    b = {0: 0.5, 1: 0.6, 2: 0.4, 3: 0.1}
    out = study.paired_by_seed(a, b)
    assert out["seeds"] == [0, 1, 2] and out["estimate"] == pytest.approx(0.30)
    assert out["supported"] is False  # the interval [0, 0.5] holds 0
    few = study.paired_by_seed({0: 1.0, 1: 2.0}, {0: 0.0, 1: 0.0})
    assert few == {"seeds": [0, 1], "n_seeds": 2, "testable": False, "supported": False}
    lower = study.paired_by_seed({s: 0.5 for s in range(5)}, {s: 0.9 for s in range(5)}, -1)
    assert lower["supported"] and lower["ci"][1] < 0
    assert not study.paired_by_seed({s: 0.5 for s in range(5)}, {s: 0.9 for s in range(5)})[
        "supported"
    ]


def test_the_curve_takes_each_seeds_latest_exam_at_or_before_each_step_count(study):
    records = {
        0: {"curve": {"0": {"env_steps": 0, "T": 0.0}, "5": {"env_steps": 3, "T": 1.0}}},
        1: {"curve": {"0": {"env_steps": 0, "T": 0.0}, "9": {"env_steps": 6, "T": 0.5}}},
    }
    points = study.curve_band(records, "T", [0, 4, 7])["points"]
    assert [p["median"] for p in points] == [0.0, 0.5, 0.75]


# ── the checkpoint reads training numbers only ──────────────────────────────────────────────


def test_the_checkpoint_is_the_last_lines_best_unit(study):
    metrics = [
        {"unit": 1, "best_unit": 1},
        {"unit": 2, "best_unit": 2},
        {"unit": 3, "best_unit": 2},
    ]
    assert study.choose_checkpoint(metrics) == 2
    with pytest.raises(ValueError):
        study.choose_checkpoint([])


@pytest.mark.slow
def test_the_chosen_checkpoint_ignores_every_held_out_number(study, tmp_path):
    data = study.resolved(study.study_config("n20", 0))
    data["budget"]["env_steps"] = 150_000
    store = train_ga_run(study.parse_config(data), tmp_path, extra_meta={"study": "S6"}).store
    assert study.ensure_exam(store.path) and not study.ensure_exam(store.path)
    before = study.run_record(store.path, "n20")
    # make every other checkpoint look perfect on the held-out tracks: the choice must not move
    chosen = before["unit"]
    card = study.read_card(store, chosen)
    for unit in store.checkpoint_units():
        if unit == chosen:
            continue
        fake = json.loads(json.dumps(card))
        fake["heldout"].update(progress_frac_median=1.0, completion_rate=1.0)
        (store.path / "exams" / f"unit_{unit}.json").write_text(json.dumps(fake), "utf-8")
    after = study.run_record(store.path, "n20")
    assert after["unit"] == chosen and after["P"] == before["P"]
    assert len(store.checkpoint_units()) >= 2  # there was something else to choose


# ── the arms ────────────────────────────────────────────────────────────────────────────────


def _flat(data, prefix=""):
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_every_arm_is_its_shipped_config_with_exactly_one_change(study):
    default = _flat(study.resolved(study.study_config("ga", 0)))
    for arm, spec in study.ARMS.items():
        mine = _flat(study.resolved(study.study_config(arm, 0)))
        assert mine["budget.env_steps"] == 40_000_000
        assert mine["budget.units"] is None and mine["budget.wall_clock_s"] is None
        assert mine["exams"] is (arm in ("ga", "random_search"))
        assert mine["tag"] == f"{spec['tag']}-s0" and mine["seed"] == 0
        if arm == "random_search":
            assert mine["method"] == "random_search"
            for key in default:  # the same network, fitness and budget as the GA
                if key.startswith(("network.", "fitness.", "budget.")):
                    assert mine[key] == default[key], key
            continue
        differs = {k for k in default if k not in ("tag", "exams") and mine[k] != default[k]}
        section, key, value = spec["change"] or (None, None, None)
        assert differs == (set() if arm == "ga" else {f"{section}.{key}"}), arm
        if value is not None:
            assert mine[f"{section}.{key}"] == value
    assert set(study.TRAIN_ORDER) == set(study.ARMS)


# ── the hypothesis rules on hand-made results ───────────────────────────────────────────────


def _records(study, **changes):
    """Ten GA seeds that beat random search everywhere, sweeps and ablations that do what the
    hypotheses predict; ``changes[arm] = {key: value or {seed: value}}`` overrides."""

    def rec(T=1.0, P=0.5, L=30.0, fit=1.25, div=4.0, e_in=1.0):
        return {"T": T, "P": P, "L": L, "best_fitness": fit, "final_diversity": div, "e_in": e_in}

    out = {
        "ga": {s: rec(L=30.0 + 0.1 * s) for s in range(10)},
        "random_search": {
            s: rec(T=0.0, P=0.1, L=None, fit=0.4, div=None, e_in=None) for s in range(10)
        },
        "n20": {s: rec(fit=1.20) for s in range(5)},
        "n100": {s: rec(fit=1.22) for s in range(5)},
        "p005": {s: rec(div=2.0) for s in range(5)},
        "p01": {s: rec(div=6.0, fit=1.10 + 0.01 * s) for s in range(5)},
        "s002": {s: rec(div=3.0) for s in range(5)},
        "s02": {s: rec(div=8.0, fit=1.00 + 0.01 * s) for s in range(5)},
        "finish_bonus": {s: rec(T=0.8, L=33.0 + 0.2 * s, fit=1.8, e_in=0.7) for s in range(5)},
        "progress": {s: rec(T=0.2, L=35.0, fit=0.9) for s in range(5)},
    }
    for arm, fields in changes.items():
        for key, value in fields.items():
            for seed in out[arm]:
                out[arm][seed][key] = value[seed] if isinstance(value, dict) else value
    return out


def test_every_hypothesis_is_supported_on_results_built_to_support_it(study):
    h = study.hypotheses(_records(study), expert_e_in=0.63)
    for name in ("G1", "G2", "N1", "M1", "M2", "F1", "F2", "R1", "R2"):
        assert h[name]["supported"] is True, name
    assert h["G1"]["mastery_seeds"] == 10 and h["G1"]["mastery"]
    assert h["R1"]["estimate"] == pytest.approx(1.0 - 0.63)
    assert h["F1"]["seeds"] == [0, 1, 2, 3, 4]  # paired with the default's seeds 0-4 only


def test_the_rules_fail_when_the_results_say_otherwise(study):
    h = study.hypotheses(
        _records(
            study,
            ga={"T": {s: 1.0 if s < 4 else 0.0 for s in range(10)}, "P": 0.1},
            n100={"best_fitness": 1.3},  # the bigger population wins
            p005={"final_diversity": 5.0},  # more diversity than the default at a lower rate
            s02={"best_fitness": 1.25},  # strong mutation costs nothing
            finish_bonus={"L": {s: 30.0 + 0.1 * s for s in range(5)}},  # no slower
            progress={"T": 0.9},  # finishing without a reward for it
        ),
        expert_e_in=1.0,  # evolution cuts exactly as much as the expert
    )
    assert h["G1"]["mastery_seeds"] == 4 and not h["G1"]["mastery"]
    assert not h["G2"]["supported"]  # GA's P = random search's P
    assert not h["N1"]["supported"]
    assert not h["M1"]["supported"] and h["M1"]["strength"] and not h["M1"]["rate"]
    assert not h["M2"]["supported"] and h["M2"]["p01"]["supported"]
    assert not h["F1"]["supported"]
    assert not h["F2"]["supported"]
    assert not h["R1"]["supported"]


def test_rules_on_too_few_finishing_seeds_are_not_testable(study):
    h = study.hypotheses(
        _records(study, finish_bonus={"L": {0: 31.0, 1: None, 2: None, 3: None, 4: 32.0}}),
        expert_e_in=0.63,
    )
    assert h["F1"]["testable"] is False and h["F1"]["supported"] is False
    assert h["F1"]["seeds"] == [0, 4]


# ── the exploit flags ───────────────────────────────────────────────────────────────────────


def test_the_exploit_flags(study):
    records = {arm: {} for arm in study.ARMS}

    def rec(T, C, smooth, times, unseen):
        return {
            "run_id": "r",
            "unit": 1,
            "T": T,
            "C": C,
            "P": 0.1,
            "training_smoothness": smooth,
            "training_time_limit_s": 92.0,
            "training_finish_times_s": times,
            "unseen_start_completion": unseen,
        }

    for s in range(9):
        records["ga"][s] = rec(1.0, 0.25, 0.01, [70.0] * 10, 1.0)
    records["ga"][9] = rec(1.0, 0.0, 0.01, [70.0] * 10, 0.5)  # overfit; misses unseen starts
    records["n20"][0] = rec(0.3, 0.0, 0.05, [90.5, 80.0, 70.0], None)  # jerky and just in time
    out = study.exploit_search(records)
    assert out["smoothness_cut"] == pytest.approx(0.05)
    flags = {(x["arm"], x["seed"]): x["flags"] for x in out["flagged"]}
    assert flags == {("ga", 9): ["overfit"], ("n20", 0): ["jerky", "just_in_time"]}
    assert out["masters"] == 10 and out["masters_lapping_every_unseen_start"] == 9
