"""Guards for Study S5a (docs/studies/S5a.md): the checkpoint rules, the hypothesis rules on a
hand-made set of run folders, and that a study run is a plain imitation run."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from neurodrive.env.track import Track
from neurodrive.experiment.config import load_config, parse_config, resolved
from neurodrive.experiment.runner import checkpoint_due, train_imitation_run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.supervised.demos import record_expert, save_dataset

ROOT = Path(__file__).resolve().parents[2]
HELDOUT = ("hairpins_01", "sweeper_01")
SHORT = 12  # epochs of the hand-made runs; examined: 0, 1, 2, 3, 5, 6, 9, 10, 12


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_s5a", ROOT / "scripts" / "study_s5a.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_checkpoint_rules_on_the_briefs_worked_example(study):
    loss = [0.10, 0.06, 0.04, 0.02, 0.01]
    lap = [1.0, 1.0, 1.0, 0.0, 0.0]
    assert study.checkpoint_rules([1, 2, 3, 4, 5], loss, lap) == {
        "last": 5,
        "min_loss": 5,
        "training_exam": 3,  # the lowest loss among the epochs that lap best
    }
    # Equal losses go to the earliest epoch; with no full lap the best partial one counts.
    assert study.checkpoint_rules([3, 6, 9], [0.05, 0.05, 0.07], [0.4, 0.6, 0.6]) == {
        "last": 9,
        "min_loss": 3,
        "training_exam": 6,
    }
    with pytest.raises(ValueError):
        study.checkpoint_rules([1, 2], [0.1], [1.0, 1.0])


def test_spearman_is_reported_for_comparison(study):
    assert study.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert study.spearman([1, 2, 3, 4], [0.1, 0.1, 0.3, 0.3]) == pytest.approx(0.894427, abs=1e-6)
    assert study.spearman([1, 2, 3], [0.5, 0.5, 0.5]) is None


# --- a hand-made study ----------------------------------------------------------------------


def card(progress: float, completion: float = 0.0, lap: float = 0.0) -> dict:
    """An exam card with only what the study reads: every held-out track scores alike, and the
    training loop finishes the share ``lap`` of its starts."""
    held = {"progress_frac_median": progress, "completion_rate": completion, "crash_rate": 0.5}
    train = {"progress_frac_median": lap, "completion_rate": lap, "crash_rate": 1 - lap}
    tracks = [{"track": t, "split": "heldout", "aggregate": held} for t in HELDOUT]
    loop = {
        "track": "train_loop_01",
        "split": "train",
        "aggregate": train,
        "episodes": [{"mean_speed": 20.0}, {"mean_speed": 22.0}],
    }
    return {"training": train, "heldout": held, "tracks": [loop, *tracks], "policy": {}}


def fake_run(runs_dir: Path, method: str, seed, cards: dict, lines=(), tag="s5a") -> None:
    config = {"method": method} if seed is None else {"method": method, "seed": seed}
    tag = f"{tag}-s{seed}" if seed is not None else tag
    store = RunStore.create(runs_dir, method, tag, config, extra_meta={"study": "S5a"})
    for line in lines:
        store.append_metrics(line)
    (store.path / "exams").mkdir(exist_ok=True)
    for unit, c in cards.items():
        (store.path / "exams" / f"unit_{unit}.json").write_text(json.dumps(c), encoding="utf-8")
    dataset = {"sha256": "ab" * 32, "val_episodes": [1, 2]}
    store.update_meta(
        sweep="sw" if method == "imitation" else None, exam_wall_clock_s=1.0, dataset=dataset
    )
    store.finish("completed", "units")


def lines_of(val: list[float]) -> list[dict]:
    names = ("loss", "loss_steer", "loss_throttle", "val_loss_steer", "val_loss_throttle")
    return [
        {"unit": e, "wall_clock_s": 0.1 * e, "val_loss": v, **dict.fromkeys(names, v)}
        for e, v in enumerate(val, start=1)
    ]


def make_study(runs: Path, study) -> None:
    """Every seed: the loss falls every epoch (0.12 - 0.01 e); the network laps the training
    loop and makes 0.8 held-out progress up to epoch 6, then laps nothing and makes 0.2 (seed k
    adds 0.01 k to both). Random networks score 0, the neuron 0.234, the expert 1."""
    examined = [e for e in range(SHORT + 1) if checkpoint_due(e, study.CHECKPOINT_EVERY)]
    for seed in study.SEEDS:
        good = {e: card(0.8 + 0.01 * seed, 0.5, 1.0) for e in examined if 1 <= e <= 6}
        bad = {e: card(0.2 + 0.01 * seed, 0.0, 0.0) for e in examined if e > 6}
        cards = {0: card(0.0), **good, **bad}
        fake_run(runs, "imitation", seed, cards, lines_of([0.12 - 0.01 * e for e in range(1, 13)]))
    for seed in study.RANDOM_SEEDS:
        fake_run(runs, "random_network", seed, {0: card(0.0)}, tag="baseline")
    fake_run(runs, "hand_neuron", None, {0: card(0.234, 0.25)}, tag="baseline")
    fake_run(runs, "scripted_expert", None, {0: card(1.0, 1.0, 1.0)}, tag="baseline")


@pytest.fixture
def short(study, monkeypatch):
    monkeypatch.setattr(study, "EPOCHS", SHORT)
    monkeypatch.setattr(study, "H1_FROM", 3)
    monkeypatch.setattr(study, "EARLY", 3)
    return study


def test_the_analysis_and_the_hypothesis_rules_on_the_hand_made_runs(short, tmp_path):
    make_study(tmp_path, short)
    made = short.analyse(tmp_path)
    assert made["design"]["examined_epochs"] == [1, 2, 3, 5, 6, 9, 10, 12]
    assert len(made["run_ids"]) == 22 and made["seed0_equals_task_6_3"] == {"found": False}
    # Every pair across epoch 6 / 9 has the lower loss later and the lower progress: 15 pairs
    # disagree, the 13 within a group are left out, none agrees.
    seed0 = made["seeds"][0]["agreement_heldout_progress"]
    assert (seed0["agree"], seed0["disagree"], seed0["left_out"]) == (0, 15, 13)
    assert made["agreement"]["agreement_heldout_progress"]["median"] == 0.0
    rules = made["rules"]
    assert rules["last"]["epoch"]["median"] == 12 and rules["min_loss"]["epoch"]["median"] == 12
    assert rules["training_exam"]["epoch"]["median"] == 6
    assert rules["training_exam"]["heldout_progress"]["median"] == pytest.approx(0.845)
    d = made["rule_differences"]["training_exam-min_loss:heldout_progress"]
    assert d["estimate"] == pytest.approx(0.6) and d["ci"][0] == pytest.approx(0.6)
    assert made["seeds"][3]["first_full_lap_epoch"] == 1
    assert made["seeds"][3]["first_lost_lap_epoch"] == 9
    assert made["seeds"][3]["last_full_lap_epoch"] == 6
    h = made["hypotheses"]
    assert all(h[k]["supported"] for k in ("H1", "H2", "H3", "H4"))
    assert made["baselines"]["hand_neuron"]["heldout_progress"] == 0.234


def test_the_hypothesis_rules_can_fail(short, tmp_path):
    make_study(tmp_path, short)
    made = short.analyse(tmp_path)
    records = [{**s, "first_full_lap_epoch": s["first_full_lap_epoch"]} for s in made["seeds"]]
    records[4] = {**records[4], "first_full_lap_epoch": None}  # one seed never laps
    agree = {"median": 0.5}  # a coin flip is not "below half"
    rules = short.hypotheses(records, agree, {"estimate": 0.1, "ci": [-0.05, 0.2]})
    assert not rules["H2"]["supported"] and not rules["H3"]["supported"]
    assert not rules["H4"]["supported"]  # the interval includes 0
    with pytest.raises(SystemExit, match="exactly one imitation run"):
        fake_run(tmp_path, "imitation", 0, {0: card(0.0)}, lines_of([0.1] * SHORT))
        short.analyse(tmp_path)


# --- a study run is a plain run -------------------------------------------------------------


@pytest.fixture(scope="module")
def datasets(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("demos")
    track = Track.from_json(ROOT / "tracks" / "train_loop_01.json")
    demo = record_expert(
        track, start_fractions=(0.0, 0.5), lateral_offsets=(0.0,), random_starts=2, seed=0
    )
    save_dataset(demo, folder, tag="t", when=datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC))
    return folder


def test_a_study_run_is_a_plain_imitation_run(study, datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(study, "EPOCHS", 3)
    monkeypatch.setattr(study, "DATASET", "20261104-120000-expert-t")
    monkeypatch.setattr(study, "DATASETS_DIR", datasets)
    made = RunStore.open(tmp_path / "study" / study._train_seed((3, "sw", str(tmp_path / "study"))))
    data = resolved(load_config(study.CONFIG, epochs=3))
    data["data"]["dataset"], data["checkpoint_every"] = "20261104-120000-expert-t", 3
    plain_config = parse_config(data, seed=3, tag="s5a-s3")
    plain = train_imitation_run(plain_config, tmp_path / "plain", datasets_dir=datasets).store

    meta = made.read_meta()
    assert (meta["study"], meta["sweep"]) == ("S5a", "sw") and "study" not in plain.read_meta()
    assert made.read_config() == plain.read_config()
    assert made.read_config()["training"]["epochs"] == 3 and meta["checkpoint_every"] == 3

    def without_clock(store):
        return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]

    assert without_clock(made) == without_clock(plain)
    assert made.checkpoint_units() == plain.checkpoint_units() == [0, 1, 2, 3]
    for unit in made.checkpoint_units():
        assert made.load_checkpoint(unit).equals(plain.load_checkpoint(unit))
        a, b = study.read_card(made, unit), study.read_card(plain, unit)
        a.pop("policy"), b.pop("policy")
        assert a == b


def test_the_post_hoc_rule_picks_the_earliest_best_lap(study):
    assert study.earliest_lap_rule([1, 2, 3, 4], [0.0, 1.0, 1.0, 0.0]) == 2
    assert study.earliest_lap_rule([3, 6, 9], [0.4, 0.9, 0.9]) == 6  # no full lap: the best one
