"""Guards for Study S5c (docs/studies/S5c.md): the paired difference and the label ratio on the
Brief's worked examples, the checkpoint choice reading no held-out number, the hypothesis rules on
a hand-made set of run folders (including rules that must fail), and that a study run is a plain
imitation run."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from neurodrive.env.track import Track
from neurodrive.experiment.config import load_config, parse_config, resolved
from neurodrive.experiment.runner import train_imitation_run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.supervised.demos import record_expert, save_dataset

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ("hairpins_01", "monza_ccw", "sweeper_01", "train_loop_01_mirror")
UNITS = (0, 1, 2, 3, 4, 5, 6)  # examined units of the hand-made runs


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_s5c", ROOT / "scripts" / "study_s5c.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── the Brief's worked examples ─────────────────────────────────────────────────────────────


def test_paired_difference_on_the_briefs_three_seed_example(study):
    d = study.paired_difference([0.80, 0.60, 0.90], [0.50, 0.60, 0.40])
    assert d["per_seed"] == pytest.approx([0.30, 0.0, 0.50])
    assert d["estimate"] == pytest.approx(0.30)
    assert d["ci"] == pytest.approx([0.0, 0.50])  # holds 0: no clear difference
    assert d["clear"] is False
    with pytest.raises(ValueError, match="one value per seed"):
        study.paired_difference([1.0], [1.0, 2.0])


def test_a_clear_difference_has_an_interval_that_excludes_zero(study):
    d = study.paired_difference([0.9] * 10, [0.5, 0.55, 0.6, 0.5, 0.45, 0.5, 0.55, 0.6, 0.5, 0.5])
    assert d["estimate"] > 0.35 and d["ci"][0] > 0 and d["clear"]


def test_label_ratio_on_the_briefs_example(study):
    assert study.label_ratio(196_765, 39_338, 170_000) == pytest.approx(0.94, abs=0.005)


def test_the_checkpoint_rule_on_the_briefs_example(study):
    # epochs 1, 2, 5: training-loop completion 0.8, 1.0, 1.0 and validation loss 0.05, 0.04, 0.03
    picks = study.s5a.checkpoint_rules([1, 2, 5], [0.05, 0.04, 0.03], [0.8, 1.0, 1.0])
    assert picks["training_exam"] == 5


# ── a hand-made study ───────────────────────────────────────────────────────────────────────


def card(progress: float, lap: float = 0.0) -> dict:
    """An exam card with what the study reads: every held-out track scores ``progress`` and the
    training loop finishes the share ``lap`` of its starts."""
    held = {"progress_frac_median": progress, "completion_rate": progress / 2, "crash_rate": 0.5}
    train = {"progress_frac_median": lap, "completion_rate": lap, "crash_rate": 1 - lap}
    episode = {"start_index": 0, "start_s0_m": 0.0, "start_lateral_m": 0.0, "status": "crashed"}
    episode |= {"progress_frac": progress, "steps": 10, "end_s_m": 5.0, "mean_speed": 9.0}
    tracks = [
        {"track": t, "split": "heldout", "aggregate": held, "episodes": [episode]} for t in TRACKS
    ]
    loop = {
        "track": "train_loop_01",
        "split": "train",
        "aggregate": train,
        "episodes": [{"mean_speed": 20.0}],
    }
    return {"training": train, "heldout": held, "tracks": [loop, *tracks], "policy": {}}


def fake_run(runs_dir: Path, method: str, seed, cards: dict, lines=(), tag="") -> None:
    config = {"method": method} if seed is None else {"method": method, "seed": seed}
    tag = f"{tag}-s{seed}" if seed is not None and tag != "baseline" else tag
    store = RunStore.create(runs_dir, method, tag, config, extra_meta={"study": "S5c"})
    for line in lines:
        store.append_metrics(line)
    (store.path / "exams").mkdir(exist_ok=True)
    for unit, c in cards.items():
        (store.path / "exams" / f"unit_{unit}.json").write_text(json.dumps(c), encoding="utf-8")
    dataset = {"sha256": "ab" * 32, "val_episodes": [1], "demo_frames": 1000}
    store.update_meta(exam_wall_clock_s=1.0, dataset=dataset)
    store.finish("completed", "units")


def lines_of(n: int, demo: int, queries: int) -> list[dict]:
    names = ("loss", "loss_steer", "loss_throttle", "val_loss_steer", "val_loss_throttle")
    out = []
    for u in range(1, n + 1):
        v = 0.12 - 0.01 * u  # the validation loss falls every unit
        line = {"unit": u, "wall_clock_s": 0.1 * u, "val_loss": v, **dict.fromkeys(names, v)}
        line |= {"demo_frames_total": demo, "teacher_queries_total": queries * (u > 1)}
        line |= {"env_steps_total": queries * (u > 1), "train_frames": demo + queries}
        out.append(line)
    return out


def make_study(
    runs: Path, study, selected: dict[str, float], last: dict[str, float], spread: dict[str, float]
) -> None:
    """Every arm and seed: units 1-3 lap the training loop and score ``selected`` held-out
    progress, units 4-6 lap nothing and score ``last``; seed k adds ``spread`` * k to both."""
    for arm, spec in study.ARMS.items():
        extra = (100_000, 400_000) if spec["method"] == "dagger" else (1000, 0)
        for seed in study.SEEDS:
            good = {u: card(selected[arm] + spread[arm] * seed, 1.0) for u in (1, 2, 3)}
            bad = {u: card(last[arm] + spread[arm] * seed, 0.0) for u in (4, 5, 6)}
            fake_run(
                runs,
                spec["method"],
                seed,
                {0: card(0.0), **good, **bad},
                lines_of(6, *extra),
                spec["tag"],
            )
    for seed in study.RANDOM_SEEDS:
        fake_run(runs, "random_network", seed, {0: card(0.0)}, tag="baseline")
    fake_run(runs, "hand_neuron", None, {0: card(0.234)}, tag="baseline")
    fake_run(runs, "scripted_expert", None, {0: card(1.0, 1.0)}, tag="baseline")


SELECTED = {
    "human": 0.5,
    "e_match": 0.8,
    "e_full": 0.85,
    "clone_wide": 0.6,
    "dagger_wide": 0.7,
    "dagger_human": 0.6,
}
LAST = {**SELECTED, "dagger_wide": 0.65, "human": 0.1}
SPREAD = {a: 0.01 for a in SELECTED} | {"dagger_wide": 0.001}


@pytest.fixture
def small(study, monkeypatch):
    monkeypatch.setattr(study, "SEEDS", (0, 1, 2, 3))
    monkeypatch.setattr(study, "RANDOM_SEEDS", (0, 1))
    monkeypatch.setattr(study, "FOLLOW_UP_SEEDS", (0, 1))
    # the analysis describes real dataset folders; here there are none
    monkeypatch.setattr(study, "find_dataset", lambda folder, ident: ident)
    monkeypatch.setattr(study, "DemoDataset", SimpleNamespace(load=lambda path: path))
    monkeypatch.setattr(study, "describe_dataset", lambda dataset: {"id": dataset})
    return study


def test_the_analysis_and_the_hypothesis_rules_on_the_hand_made_runs(small, tmp_path):
    make_study(tmp_path, small, SELECTED, LAST, SPREAD)
    made = small.analyse(tmp_path, inspect=False)
    human = made["summary"]["human"]["rules"]
    # units 1-3 all lap; the lowest validation loss among them is unit 3's
    assert human["training_exam"]["unit"]["median"] == 3 and human["last"]["unit"]["median"] == 6
    assert human["training_exam"]["heldout_progress"]["median"] == pytest.approx(0.515)
    assert human["last"]["heldout_progress"]["median"] == pytest.approx(0.115)
    assert human["training_exam"]["per_track_progress"]["monza_ccw"]["median"] == pytest.approx(
        0.515
    )
    h = made["hypotheses"]
    assert h["T1"]["estimate"] == pytest.approx(0.3) and h["T1"]["supported"]
    assert h["T2"]["seeds_with_a_full_lap_checkpoint"] == 4 and h["T2"]["supported"]
    assert h["T3"]["supported"]
    # dagger 0.7 + 0.001 k against clone 0.6 + 0.01 k: d = 0.100, 0.091, 0.082, 0.073
    assert h["D1"]["estimate"] == pytest.approx(0.0865)
    assert h["D1"]["supported"] and h["D2"]["supported"] and h["D3"]["supported"]
    assert h["D3"]["estimate"] == pytest.approx(0.05)
    assert h["H1"]["supported"]
    assert made["label_ratio"]["median"] == pytest.approx(1000 / 500_000)
    assert made["baselines"]["hand_neuron"]["heldout_progress"] == 0.234
    assert made["summary"]["dagger_wide"]["budget"]["teacher_queries"]["median"] == 400_000
    assert made["failure"]["episode"]["seed"] == 0 and made["failure"]["episode"]["unit"] == 3
    assert made["failure"]["episode"]["progress_frac"] == pytest.approx(0.5)
    # byte-identical on a second analysis
    again = small.analyse(tmp_path, inspect=False)
    assert json.dumps(made, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_the_hypothesis_rules_can_fail(small, tmp_path):
    selected = {
        **SELECTED,
        "dagger_wide": 0.5,
        "e_full": 0.5,
        "e_match": 0.6,
        "human": 0.8,
        "dagger_human": 0.7,
    }
    spread = {**SPREAD, "dagger_wide": 0.05}
    make_study(tmp_path, small, selected, {**selected, "dagger_wide": 0.6}, spread)
    h = small.analyse(tmp_path, inspect=False)["hypotheses"]
    assert not h["T1"]["supported"] and h["T1"]["estimate"] < 0  # the human beats the expert
    assert not h["T3"]["supported"]  # more expert frames scored lower
    assert not h["D1"]["supported"] and not h["D2"]["supported"]
    assert not h["D3"]["supported"]  # the last round was better than the selected one
    assert not h["H1"]["supported"]


def test_a_run_is_missing_or_repeated(small, tmp_path):
    make_study(tmp_path, small, SELECTED, LAST, SPREAD)
    fake_run(
        tmp_path, "imitation", 0, {0: card(0.0)}, lines_of(6, 1000, 0), small.ARMS["human"]["tag"]
    )
    with pytest.raises(SystemExit, match="exactly one human run"):
        small.analyse(tmp_path, inspect=False)


def test_choosing_a_checkpoint_reads_no_held_out_number(small, tmp_path):
    make_study(tmp_path, small, SELECTED, LAST, SPREAD)
    path = small.find_runs(tmp_path)["arms"]["human"][0]
    before = small.run_record(path)["rules"]
    for unit_file in (path / "exams").glob("unit_*.json"):
        c = json.loads(unit_file.read_text(encoding="utf-8"))
        c["heldout"]["progress_frac_median"] = 1.0 - c["heldout"]["progress_frac_median"]
        unit_file.write_text(json.dumps(c), encoding="utf-8")
    after = small.run_record(path)["rules"]
    assert {k: v["unit"] for k, v in before.items()} == {k: v["unit"] for k, v in after.items()}


def test_arms_are_told_apart_by_their_tags(study):
    assert study.arm_of("s5c-human-s3") == "human" and study.arm_of("s5c-dw-s0") == "dagger_wide"
    assert study.arm_of("s5c-cw-s9") == "clone_wide" and study.arm_of("baseline") is None
    assert study.arm_of("s5c-human-sx") is None


# ── a study run is a plain run ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def datasets(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("demos")
    track = Track.from_json(ROOT / "tracks" / "train_loop_01.json")
    demo = record_expert(
        track, start_fractions=(0.0, 0.5), lateral_offsets=(0.0,), random_starts=2, seed=0
    )
    save_dataset(demo, folder, tag="t", when=datetime(2026, 11, 11, 12, 0, 0, tzinfo=UTC))
    return folder


def test_a_study_run_is_a_plain_imitation_run(study, datasets, tmp_path, monkeypatch):
    spec = {**study.ARMS["human"], "dataset": "20261111-120000-expert-t", "epochs": 3}
    monkeypatch.setitem(study.ARMS, "human", spec)
    monkeypatch.setattr(study, "DATASETS_DIR", datasets)
    runs = tmp_path / "study"
    made = RunStore.open(runs / study._train_job(("human", 3, "sw", str(runs))))
    data = resolved(load_config(study.IMITATION, epochs=3))
    data["data"]["dataset"], data["checkpoint_every"] = "20261111-120000-expert-t", 3
    plain_config = parse_config(data, seed=3, tag="s5c-human-s3")
    plain = train_imitation_run(plain_config, tmp_path / "plain", datasets_dir=datasets).store

    meta = made.read_meta()
    assert (meta["study"], meta["sweep"]) == ("S5c", "sw") and "study" not in plain.read_meta()
    assert made.read_config() == plain.read_config()
    assert made.checkpoint_units() == plain.checkpoint_units() == [0, 1, 2, 3]
    for unit in made.checkpoint_units():
        assert made.load_checkpoint(unit).equals(plain.load_checkpoint(unit))
        a, b = study.s5a.read_card(made, unit), study.s5a.read_card(plain, unit)
        a.pop("policy"), b.pop("policy")
        assert a == b
    assert ("imitation", "s5c-human-s3") in study.existing(runs)  # a second --train skips it
