"""Tests for the baseline runs (task 5.6, ADR-039): each baseline is a complete run folder with an
exam card, the card is the exam of the policy its config describes, and the three land where a
floor, a hand-tuned controller and a ceiling should."""

from contextlib import closing

import numpy as np
import pytest
import yaml

from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.env.experts import ExpertParams
from neurodrive.env.policies import MLPPolicy, NeuronPolicy
from neurodrive.env.track import Track
from neurodrive.experiment import db
from neurodrive.experiment.baselines import (
    HAND_NEURON_FILE,
    baseline_config,
    baseline_policy,
    hand_neuron_network,
    load_hand_neuron,
    make_baselines,
)
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, EXAM_V1, TRAINING_TRACK, exam_track
from neurodrive.experiment.stats import percentile

SEEDS = (0, 1)
COUNTED = ("track", "start_index", "status", "steps", "laps", "completed", "crashed")


def committed_neuron() -> NeuronPolicy:
    """The neuron straight from the committed YAML file, without the code under test."""
    raw = yaml.safe_load(HAND_NEURON_FILE.read_text(encoding="utf-8"))["neuron"]
    weights = (raw["w_left"], raw["w_right"])
    return NeuronPolicy(weights, raw["bias"], raw["activation"], raw["throttle"])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Every baseline, made once: random networks for seeds 0 and 1, the neuron, the expert."""
    runs_dir = tmp_path_factory.mktemp("baselines")
    made = make_baselines(runs_dir, seeds=SEEDS)
    return runs_dir, {r.store.path.name: r for r in made}


def by_method(runs, method):
    return [r for r in runs[1].values() if r.store.read_meta()["method"] == method]


def training_track_part(policy_for):
    """The training track's part of an exam of ``policy_for``, taken afresh."""
    track = Track.from_json(DEFAULT_TRACKS_DIR / f"{TRAINING_TRACK}.json")
    part, _ = exam_track(track, policy_for, EXAM_V1)
    return part.to_dict()


# ── configs ─────────────────────────────────────────────────────────────────────────────────


def test_configs_describe_each_baseline_completely():
    neuron = baseline_config("hand_neuron")["neuron"]
    raw = yaml.safe_load(HAND_NEURON_FILE.read_text(encoding="utf-8"))["neuron"]
    assert neuron == {k: raw[k] for k in ("w_left", "w_right", "bias", "activation", "throttle")}
    assert neuron == {
        "w_left": -12.0,
        "w_right": 12.0,
        "bias": 0.0,
        "activation": "tanh",
        "throttle": 0.24,
    }
    assert baseline_config("scripted_expert")["expert"]["look_base"] == ExpertParams().look_base
    random = baseline_config("random_network", seed=3)
    assert random["seed"] == 3 and random["network"]["layer_sizes"] == [7, 16, 16, 2]
    assert random["network"]["init_scheme"] == "xavier"
    for bad in (("nope", None), ("hand_neuron", 3), ("random_network", None)):
        with pytest.raises(ValueError):
            baseline_config(*bad)


def test_the_neuron_in_the_worked_example_steers_away_from_the_near_wall():
    policy = baseline_policy(baseline_config("hand_neuron"))(None)
    obs = np.full((2, 7), 0.5)
    obs[0, 0], obs[0, 4] = 0.3, 0.5  # wall close on the left: z = -3.6 + 6.0 = 2.4
    steer, throttle = policy.act(obs)[0]
    assert steer == pytest.approx(np.tanh(2.4)) and steer == pytest.approx(0.984, abs=5e-4)
    assert throttle == 0.24  # tanh(atanh(0.24)) is exactly 0.24
    assert policy.act(obs)[1, 0] == 0.0  # centred: z = 0, straight on


def test_the_neuron_as_a_network_drives_like_the_neuron():
    network = hand_neuron_network(load_hand_neuron())
    assert network.spec.layer_sizes == (7, 2) and network.n_parameters == 16
    assert network.weights[0][0].tolist() == [-12.0, 0, 0, 0, 12.0, 0, 0]
    assert network.weights[0][1].tolist() == [0.0] * 7
    obs = np.random.default_rng(0).random((10_000, 7))
    as_network, as_neuron = MLPPolicy(network).act(obs), committed_neuron().act(obs)
    assert np.all(as_network[:, 1] == 0.24)  # the throttle is exact
    assert np.abs(as_network - as_neuron).max() < 1e-14  # the steering: the last bit at most
    with pytest.raises(ValueError, match="constant throttle"):
        hand_neuron_network({**load_hand_neuron(), "activation": "sigmoid", "throttle": -0.2})


def test_a_broken_neuron_file_is_refused(tmp_path):
    good = yaml.safe_load(HAND_NEURON_FILE.read_text(encoding="utf-8"))
    cases = {
        "missing": {**good, "neuron": {k: v for k, v in good["neuron"].items() if k != "throttle"}},
        "inputs": {**good, "inputs": ["far_left_ray", "middle_ray"]},
        "policy": {**good, "policy": "mlp"},
    }
    messages = {"missing": "missing throttle", "inputs": "inputs must be", "policy": "not a neuron"}
    for case, data in cases.items():
        path = tmp_path / f"{case}.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError, match=messages[case]):
            load_hand_neuron(path)


# ── the baseline runs ───────────────────────────────────────────────────────────────────────


def test_every_baseline_is_a_complete_run_folder_with_an_exam_card(runs):
    runs_dir, made = runs
    assert sorted(r.store.read_meta()["method"] for r in made.values()) == [
        "hand_neuron",
        "random_network",
        "random_network",
        "scripted_expert",
    ]
    for run in made.values():
        meta = run.store.read_meta()
        assert meta["run_version"] == 1 and meta["status"] == "completed"
        assert meta["stop_reason"] == "units" and meta["exams"] is True
        assert meta["budget"] == {"units": 0, "env_steps": None, "wall_clock_s": None}
        assert meta["exam_wall_clock_s"] > 0
        assert run.store.read_metrics() == []  # nothing was trained
        assert run.store.exam_units() == [0]
        assert run.store.load_exam(0).to_dict() == run.card.to_dict()
        assert run.store.read_config() == baseline_config(
            meta["method"], run.store.read_config().get("seed")
        )
    for run in by_method(runs, "random_network"):
        assert run.store.checkpoint_units() == [0]
        assert run.card.policy == {"kind": "checkpoint", "run": run.store.path.name, "unit": 0}
    (neuron,) = by_method(runs, "hand_neuron")
    assert neuron.store.checkpoint_units() == [0]  # the neuron as a 7 -> 2 network
    assert neuron.store.load_checkpoint(0).equals(hand_neuron_network(load_hand_neuron()))
    assert neuron.card.policy == {"kind": "checkpoint", "run": neuron.store.path.name, "unit": 0}
    (expert,) = by_method(runs, "scripted_expert")
    assert expert.store.checkpoint_units() == []  # not a network: no checkpoint (ADR-039)
    assert expert.card.policy == {
        "kind": "baseline",
        "name": "scripted_expert",
        "run": expert.store.path.name,
    }
    sweeps = {r.store.read_meta().get("sweep") for r in made.values()}
    assert len(sweeps) == 2 and None in sweeps  # one sweep for the random seeds, none otherwise
    assert db.check(runs_dir) == []


def test_each_card_is_the_exam_of_the_policy_its_config_describes(runs):
    for run in runs[1].values():
        rebuilt = baseline_policy(run.store.read_config())
        assert training_track_part(rebuilt) == run.card.tracks[0].to_dict()


def test_the_random_network_is_the_one_a_learner_with_that_seed_starts_from(runs):
    for run in by_method(runs, "random_network"):
        seed = run.store.read_config()["seed"]
        assert run.store.load_checkpoint(0).equals(init_network(NetworkSpec(), make_rng(seed)))


def test_the_neuron_card_is_the_committed_yaml_neuron(runs):
    """The checkpoint is examined; the YAML neuron gives the same drives: every outcome equal,
    every number equal to 1e-9 (the network's sums differ from the neuron's in the last bit)."""
    (run,) = by_method(runs, "hand_neuron")
    direct = training_track_part(lambda world: committed_neuron())
    card = run.card.tracks[0].to_dict()
    for ours, theirs in zip(card["episodes"], direct["episodes"], strict=True):
        assert {k: ours[k] for k in COUNTED if k in ours} == {
            k: theirs[k] for k in COUNTED if k in theirs
        }
        for key in ("progress_m", "time_s", "mean_speed", "best_lap_s", "end_s_m"):
            assert ours[key] == pytest.approx(theirs[key], rel=1e-9, abs=1e-9)


def test_floor_hand_tuned_and_ceiling(runs):
    (neuron,) = by_method(runs, "hand_neuron")
    (expert,) = by_method(runs, "scripted_expert")
    # the expert: every start of every track, no crash (task 5.4's card)
    assert expert.card.training["completion_rate"] == expert.card.heldout["completion_rate"] == 1.0
    assert expert.card.heldout["crash_rate"] == 0.0
    # the neuron (Study S1): the loop and its mirror, nothing else
    assert neuron.card.training["completion_rate"] == 1.0
    assert neuron.card.heldout["completion_rate"] == 0.25
    rates = {t.track: t.aggregate["completion_rate"] for t in neuron.card.tracks}
    assert rates == {
        "train_loop_01": 1.0,
        "hairpins_01": 0.0,
        "monza_ccw": 0.0,
        "sweeper_01": 0.0,
        "train_loop_01_mirror": 1.0,
    }
    # random networks: never a lap, and below the neuron's progress everywhere
    for run in by_method(runs, "random_network"):
        assert run.card.training["completion_rate"] == run.card.heldout["completion_rate"] == 0.0
        neuron_progress = neuron.card.heldout["progress_frac_median"]
        assert run.card.heldout["progress_frac_median"] < neuron_progress


def test_the_random_sweep_median_from_sql_equals_the_cards(runs):
    runs_dir, _ = runs
    randoms = by_method(runs, "random_network")
    (sweep,) = {r.store.read_meta()["sweep"] for r in randoms}
    values = [r.card.heldout["progress_frac_median"] for r in randoms]
    with closing(db.connect(runs_dir / db.DB_NAME)) as conn:
        assert db.median_final_heldout_progress(conn, sweep) == percentile(values, 50)
