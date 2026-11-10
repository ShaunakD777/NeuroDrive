"""Tests for DAgger (task 6.6, ADR-049): the on-policy error, the collection drive (the network
drives, the expert labels, the budget counts every frame), round 1 as behavioural cloning,
aggregation, divergence, an exact resume and the dataset of collected frames."""

from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.nn.init import init_network
from neurodrive.core.rng import child_rng, make_rng
from neurodrive.env.episode import run_episode
from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.policies import MLPPolicy
from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig
from neurodrive.learn.base import Learner, Resumable
from neurodrive.learn.supervised.dagger import (
    COLLECT_METRICS,
    DaggerLearner,
    DaggerSettings,
    collect,
    collected_dataset,
    onpolicy_errors,
)
from neurodrive.learn.supervised.demos import (
    RANDOM_START_HEADING,
    RANDOM_START_LATERAL,
    DemoDataset,
    driven_steps,
    frames_from_recording,
    record_expert,
    save_dataset,
)
from neurodrive.learn.supervised.imitation import (
    ImitationLearner,
    ImitationSettings,
    check_trainable,
)

TRACKS = Path(__file__).resolve().parents[3] / "tracks"
IMITATION = ImitationSettings(batch_size=512)
SETTINGS = DaggerSettings(imitation=IMITATION, epochs_per_round=2, random_starts=3)


@pytest.fixture(scope="module")
def track() -> Track:
    return Track.from_json(TRACKS / "train_loop_01.json")


@pytest.fixture(scope="module")
def demo(track) -> DemoDataset:
    """A small expert dataset: 2 exam-like starts and 2 random starts (4 drives)."""
    data = record_expert(
        track, start_fractions=(0.0, 0.5), lateral_offsets=(0.0,), random_starts=2, seed=0
    )
    data.meta["dataset_id"] = "20261110-120000-expert-t"
    return data


def learner(demo, track, seed: int = 0, settings: DaggerSettings = SETTINGS) -> DaggerLearner:
    return DaggerLearner(settings, demo, track, make_rng(seed))


# ── the on-policy error (Brief §4d) ─────────────────────────────────────────────────────────


def test_the_onpolicy_error_worked_example():
    action = np.array([[0.0, 0.5], [0.1, 0.9], [0.0, 1.0]])
    label = np.array([[0.0, 0.4], [0.1, 0.3], [-0.2, -0.4]])
    found = onpolicy_errors(action, label)
    assert found["onpolicy_loss"] == pytest.approx(2.37 / 6)  # 0.395
    assert found["onpolicy_loss_steer"] == pytest.approx(0.04 / 3)
    assert found["onpolicy_loss_throttle"] == pytest.approx(2.33 / 3)
    assert set(onpolicy_errors(np.zeros((0, 2)), np.zeros((0, 2))).values()) == {None}


# ── the collection drive ────────────────────────────────────────────────────────────────────


def test_the_network_drives_and_the_expert_labels(track):
    """The collected drive is the plain drive of the network (the labels never steer), and every
    label is the expert's label of exactly the recorded state."""
    network = init_network(IMITATION.spec, make_rng(3))
    found = collect(network, track, make_rng(7), n_starts=3, target_laps=2)
    # 1. The same network from the same starts, without any labelling.
    world = World(track, WorldConfig.for_track(track, 2))
    world.reset(
        n=3,
        rng=make_rng(7),
        random_s0=True,
        lateral_jitter=RANDOM_START_LATERAL,
        heading_jitter=RANDOM_START_HEADING,
    )
    plain = run_episode(world, MLPPolicy(network))
    rows = frames_from_recording(plain)
    steps = driven_steps(plain)
    assert np.array_equal(found.frames["observation"], rows["observation"])
    assert np.array_equal(found.frames["x"], rows["x"])
    actions = np.concatenate([plain.action[:n, car] for car, n in enumerate(steps)])
    assert np.array_equal(found.action, actions)
    # 2. Replay the recorded actions in a fresh world and ask a fresh expert at every state.
    replay = World(track, WorldConfig.for_track(track, 2))
    replay.reset(
        s0=plain.start_s0, lateral=plain.start_lateral, heading_offset=plain.start_heading_offset
    )
    expert = ScriptedExpert(replay)
    labels = []
    for t in range(plain.n_steps):
        labels.append(np.clip(expert.label(), -1.0, 1.0))
        replay.step(plain.action[t, :, 0], plain.action[t, :, 1])
    labels = np.array(labels)
    expected = np.concatenate([labels[:n, car] for car, n in enumerate(steps)])
    assert np.array_equal(found.frames["label"], expected)
    assert not np.array_equal(found.frames["label"], found.action)


def test_every_collected_frame_is_one_step_and_one_query(track):
    """The budget rule (Brief §4c): a car that drove n_c steps gives n_c frames; frozen steps
    of cars that stopped early are left out. M = n_1 + ... + n_R."""
    network = init_network(
        IMITATION.spec, make_rng(1)
    )  # untrained: the cars stop at different times
    found = collect(network, track, make_rng(2), n_starts=3, target_laps=2, round_number=2)
    per_car = [e["frames"] for e in found.episodes]
    assert len(set(per_car)) > 1  # the cars did stop at different steps
    assert found.n_frames == sum(per_car) == len(found.frames["episode"])
    for car, n in enumerate(per_car):
        assert np.array_equal(found.frames["step"][found.frames["episode"] == car], np.arange(n))
    assert {e["round"] for e in found.episodes} == {2}
    assert {e["start_kind"] for e in found.episodes} == {"dagger"}


def test_a_held_out_track_is_refused(demo):
    heldout = Track.from_json(TRACKS / "hairpins_01.json")
    network = init_network(IMITATION.spec, make_rng(0))
    with pytest.raises(ValueError, match="held-out"):
        collect(network, heldout, make_rng(0), n_starts=1, target_laps=1)
    with pytest.raises(ValueError, match="held-out"):
        DaggerLearner(SETTINGS, demo, heldout, make_rng(0))


@pytest.mark.parametrize("field", ["epochs_per_round", "random_starts", "target_laps"])
def test_bad_settings_are_refused(field):
    with pytest.raises(ValueError, match=field):
        DaggerSettings(**{field: 0})


# ── the learner ─────────────────────────────────────────────────────────────────────────────


def test_round_one_is_behavioural_cloning(demo, track):
    dagger = learner(demo, track)
    assert isinstance(dagger, Learner) and isinstance(dagger, Resumable)
    assert (dagger.name, dagger.feedback_kind) == ("dagger", "labels")
    cloning = ImitationLearner(IMITATION, demo, make_rng(0))
    clone_results = [cloning.train_unit() for _ in range(SETTINGS.epochs_per_round)]
    result = dagger.train_unit()
    assert dagger.network.equals(cloning.network)  # bit for bit: same split, init and batches
    assert dagger.val_episodes == cloning.val_episodes
    assert (result.demo_frames, result.env_steps, result.teacher_queries) == (demo.n_frames, 0, 0)
    assert all(result.metrics[k] is None for k in COLLECT_METRICS)
    assert result.metrics["val_loss"] == clone_results[-1].metrics["val_loss"]
    assert result.metrics["loss_before"] == clone_results[0].metrics["loss_before"]
    assert result.metrics["epochs"] == SETTINGS.epochs_per_round


def test_rounds_aggregate_and_count_every_query(demo, track):
    dagger = learner(demo, track)
    base = dagger.x_train.shape[0]
    results = [dagger.train_unit() for _ in range(3)]
    queries = [r.teacher_queries for r in results]
    assert queries[0] == 0 and all(q > 0 for q in queries[1:])
    assert [r.env_steps for r in results] == queries
    assert [r.demo_frames for r in results] == [demo.n_frames, 0, 0]
    assert [r.metrics["collect_frames"] for r in results[1:]] == queries[1:]
    # D = D_0 followed by every collected frame, in order; nothing dropped or replaced.
    assert dagger.queries == sum(queries)
    assert [r.metrics["train_frames"] for r in results] == [
        base,
        base + queries[1],
        base + queries[1] + queries[2],
    ]
    assert np.array_equal(dagger.x_train[:base], dagger.base_x)
    assert np.array_equal(dagger.x_train[base:], dagger.collected["observation"])
    assert np.array_equal(dagger.y_train[base:], dagger.collected["label"])
    assert dagger.loader.n_examples == dagger.x_train.shape[0]
    assert [e["index"] for e in dagger.episodes] == list(range(2 * SETTINGS.random_starts))
    assert [e["round"] for e in dagger.episodes] == [2] * 3 + [3] * 3
    assert results[-1].metrics["epochs"] == 3 * SETTINGS.epochs_per_round
    for r in results[1:]:
        assert 0.0 <= r.metrics["onpolicy_loss"] and 0 <= r.metrics["collect_finished"] <= 1


def test_the_collection_starts_come_from_their_own_stream(demo, track):
    """Round 2 drives from the first R starts of the "collect" generator; other seeds differ."""
    dagger = learner(demo, track)
    dagger.train_unit()
    network = dagger.network.clone()
    dagger.train_unit()
    expected = collect(
        network,
        track,
        child_rng(make_rng(0), "collect"),
        n_starts=SETTINGS.random_starts,
        target_laps=2,
    )
    assert np.array_equal(dagger.collected["observation"], expected.frames["observation"])
    other = learner(demo, track, seed=1)
    other.train_unit()
    other.train_unit()
    starts = [e["start_s0_m"] for e in dagger.episodes]
    assert starts != [e["start_s0_m"] for e in other.episodes]


def test_divergence_in_a_round_keeps_the_last_rounds_network(demo, track, monkeypatch):
    dagger = learner(demo, track)
    dagger.train_unit()
    before = dagger.network.clone()
    original = DaggerLearner._epoch
    calls = []

    def second_epoch_breaks(self):
        calls.append(1)
        return None if len(calls) == 2 else original(self)

    monkeypatch.setattr(DaggerLearner, "_epoch", second_epoch_breaks)
    result = dagger.train_unit()  # epoch 1 of round 2 changes the network, epoch 2 diverges
    assert result.diverged and result.metrics["loss"] is None
    assert result.teacher_queries == result.env_steps > 0  # the queries were still spent
    assert dagger.network.equals(before)


def test_state_and_restore_continue_exactly(demo, track):
    reference = learner(demo, track)
    ref = [reference.train_unit() for _ in range(3)]
    first = learner(demo, track)
    for _ in range(2):
        first.train_unit()
    saved = first.state()
    resumed = learner(demo, track)
    resumed.restore(saved)
    third = resumed.train_unit()
    assert third.metrics == ref[2].metrics
    assert (third.teacher_queries, third.env_steps) == (ref[2].teacher_queries, ref[2].env_steps)
    assert resumed.network.equals(reference.network)
    a, b = resumed.state(), reference.state()
    assert a.info == b.info
    assert a.arrays.keys() == b.arrays.keys()
    assert all(np.array_equal(a.arrays[k], b.arrays[k]) for k in a.arrays)


# ── the collected frames as a dataset (decision D4A) ────────────────────────────────────────


def test_the_collected_frames_save_as_a_dataset(demo, track, tmp_path):
    dagger = learner(demo, track)
    dagger.train_unit()
    assert collected_dataset(dagger) is None  # round 1 collects nothing
    dagger.train_unit()
    dagger.train_unit()
    dataset = collected_dataset(dagger)
    assert dataset is not None and dataset.n_frames == dagger.queries
    assert dataset.meta["teacher"]["kind"] == "dagger"
    assert dataset.meta["demo_frames"] == dataset.meta["teacher_queries"] == dagger.queries
    assert dataset.meta["dagger"]["base_dataset"] == "20261110-120000-expert-t"
    folder = save_dataset(dataset, tmp_path, tag="t")
    assert folder.name.endswith("-dagger-t")
    loaded = DemoDataset.load(folder)
    assert loaded.identical_to(dataset)
    check_trainable(loaded)  # plain imitation could learn from it
    assert len(loaded.meta["episodes"]) == 2 * SETTINGS.random_starts


# ── wider collection starts (task 6.6 follow-up, ADR-050) ──────────────────────────────────


def test_the_start_ranges_reach_the_world(track):
    network = init_network(IMITATION.spec, make_rng(3))
    wide = collect(
        network, track, make_rng(5), n_starts=20, target_laps=1, lateral_m=3.0, heading_rad=0.35
    )
    lateral = np.array([e["start_lateral_m"] for e in wide.episodes])
    heading = np.array([e["start_heading_offset_rad"] for e in wide.episodes])
    assert np.all(np.abs(lateral) <= 3.0) and np.max(np.abs(lateral)) > 1.0
    assert np.all(np.abs(heading) <= 0.35) and np.max(np.abs(heading)) > 0.1
    # The defaults are the dataset's recipe, so 6.6's collections are unchanged.
    default = collect(network, track, make_rng(5), n_starts=3, target_laps=1)
    explicit = collect(
        network, track, make_rng(5), n_starts=3, target_laps=1, lateral_m=1.0, heading_rad=0.1
    )
    assert all(np.array_equal(default.frames[k], explicit.frames[k]) for k in default.frames)


def test_the_learner_collects_from_its_start_ranges(demo, track):
    settings = DaggerSettings(
        imitation=IMITATION,
        epochs_per_round=1,
        random_starts=12,
        start_lateral_m=3.0,
        start_heading_rad=0.35,
    )
    dagger = learner(demo, track, settings=settings)
    dagger.train_unit()
    dagger.train_unit()
    heading = [abs(e["start_heading_offset_rad"]) for e in dagger.episodes]
    assert max(heading) > 0.1 and max(heading) <= 0.35
    starts = collected_dataset(dagger).meta["starts"]
    assert (starts["random_lateral_m"], starts["random_heading_rad"]) == (3.0, 0.35)


def test_negative_start_ranges_are_refused():
    with pytest.raises(ValueError, match="start_heading_rad"):
        DaggerSettings(start_heading_rad=-0.1)


def test_the_expert_records_from_wide_starts(track):
    wide = record_expert(
        track,
        start_fractions=(),
        lateral_offsets=(),
        random_starts=12,
        seed=1,
        target_laps=1,
        random_lateral_m=3.0,
        random_heading_rad=0.35,
    )
    heading = [abs(e["start_heading_offset_rad"]) for e in wide.meta["episodes"]]
    assert max(heading) > 0.1 and max(heading) <= 0.35
    assert wide.meta["starts"]["random_heading_rad"] == 0.35
