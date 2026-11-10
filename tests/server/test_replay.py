"""Tests for replaying checkpoints and the run timeline's endpoints (task 5.9, ADR-042, api v1.14),
and for replays from shifted starts (task 6.5, ADR-048, api v1.18).

The central check: a checkpoint loaded as a live driver and played to the end drives exactly the
exam drive its card was made from, car for car, to the last digit. From shifted starts it drives
exactly what Study S5b's own drive of the same network from the same starts gives.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.env.episode import run_episode
from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.policies import MLPPolicy
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig
from neurodrive.experiment.config import load_config, parse_config
from neurodrive.experiment.exam import shifted_starts
from neurodrive.experiment.runner import train_random_search_run, train_toy_run
from neurodrive.experiment.store import RunStore
from neurodrive.server.app import create_app
from neurodrive.server.live import LiveSession, TrackLibrary
from neurodrive.server.runs import RunLibrary

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ROOT / "tracks"
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    """A runs folder with a small random-search run (exams on), a toy run and a run without
    checkpoints (like the scripted-expert baseline)."""
    folder = tmp_path_factory.mktemp("runs")
    config = parse_config(
        {
            "method": "random_search",
            "seed": 4,  # new bests at units 1 to 5: units 3 and 4 are off the fixed schedule
            "tag": "replay",
            "search": {"candidates_per_unit": 4},
            "budget": {"units": 6},
        }
    )
    train_random_search_run(config, folder)
    train_toy_run(load_config(CONFIGS / "toy_xor.yaml", epochs=3, tag="replay"), folder)
    RunStore.create(folder, "scripted_expert", "replay", {}).finish("completed", "units")
    return folder


def run_of(runs: Path, method: str) -> RunStore:
    (path,) = [p for p in runs.iterdir() if f"-{method}-" in p.name]
    return RunStore.open(path)


def session_for(runs: Path) -> LiveSession:
    return LiveSession(TrackLibrary(TRACKS), RunLibrary(runs))


def send(session: LiveSession, **command) -> list:
    return session.handle_text(json.dumps(command))


def load_checkpoint(session: LiveSession, run: str, unit: int, track: str) -> list:
    return send(
        session,
        type="load_scenario",
        track=track,
        policy="checkpoint",
        checkpoint={"run": run, "unit": unit},
    )


# ── the replay is the exam drive ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("track", ["train_loop_01", "hairpins_01"])
def test_a_replayed_checkpoint_drives_exactly_its_exam_drive(runs, track):
    store = run_of(runs, "random_search")
    units = store.exam_units()
    assert units == store.checkpoint_units() and len(units) >= 4
    for unit in units:
        session = session_for(runs)
        replies = load_checkpoint(session, store.path.name, unit, track)
        scenario = replies[0]
        assert scenario.type == "scenario" and scenario.policy == "checkpoint"
        assert scenario.checkpoint.run == store.path.name and scenario.checkpoint.unit == unit
        assert scenario.n_cars == 10 and scenario.seed is None and scenario.target_laps == 2
        network = replies[1]
        saved = store.load_checkpoint(unit)
        assert network.type == "network"
        for sent, w in zip(network.weights, saved.weights, strict=True):
            assert np.array_equal(sent, np.round(w, 4))
        world = session._world
        session._advance(10**7)  # play to the end
        assert world.done
        (part,) = [t for t in store.load_exam(unit).to_dict()["tracks"] if t["track"] == track]
        for car, episode in enumerate(part["episodes"]):
            assert episode["start_index"] == car  # car k is exam start k
            assert Status(int(world.status[car])).name.lower() == episode["status"]
            assert float(world.progress[car]) == episode["progress_m"]  # to the last digit
        assert world.step_count == max(e["steps"] for e in part["episodes"])


def test_the_replay_offers_what_every_network_driver_offers(runs):
    store = run_of(runs, "random_search")
    session = session_for(runs)
    load_checkpoint(session, store.path.name, store.checkpoint_units()[-1], "train_loop_01")
    (frame,) = send(session, type="step")
    assert frame.selected.trace is not None and frame.selected.explain is not None
    (answer,) = send(session, type="what_if", overrides={"0": 1.0})
    assert answer.type == "explanation"
    replies = send(session, type="select_car", index=4)
    assert [r.type for r in replies] == ["network", "frame"]  # same network for every car
    (reset,) = send(session, type="reset")
    assert reset.step == 0 and len(reset.cars.x) == 10


def test_refused_checkpoints_keep_the_loaded_scenario(runs):
    search, toy, expert = (run_of(runs, m) for m in ("random_search", "toy_xor", "scripted_expert"))
    session = session_for(runs)
    send(session, type="load_scenario", track="train_loop_01", policy="constant", n_cars=3)
    cases = [
        ({"run": "20261102-000000-random_search-nope", "unit": 0}, "unknown run"),
        ({"run": "../runs", "unit": 0}, "unknown run"),
        ({"run": search.path.name, "unit": 3000}, "has no checkpoint 3000"),
        ({"run": toy.path.name, "unit": 0}, "not a driver"),
        ({"run": expert.path.name, "unit": 0}, "has no checkpoint 0"),
    ]
    for checkpoint, problem in cases:
        (reply,) = send(
            session,
            type="load_scenario",
            track="train_loop_01",
            policy="checkpoint",
            checkpoint=checkpoint,
        )
        assert reply.type == "error" and problem in reply.message, reply.message
        assert session._scenario.policy == "constant" and len(session._world.cars.x) == 3
    (reply,) = send(session, type="load_scenario", track="train_loop_01", policy="checkpoint")
    assert reply.type == "error" and "needs checkpoint" in reply.message
    no_runs = LiveSession(TrackLibrary(TRACKS))
    (reply,) = load_checkpoint(no_runs, search.path.name, 0, "train_loop_01")
    assert reply.type == "error" and "no run folders" in reply.message


# ── shifted starts (task 6.5, ADR-048) ────────────────────────────────────────────────────────


def drive_shifted(track_name: str, policy_for, lateral: float, heading: float) -> World:
    """The study's drive, made without the server: the 10 shifted exam starts as one batch."""
    track = Track.from_json(TRACKS / f"{track_name}.json")
    world = World(track, WorldConfig.for_track(track, 2))
    s0, lat, psi = shifted_starts(track.length, lateral, heading)
    world.reset(s0=s0, lateral=lat, heading_offset=psi)
    run_episode(world, policy_for(world))
    return world


def assert_same_ending(session: LiveSession, direct: World) -> None:
    world = session._world
    session._advance(10**7)  # play to the end
    assert world.done and world.step_count == direct.step_count
    assert world.status.tolist() == direct.status.tolist()
    assert world.progress.tolist() == direct.progress.tolist()  # to the last digit


@pytest.mark.parametrize("shift", [(-3.0, 0.2618), (1.5, -0.5236)])
def test_a_shifted_replay_drives_exactly_the_studys_drive(runs, shift):
    store = run_of(runs, "random_search")
    unit = store.checkpoint_units()[-1]
    session = session_for(runs)
    lateral, heading = shift
    replies = send(
        session,
        type="load_scenario",
        track="train_loop_01",
        policy="checkpoint",
        checkpoint={"run": store.path.name, "unit": unit},
        start_shift={"lateral_m": lateral, "heading_rad": heading},
    )
    scenario = replies[0]
    assert scenario.type == "scenario" and scenario.n_cars == 10
    assert (scenario.start_shift.lateral_m, scenario.start_shift.heading_rad) == shift
    start = session._world.start
    assert start["lateral"].tolist() == [lateral] * 5 + [1.5 + lateral] * 5
    assert start["heading_offset"].tolist() == [heading] * 10
    network = store.load_checkpoint(unit)
    direct = drive_shifted("train_loop_01", lambda w: MLPPolicy(network), lateral, heading)
    assert_same_ending(session, direct)
    (reset,) = send(session, type="reset")  # Reset keeps the shift
    assert session._world.start["heading_offset"].tolist() == [heading] * 10


def test_the_expert_drives_the_same_shifted_starts(runs):
    session = session_for(runs)
    replies = send(
        session,
        type="load_scenario",
        track="train_loop_01",
        policy="expert",
        n_cars=3,
        seed=5,
        start_shift={"lateral_m": -3.0, "heading_rad": 0.2618},
    )
    assert replies[0].n_cars == 10 and replies[0].seed is None
    assert_same_ending(session, drive_shifted("train_loop_01", ScriptedExpert, -3.0, 0.2618))


def test_a_shift_without_the_checkpoint_or_expert_driver_is_refused(runs):
    session = session_for(runs)
    send(session, type="load_scenario", track="train_loop_01", policy="constant", n_cars=3)
    shift = {"lateral_m": 1.0, "heading_rad": 0.0}
    (reply,) = send(
        session, type="load_scenario", track="train_loop_01", policy="neuron", start_shift=shift
    )
    assert reply.type == "error" and "start_shift" in reply.message
    assert session._scenario.policy == "constant" and len(session._world.cars.x) == 3
    for bad in ({"lateral_m": 6.0, "heading_rad": 0.0}, {"lateral_m": 0.0, "heading_rad": 2.0}):
        (reply,) = send(
            session, type="load_scenario", track="train_loop_01", policy="expert", start_shift=bad
        )
        assert reply.type == "error"
        assert session._scenario.policy == "constant"


# ── new bests get a checkpoint (D2) ────────────────────────────────────────────────────────────


def test_every_new_best_of_random_search_has_a_checkpoint_and_an_exam(runs):
    store = run_of(runs, "random_search")
    improved = [m["unit"] for m in store.read_metrics() if m["improved"]]
    assert improved == [1, 2, 3, 4, 5]
    assert store.checkpoint_units() == [0, 1, 2, 3, 4, 5, 6]  # 3 and 4 only because they improved
    assert store.exam_units() == store.checkpoint_units()
    # Each new best is a different network; after the last one the kept network stays the same.
    nets = [store.load_checkpoint(u) for u in improved]
    assert not any(a.equals(b) for a, b in zip(nets, nets[1:], strict=False))
    assert store.load_checkpoint(6).equals(nets[-1])


# ── diffs and exam cards over REST ─────────────────────────────────────────────────────────────


@pytest.fixture
def diff_run(tmp_path) -> RunStore:
    """The Brief's worked example: three weights go from [0.5, -1, 2] to [0.75, -1, 1]."""
    store = RunStore.create(tmp_path, "hand", "diff", {})
    spec = NetworkSpec(layer_sizes=(3, 1), output_activation="linear")
    store.save_checkpoint(0, Network(spec, [[[0.5, -1.0, 2.0]]], [[0.1]]))
    store.save_checkpoint(1, Network(spec, [[[0.75, -1.0, 1.0]]], [[0.1]]))
    other = NetworkSpec(layer_sizes=(3, 2), output_activation="linear")
    store.save_checkpoint(2, Network(other, [np.zeros((2, 3))], [np.zeros(2)]))
    return store


def test_the_diff_of_the_worked_example(diff_run):
    with TestClient(create_app(runs_dir=diff_run.path.parent)) as client:
        base = f"/api/runs/{diff_run.path.name}/checkpoints"
        answer = client.get(f"{base}/1/diff", params={"from": 0})
        assert answer.status_code == 200, answer.text
        d = answer.json()
        assert (d["unit"], d["from_unit"], d["layer_sizes"]) == (1, 0, [3, 1])
        assert d["weight_delta"] == [[[0.25, 0.0, -1.0]]] and d["bias_delta"] == [[0.0]]
        assert d["max_abs_delta"] == 1.0
        assert d["weight_change"] == [[[0.25, 0.0, -1.0]]] and d["bias_change"] == [[0.0]]
        backwards = client.get(f"{base}/0/diff", params={"from": 1}).json()
        assert backwards["weight_change"] == [[[-0.25, 0.0, 1.0]]]
        same = client.get(f"{base}/1/diff", params={"from": 1}).json()
        assert same["max_abs_delta"] == 0.0 and same["weight_change"] == [[[0.0, 0.0, 0.0]]]
        sizes = client.get(f"{base}/2/diff", params={"from": 0})
        assert sizes.status_code == 400 and "different layer sizes" in sizes.text
        assert client.get(f"{base}/9/diff", params={"from": 0}).status_code == 404
        assert client.get(f"{base}/1/diff").status_code == 422  # "from" is required
        unknown = "/api/runs/20261102-000000-hand-nope/checkpoints/1/diff?from=0"
        assert client.get(unknown).status_code == 404


def test_exam_cards_are_served_as_stored(runs):
    store = run_of(runs, "random_search")
    unit = store.exam_units()[-1]
    with TestClient(create_app(runs_dir=runs)) as client:
        answer = client.get(f"/api/runs/{store.path.name}/exams/{unit}")
        assert answer.status_code == 200
        assert answer.json()["card"] == store.load_exam(unit).to_dict()
        assert client.get(f"/api/runs/{store.path.name}/exams/999").status_code == 404
        toy = run_of(runs, "toy_xor")
        assert client.get(f"/api/runs/{toy.path.name}/exams/0").status_code == 404


def test_a_checkpoint_replay_over_the_live_socket(runs):
    store = run_of(runs, "random_search")
    with TestClient(create_app(runs_dir=runs)) as client:
        with client.websocket_connect("/ws/live") as ws:
            assert json.loads(ws.receive_text())["api"] == "v1.18"
            ws.send_text(
                json.dumps(
                    {
                        "type": "load_scenario",
                        "track": "sweeper_01",
                        "policy": "checkpoint",
                        "checkpoint": {"run": store.path.name, "unit": 0},
                    }
                )
            )
            scenario = json.loads(ws.receive_text())
            assert scenario["policy"] == "checkpoint" and scenario["n_cars"] == 10
            assert scenario["checkpoint"] == {"run": store.path.name, "unit": 0}
            assert json.loads(ws.receive_text())["type"] == "network"
            assert len(json.loads(ws.receive_text())["cars"]["x"]) == 10
