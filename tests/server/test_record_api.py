"""Tests for keyboard demonstrations in the live session (api v1.20, task 6.1b, ADR-051)."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodrive.env.sensors import edge_grid, observe
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig
from neurodrive.learn.supervised.demos import DemoDataset, find_dataset, straight_share
from neurodrive.server.app import create_app
from neurodrive.server.live import LiveSession, TrackLibrary
from neurodrive.server.schemas import (
    DatasetDetail,
    DatasetSavedMessage,
    EpisodeEndMessage,
    ErrorMessage,
    RecordingMessage,
)

TRACKS = Path(__file__).resolve().parents[2] / "tracks"
TRACK = "train_loop_01"


def send(session: LiveSession, **command) -> list:
    return session.handle_text(json.dumps(command))


def session_in(tmp_path, *, track=TRACK, policy="keyboard", n_cars=1, seed=None) -> LiveSession:
    session = LiveSession(TrackLibrary(TRACKS), datasets_dir=tmp_path)
    send(session, type="load_scenario", track=track, policy=policy, n_cars=n_cars, seed=seed)
    return session


def drive(session: LiveSession, keys: list[tuple[float, float]]) -> list:
    """One step per entry, holding that key input during the step; return all messages."""
    out = []
    for steer, throttle in keys:
        send(session, type="key_state", steer=steer, throttle=throttle)
        out += send(session, type="step")
    return out


def errors(messages) -> list[str]:
    return [m.message for m in messages if isinstance(m, ErrorMessage)]


# ── the frames are the session's steps ──────────────────────────────────────────────────────


def test_frames_equal_the_steps_of_the_session(tmp_path):
    """Observation rows = the observations the session decided on (bit for bit); labels = the
    clipped keys; one frame per step."""
    session = session_in(tmp_path)
    (recording,) = send(session, type="record_start")
    assert isinstance(recording, RecordingMessage) and recording.on and recording.frames == 0
    keys = [(0.0, 1.0)] * 10 + [(1.0, 1.0)] * 5 + [(-0.7, 0.4)] * 5 + [(0.0, -1.0)] * 5
    drive(session, keys)
    messages = send(session, type="record_stop", save=True, tag="unit")
    saved = next(m for m in messages if isinstance(m, DatasetSavedMessage))
    assert saved.frames == len(keys) and saved.episodes == 1

    dataset = DemoDataset.load(tmp_path / saved.dataset_id)
    memory = session._memory
    assert dataset.n_frames == memory.end == len(keys)
    for t in range(len(keys)):
        assert np.array_equal(dataset.observation[t], memory.observation[t % memory.capacity, 0])
        assert np.array_equal(dataset.label[t], memory.action[t % memory.capacity, 0])
        assert dataset.x[t] == memory.x[t % memory.capacity, 0]
    assert np.array_equal(dataset.label, np.clip(np.array(keys), -1.0, 1.0))
    assert dataset.meta["teacher"]["kind"] == "keyboard" and dataset.meta["demo_frames"] == 25
    assert list(dataset.episode) == [0] * 25 and list(dataset.step) == list(range(25))


def test_worked_example_a_crash_then_a_cut_short_drive_gives_3_plus_5_style_counts(tmp_path):
    """Drive 1 ends in a crash; Reset; drive 2 is cut short after 5 steps: F = n1 + 5."""
    session = session_in(tmp_path)
    send(session, type="record_start")
    messages = []
    for _ in range(400):  # full lock and full throttle leaves the road
        messages = drive(session, [(1.0, 1.0)])
        if any(isinstance(m, EpisodeEndMessage) for m in messages):
            break
    end = next(m for m in messages if isinstance(m, EpisodeEndMessage))
    assert end.status == ["crashed"]
    n1 = end.step  # the car drove steps 0 .. n1 - 1
    send(session, type="reset")
    drive(session, [(0.0, 1.0)] * 5)
    messages = send(session, type="record_stop", save=True, tag="two")
    saved = next(m for m in messages if isinstance(m, DatasetSavedMessage))
    assert saved.frames == n1 + 5 and saved.episodes == 2
    meta = DemoDataset.load(tmp_path / saved.dataset_id).meta
    assert [e["status"] for e in meta["episodes"]] == ["crashed", "stopped"]
    assert [e["frames"] for e in meta["episodes"]] == [n1, 5]
    assert all(e["start_kind"] == "live" for e in meta["episodes"])
    assert meta["env_steps"] == meta["demo_frames"] == n1 + 5


def test_each_episode_replays_from_its_stored_start(tmp_path):
    """Alone in a fresh world from the stored start, the stored labels drive the car through the
    stored observations bit for bit, and it ends with the stored status."""
    session = session_in(tmp_path, seed=3)  # a seeded random start
    send(session, type="record_start")
    for _ in range(400):
        messages = drive(session, [(1.0, 1.0)])
        if any(isinstance(m, EpisodeEndMessage) for m in messages):
            break
    send(session, type="reset")
    drive(session, [(0.3, 1.0)] * 12)
    messages = send(session, type="record_stop", save=True, tag="replay")
    saved = next(m for m in messages if isinstance(m, DatasetSavedMessage))
    dataset = DemoDataset.load(tmp_path / saved.dataset_id)
    assert dataset.meta["starts"] == {"kind": "live", "seed": 3}

    track = Track.from_json(TRACKS / f"{TRACK}.json")
    segments = edge_grid(track)
    for e in dataset.meta["episodes"]:
        rows = dataset.episode == e["index"]
        world = World(track, WorldConfig.for_track(track))
        world.reset(
            s0=[e["start_s0_m"]],
            lateral=[e["start_lateral_m"]],
            heading_offset=[e["start_heading_offset_rad"]],
            speed=[e["start_speed_mps"]],
        )
        for obs, label in zip(dataset.observation[rows], dataset.label[rows], strict=True):
            assert world.running[0]
            assert np.array_equal(observe(world, segments)[0], obs)
            world.step(label[:1], label[1:])
        if e["status"] == "stopped":
            assert world.running[0]  # cut short: still driving
        else:
            assert Status(int(world.status[0])).name.lower() == e["status"]


def test_a_reset_keeps_the_drive_so_far_and_an_empty_drive_is_dropped(tmp_path):
    session = session_in(tmp_path)
    send(session, type="record_start")
    drive(session, [(0.0, 1.0)] * 4)
    (_, rec) = send(session, type="reset")
    assert rec.on and rec.frames == 4 and rec.episodes == 1
    send(session, type="reset")  # nothing driven since: no empty episode
    drive(session, [(0.0, 1.0)] * 2)
    saved = next(
        m
        for m in send(session, type="record_stop", save=True, tag="r")
        if isinstance(m, DatasetSavedMessage)
    )
    meta = DemoDataset.load(tmp_path / saved.dataset_id).meta
    assert [e["frames"] for e in meta["episodes"]] == [4, 2]
    assert [e["index"] for e in meta["episodes"]] == [0, 1]


def test_recording_status_comes_with_every_frame(tmp_path):
    session = session_in(tmp_path)
    send(session, type="record_start")
    first = drive(session, [(0.0, 1.0)])
    second = drive(session, [(0.0, 1.0)])
    assert [m for m in first if isinstance(m, RecordingMessage)][0].frames == 1
    assert [m for m in second if isinstance(m, RecordingMessage)][0].frames == 2


# ── refusals ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "word"),
    [
        ({"n_cars": 2}, "one car"),
        ({"track": "hairpins_01"}, "held-out"),
        ({"policy": "constant"}, "keyboard"),
        ({"policy": "expert"}, "keyboard"),
    ],
)
def test_record_start_is_refused_and_nothing_is_recorded(tmp_path, kwargs, word):
    session = session_in(tmp_path, **kwargs)
    (reply,) = send(session, type="record_start")
    assert isinstance(reply, ErrorMessage) and word in reply.message
    assert errors(send(session, type="record_stop", save=True, tag="x")) == ["not recording"]
    assert list(tmp_path.iterdir()) == []


def test_record_start_needs_step_zero_and_a_datasets_folder(tmp_path):
    session = session_in(tmp_path)
    drive(session, [(0.0, 1.0)])
    assert "step 0" in errors(send(session, type="record_start"))[0]
    send(session, type="reset")
    assert not errors(send(session, type="record_start"))
    assert errors(send(session, type="record_start")) == ["already recording"]
    bare = LiveSession(TrackLibrary(TRACKS))
    send(bare, type="load_scenario", track=TRACK)
    assert "datasets folder" in errors(send(bare, type="record_start"))[0]


def test_saving_nothing_is_refused_and_discard_writes_nothing(tmp_path):
    session = session_in(tmp_path)
    send(session, type="record_start")
    assert "nothing recorded" in errors(send(session, type="record_stop", save=True, tag="x"))[0]
    drive(session, [(0.0, 1.0)] * 3)  # the recording is still open
    (off,) = send(session, type="record_stop", save=False)
    assert isinstance(off, RecordingMessage) and not off.on
    assert list(tmp_path.iterdir()) == []
    assert errors(send(session, type="record_stop", save=False)) == ["not recording"]


def test_loading_another_scenario_while_recording_is_refused(tmp_path):
    session = session_in(tmp_path)
    send(session, type="record_start")
    assert "recording is open" in errors(send(session, type="load_scenario", track=TRACK))[0]
    send(session, type="record_stop", save=False)
    assert not errors(send(session, type="load_scenario", track=TRACK))


def test_a_bad_tag_is_an_invalid_message(tmp_path):
    session = session_in(tmp_path)
    send(session, type="record_start")
    drive(session, [(0.0, 1.0)] * 2)
    reply = errors(send(session, type="record_stop", save=True, tag="../evil"))
    assert reply and "invalid message" in reply[0]
    assert list(tmp_path.iterdir()) == []


# ── the saved file ──────────────────────────────────────────────────────────────────────────


def test_the_dataset_is_found_listed_and_described_like_any_other(tmp_path):
    app = create_app(datasets_dir=tmp_path, runs_dir=tmp_path / "runs")
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_text()  # hello

        def talk(**command):
            ws.send_text(json.dumps(command))

        talk(type="load_scenario", track=TRACK, policy="keyboard", n_cars=1)
        assert json.loads(ws.receive_text())["type"] == "scenario"
        assert json.loads(ws.receive_text())["type"] == "frame"
        talk(type="record_start")
        assert json.loads(ws.receive_text()) == {
            "type": "recording",
            "on": True,
            "episodes": 0,
            "frames": 0,
        }
        talk(type="key_state", steer=0.0, throttle=1.0)
        for _ in range(6):
            talk(type="step")
            assert json.loads(ws.receive_text())["type"] == "frame"
            assert json.loads(ws.receive_text())["type"] == "recording"
        talk(type="record_stop", save=True, tag="me")
        saved = json.loads(ws.receive_text())
        assert saved["type"] == "dataset_saved" and saved["frames"] == 6 and saved["episodes"] == 1
        assert json.loads(ws.receive_text())["on"] is False

    path = find_dataset(tmp_path, "keyboard-me")
    assert path.name == saved["dataset_id"]
    listed = client.get("/api/datasets").json()["datasets"]
    assert [d["dataset_id"] for d in listed] == [path.name]
    assert listed[0]["teacher"] == "keyboard" and listed[0]["demo_frames"] == 6
    detail = DatasetDetail.model_validate(client.get(f"/api/datasets/{path.name}").json())
    assert sum(detail.labels[0].counts) == 6
    dataset = DemoDataset.load(path)
    assert dataset.meta["straight_share"] == straight_share(dataset.label) == 1.0
    assert dataset.meta["git_commit"] is None or len(dataset.meta["git_commit"]) == 40
