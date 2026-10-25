"""Tests for live sessions: the clock logic with a hand-driven clock, then the real WebSocket."""

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodrive.server.app import create_app
from neurodrive.server.live import LiveSession, TrackLibrary
from neurodrive.server.schemas import (
    API_VERSION,
    EpisodeEndMessage,
    ErrorMessage,
    ExplanationMessage,
    FrameMessage,
    NetworkMessage,
    ScenarioMessage,
)

TRACKS = Path(__file__).resolve().parents[2] / "tracks"
TRACK = "train_loop_01"


def send(session: LiveSession, **command) -> list:
    return session.handle_text(json.dumps(command))


def loaded(policy: str = "keyboard", n_cars: int = 1, seed: int | None = None) -> LiveSession:
    session = LiveSession(TrackLibrary(TRACKS))
    send(session, type="load_scenario", track=TRACK, policy=policy, n_cars=n_cars, seed=seed)
    return session


def run_clock(session: LiveSession, seconds: float, tick: float = 0.001):
    """Tick a playing session every ``tick`` seconds; return [(time, message), ...] sent."""
    sent = []
    now = 100.0
    session.tick(now)  # the first tick only starts the clock
    for _ in range(round(seconds / tick)):
        now += tick
        for message in session.tick(now):
            sent.append((now, message))
    return sent


# ── loading and the first frame ─────────────────────────────────────────────────────────────


def test_load_scenario_answers_with_scenario_then_frame():
    session = LiveSession(TrackLibrary(TRACKS))
    scenario, frame = send(session, type="load_scenario", track=TRACK, n_cars=3)
    assert isinstance(scenario, ScenarioMessage) and isinstance(frame, FrameMessage)
    assert scenario.track == TRACK and scenario.n_cars == 3 and scenario.policy == "keyboard"
    assert scenario.dt == pytest.approx(1 / 30) and scenario.ray_range == 50.0
    assert len(scenario.ray_angles) == 5 and scenario.target_laps == 2
    assert scenario.time_limit == 92.0 and scenario.v_max == 30.0  # 2 laps at 16 m/s (ADR-022)
    assert frame.step == 0 and frame.time == 0.0 and not frame.playing and frame.speed == 1.0
    assert len(frame.cars.x) == len(frame.cars.action) == len(frame.cars.status) == 3
    assert frame.cars.alive == [True] * 3 and frame.cars.status == [0, 0, 0]
    assert frame.cars.x == [0.0] * 3 and frame.cars.action == [[0.0, 0.0]] * 3
    assert frame.selected is not None and frame.selected.index == 0
    assert len(frame.selected.rays) == 5 and len(frame.selected.observation) == 7
    assert frame.selected.rays[0] == pytest.approx(7.0, abs=0.2)  # far-left ray, about 6 / sin 60


def test_commands_before_a_scenario_are_errors_but_key_state_is_ignored():
    session = LiveSession(TrackLibrary(TRACKS))
    for command in ({"type": "play"}, {"type": "step"}, {"type": "reset"}, {"type": "pause"}):
        (reply,) = send(session, **command)
        assert isinstance(reply, ErrorMessage) and "load_scenario" in reply.message
    assert send(session, type="key_state", steer=1.0, throttle=1.0) == []
    assert session.tick(5.0) == []


def test_bad_messages_give_errors_and_leave_the_session_usable():
    session = LiveSession(TrackLibrary(TRACKS))
    bad = [
        "not json at all",
        json.dumps({"type": "fly"}),
        json.dumps({"no_type": 1}),
        json.dumps({"type": "load_scenario", "track": TRACK, "colour": "red"}),
        json.dumps({"type": "load_scenario", "track": TRACK, "n_cars": 0}),
        json.dumps({"type": "load_scenario", "track": "no_such_track"}),
        json.dumps({"type": "load_scenario", "track": "../README"}),
    ]
    for text in bad:
        (reply,) = session.handle_text(text)
        assert isinstance(reply, ErrorMessage), text
    scenario, _ = send(session, type="load_scenario", track=TRACK)
    assert isinstance(scenario, ScenarioMessage)
    for command in (
        {"type": "set_speed", "speed": 50},
        {"type": "set_speed", "speed": 0},
        {"type": "key_state", "steer": 2, "throttle": 0},
        {"type": "select_car", "index": -1},
        {"type": "select_car", "index": 4},
    ):
        (reply,) = send(session, **command)
        assert isinstance(reply, ErrorMessage), command


# ── step, pause, play, reset ────────────────────────────────────────────────────────────────


def test_step_advances_exactly_one_physics_step_each_time():
    session = loaded()
    for expected in (1, 2, 3):
        (frame,) = send(session, type="step")
        assert frame.step == expected and frame.time == pytest.approx(expected / 30, abs=1e-4)
    assert frame.cars.alive == [True]


def test_pause_stops_the_clock_and_play_resumes_it():
    session = loaded("constant")
    assert run_clock(session, 0.5) == []  # paused: no frames, no progress
    (frame,) = send(session, type="play")
    assert frame.playing
    sent = run_clock(session, 0.5)
    assert sent and all(isinstance(m, FrameMessage) and m.playing for _, m in sent)
    at_pause = sent[-1][1].step
    (frame,) = send(session, type="pause")
    assert not frame.playing and frame.step >= at_pause
    assert run_clock(session, 0.5) == []
    (frame2,) = send(session, type="step")
    assert frame2.step == frame.step + 1


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0, 8.0])
def test_frame_rate_never_exceeds_thirty_per_second_at_any_speed(speed):
    session = loaded("constant")
    send(session, type="set_speed", speed=speed)
    send(session, type="play")
    seconds = 1.0 if speed <= 2.0 else 0.4  # keep the constant-throttle car on the road
    sent = run_clock(session, seconds)
    frames = [(t, m) for t, m in sent if isinstance(m, FrameMessage)]
    assert len(frames) <= 30 * seconds + 1
    times = [t for t, _ in frames]
    for start in times:  # any window of one second holds at most 31 frames
        assert sum(start <= t < start + 1.0 for t in times) <= 31
    # The world keeps up with real time: 30 * speed steps per second, whatever the frame rate.
    steps = frames[-1][1].step
    assert steps == pytest.approx(30 * speed * seconds, rel=0.1, abs=3)
    # Physics step size never changes: simulated time is always steps / 30.
    assert frames[-1][1].time == pytest.approx(steps / 30, abs=1e-3)


def test_at_double_speed_each_frame_covers_two_steps():
    session = loaded("constant")
    send(session, type="set_speed", speed=2.0)
    send(session, type="play")
    frames = [m for _, m in run_clock(session, 0.5) if isinstance(m, FrameMessage)]
    jumps = {b.step - a.step for a, b in zip(frames, frames[1:], strict=False)}
    assert jumps == {2}


def test_at_half_speed_each_frame_is_one_step_at_fifteen_frames_per_second():
    session = loaded("constant")
    send(session, type="set_speed", speed=0.5)
    send(session, type="play")
    sent = run_clock(session, 2.0)
    frames = [(t, m) for t, m in sent if isinstance(m, FrameMessage)]
    assert {b[1].step - a[1].step for a, b in zip(frames, frames[1:], strict=False)} == {1}
    assert 28 <= len(frames) <= 31  # 15 per second for 2 seconds


def test_a_long_stall_does_not_cause_a_burst_of_catch_up_steps():
    session = loaded("constant")
    send(session, type="play")
    session.tick(0.0)
    messages = session.tick(60.0)  # one real minute passes in a single tick
    assert len(messages) == 1 and messages[0].step <= 8  # at most 0.25 s of simulation owed


def test_reset_returns_to_step_zero_and_keeps_the_play_state():
    session = loaded("constant")
    for _ in range(5):
        send(session, type="step")
    (frame,) = send(session, type="reset")
    assert frame.step == 0 and frame.cars.x == [0.0] and not frame.playing
    send(session, type="play")
    run_clock(session, 0.3)
    (frame,) = send(session, type="reset")
    assert frame.step == 0 and frame.playing  # playing before, still playing after


def test_seeded_scenarios_start_at_reproducible_random_places():
    def first_frame(seed):
        session = LiveSession(TrackLibrary(TRACKS))
        return send(session, type="load_scenario", track=TRACK, n_cars=4, seed=seed)[1]

    a, b, c = first_frame(3), first_frame(3), first_frame(4)
    assert a.cars == b.cars and a.cars != c.cars
    assert len(set(a.cars.x)) == 4  # four different start places
    session = LiveSession(TrackLibrary(TRACKS))
    send(session, type="load_scenario", track=TRACK, n_cars=4, seed=3)
    send(session, type="step")
    (after_reset,) = send(session, type="reset")
    assert after_reset.cars == a.cars  # reset gives the same seeded start again


# ── driving: keyboard, constant policy, selection ───────────────────────────────────────────


def test_key_state_drives_a_keyboard_car_and_changes_what_happens():
    def psi_after(steer: float, throttle: float) -> tuple[float, float]:
        session = loaded("keyboard")
        send(session, type="key_state", steer=steer, throttle=throttle)
        for _ in range(20):
            (frame,) = send(session, type="step")
        return frame.cars.psi[0], frame.cars.v[0]

    straight_psi, straight_v = psi_after(0.0, 1.0)
    right_psi, _ = psi_after(1.0, 1.0)
    left_psi, _ = psi_after(-1.0, 1.0)
    assert straight_v > 4.0  # full throttle for 20 steps
    assert right_psi < straight_psi < left_psi  # positive steer turns right (heading decreases)


def test_the_last_action_applied_appears_in_the_frame():
    session = loaded("keyboard")
    send(session, type="key_state", steer=0.5, throttle=-0.25)
    (frame,) = send(session, type="step")
    assert frame.cars.action == [[0.5, -0.25]]


def test_constant_policy_ignores_key_state():
    session = loaded("constant")
    send(session, type="key_state", steer=1.0, throttle=-1.0)
    (frame,) = send(session, type="step")
    assert frame.cars.action == [[0.0, 0.3]]


def test_select_car_changes_the_detail_shown_and_only_that_car():
    session = loaded("constant", n_cars=3, seed=9)
    (frame,) = send(session, type="select_car", index=2)
    assert frame.selected.index == 2
    other = loaded("constant", n_cars=3, seed=9)
    send(other, type="select_car", index=1)
    (frame1,) = send(other, type="step")
    (frame2,) = send(session, type="step")
    assert frame1.selected.observation != frame2.selected.observation  # different cars
    assert frame2.selected.observation[5] == pytest.approx(frame2.cars.v[2] / 30, abs=1e-3)


def test_numbers_on_the_wire_are_rounded():
    session = loaded("keyboard", n_cars=2, seed=1)
    send(session, type="key_state", steer=0.3, throttle=0.9)
    for _ in range(7):
        (frame,) = send(session, type="step")
    for value in frame.cars.x + frame.cars.y + frame.cars.v + frame.cars.progress:
        assert value == round(value, 3)
    for value in frame.cars.psi + frame.selected.observation:
        assert value == round(value, 4)


# ── the end of an episode ───────────────────────────────────────────────────────────────────


def test_episode_end_is_sent_once_then_the_session_pauses():
    session = loaded("keyboard")  # no keys pressed: the car sits still and stalls after 2 s
    send(session, type="play")
    sent = run_clock(session, 4.0)
    ends = [m for _, m in sent if isinstance(m, EpisodeEndMessage)]
    assert len(ends) == 1
    assert ends[0].status == ["stalled"] and ends[0].step == 60 and ends[0].laps == [0]
    final_frame = [m for _, m in sent if isinstance(m, FrameMessage)][-1]
    assert final_frame.step == 60 and not final_frame.playing
    assert final_frame.cars.alive == [False] and final_frame.cars.status == [3]
    assert not session.playing
    (reply,) = send(session, type="play")
    assert isinstance(reply, ErrorMessage) and "reset" in reply.message
    (reply,) = send(session, type="step")
    assert isinstance(reply, ErrorMessage)
    (frame,) = send(session, type="reset")
    assert frame.step == 0 and frame.cars.alive == [True]
    (frame,) = send(session, type="step")
    assert frame.step == 1


def test_a_step_that_ends_the_episode_also_sends_episode_end():
    session = loaded("keyboard")
    messages = []
    for _ in range(60):
        messages = send(session, type="step")
    assert isinstance(messages[0], FrameMessage) and isinstance(messages[1], EpisodeEndMessage)


# ── determinism ─────────────────────────────────────────────────────────────────────────────


def test_same_commands_and_clock_give_identical_frames():
    def run():
        session = loaded("keyboard", n_cars=3, seed=5)
        send(session, type="key_state", steer=0.2, throttle=0.7)
        send(session, type="set_speed", speed=2.0)
        send(session, type="play")
        return [m.model_dump_json() for _, m in run_clock(session, 1.0)]

    assert run() == run()


# ── the track library ───────────────────────────────────────────────────────────────────────


def test_track_library_lists_loads_and_refuses_unknown_names(tmp_path):
    shutil.copy(TRACKS / "train_loop_01.json", tmp_path / "copy_of_loop.json")
    library = TrackLibrary(tmp_path)
    assert library.names() == ["copy_of_loop"]
    assert [t.name for t in library.info()] == ["copy_of_loop"]
    with pytest.raises(KeyError):
        library.get("train_loop_01")
    with pytest.raises(KeyError):
        library.get("../tracks/train_loop_01")


def test_track_geometry_has_matching_closed_loops():
    geometry = TrackLibrary(TRACKS).geometry(TRACK)
    n = len(geometry.centerline)
    assert n == len(geometry.left) == len(geometry.right) and 700 < n < 800
    assert geometry.width == 12.0 and geometry.split == "train"
    assert geometry.length == pytest.approx(730.7, abs=0.1) and geometry.start_s0 == 0.0
    assert geometry.centerline[0] == [0.0, 0.0]
    assert all(len(p) == 2 for p in geometry.left)


# ── over a real WebSocket and REST ──────────────────────────────────────────────────────────


def test_rest_track_endpoints():
    client = TestClient(create_app())
    listing = client.get("/api/tracks").json()
    assert listing == {
        "tracks": [
            {"name": "monza_ccw", "split": "heldout"},
            {"name": "monza_cw", "split": "train"},
            {"name": TRACK, "split": "train"},
        ]
    }
    geometry = client.get(f"/api/tracks/{TRACK}").json()
    assert set(geometry) == {
        "name", "split", "width", "length", "start_s0", "centerline", "left", "right",
    }  # fmt: skip
    assert len(geometry["left"]) == len(geometry["centerline"])
    assert client.get("/api/tracks/nope").status_code == 404


def test_websocket_session_end_to_end():
    client = TestClient(create_app())
    with client.websocket_connect("/ws/live") as ws:
        assert json.loads(ws.receive_text())["api"] == API_VERSION == "v1.10"
        ws.send_text(json.dumps({"type": "load_scenario", "track": TRACK, "policy": "constant"}))
        assert json.loads(ws.receive_text())["type"] == "scenario"
        assert json.loads(ws.receive_text())["step"] == 0
        ws.send_text(json.dumps({"type": "step"}))
        frame = json.loads(ws.receive_text())
        assert frame["type"] == "frame" and frame["step"] == 1
        ws.send_text("garbage")
        assert json.loads(ws.receive_text())["type"] == "error"  # connection stays open
        ws.send_text(json.dumps({"type": "step"}))
        assert json.loads(ws.receive_text())["step"] == 2


def test_websocket_play_streams_frames_until_paused():
    client = TestClient(create_app())
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "load_scenario", "track": TRACK, "policy": "constant"}))
        ws.receive_text(), ws.receive_text()
        ws.send_text(json.dumps({"type": "play"}))
        steps = []
        while len(steps) < 5:
            message = json.loads(ws.receive_text())
            if message["type"] == "frame" and message["step"] > 0:
                steps.append(message["step"])
        assert steps == sorted(steps) and len(set(steps)) == 5
        ws.send_text(json.dumps({"type": "pause"}))
        while True:  # frames already in flight may arrive first
            message = json.loads(ws.receive_text())
            if message["type"] == "frame" and not message["playing"]:
                break
        paused_at = message["step"]
        ws.send_text(json.dumps({"type": "step"}))
        assert json.loads(ws.receive_text())["step"] == paused_at + 1


def test_each_connection_has_its_own_session():
    client = TestClient(create_app())
    with client.websocket_connect("/ws/live") as a, client.websocket_connect("/ws/live") as b:
        for ws in (a, b):
            ws.receive_text()
            ws.send_text(json.dumps({"type": "load_scenario", "track": TRACK}))
            ws.receive_text(), ws.receive_text()
        for _ in range(3):
            a.send_text(json.dumps({"type": "step"}))
            frame_a = json.loads(a.receive_text())
        b.send_text(json.dumps({"type": "step"}))
        frame_b = json.loads(b.receive_text())
        assert frame_a["step"] == 3 and frame_b["step"] == 1


# ── the neuron driver: readout, set_param, activation curves (api v1.3) ─────────────────────


def neuron_session(n_cars: int = 1, seed: int | None = None) -> LiveSession:
    return loaded("neuron", n_cars=n_cars, seed=seed)


def set_param(session: LiveSession, name: str, value) -> list:
    return send(session, type="set_param", name=name, value=value)


def test_neuron_scenario_starts_with_zero_weights_and_a_readout_in_every_frame():
    session = LiveSession(TrackLibrary(TRACKS))
    scenario, frame = send(session, type="load_scenario", track=TRACK, policy="neuron")
    assert scenario.policy == "neuron"
    neuron = frame.selected.neuron
    assert neuron.weights == [0.0, 0.0] and neuron.bias == 0.0
    assert neuron.activation == "tanh" and neuron.throttle == 0.15
    assert neuron.input_rays == [0, 4] and len(neuron.inputs) == 2
    assert neuron.inputs == [frame.selected.observation[0], frame.selected.observation[4]]
    assert neuron.z == 0.0 and neuron.output == 0.0 and neuron.contributions == [0.0, 0.0]


def test_the_other_drivers_have_no_neuron_readout():
    for policy in ("keyboard", "constant"):
        (frame,) = send(loaded(policy), type="step")
        assert frame.selected.neuron is None


def test_set_param_answers_with_a_frame_showing_the_new_numbers_even_while_paused():
    session = neuron_session()
    (frame,) = set_param(session, "bias", 1.0)
    neuron = frame.selected.neuron
    assert isinstance(frame, FrameMessage) and not frame.playing and frame.step == 0
    assert neuron.bias == 1.0 and neuron.z == pytest.approx(1.0)
    assert neuron.output == pytest.approx(0.7616, abs=1e-4)  # tanh(1)
    (frame,) = set_param(session, "activation", "sigmoid")
    assert frame.selected.neuron.activation == "sigmoid"
    assert frame.selected.neuron.output == pytest.approx(0.7311, abs=1e-4)  # sigmoid(1)
    (frame,) = set_param(session, "bias", -1.0)
    assert frame.selected.neuron.output == pytest.approx(0.2689, abs=1e-4)  # sigmoid(-1) > 0


def test_a_weight_change_shows_in_the_very_next_steering_action():
    session = neuron_session()
    (frame,) = send(session, type="step")
    assert frame.cars.action == [[0.0, 0.15]]  # zero weights: straight
    set_param(session, "bias", 0.5)
    (frame,) = send(session, type="step")
    assert frame.cars.action[0][0] == pytest.approx(0.4621, abs=1e-4)  # tanh(0.5)
    set_param(session, "bias", -0.5)
    (frame,) = send(session, type="step")
    assert frame.cars.action[0][0] == pytest.approx(-0.4621, abs=1e-4)


def test_the_readout_matches_the_hand_example_and_its_own_sums():
    session = neuron_session()
    for name, value in (("w_left", -4.0), ("w_right", 4.0), ("bias", 0.1)):
        (frame,) = set_param(session, name, value)
    n = frame.selected.neuron
    assert n.contributions == pytest.approx(
        [w * x for w, x in zip(n.weights, n.inputs, strict=True)], abs=1e-3
    )
    assert n.z == pytest.approx(sum(n.contributions) + n.bias, abs=2e-4)
    assert n.output == pytest.approx(math.tanh(n.z), abs=2e-4)


def test_the_readout_follows_the_selected_car_and_all_cars_share_the_neuron():
    session = neuron_session(n_cars=3, seed=4)
    set_param(session, "w_left", -3.0)
    (frame0,) = send(session, type="select_car", index=0)
    (frame2,) = send(session, type="select_car", index=2)
    assert frame0.selected.neuron.weights == frame2.selected.neuron.weights == [-3.0, 0.0]
    assert (
        frame0.selected.neuron.inputs != frame2.selected.neuron.inputs
    )  # different cars see different walls


def test_set_param_errors_leave_the_session_and_the_neuron_unchanged():
    session = neuron_session()
    set_param(session, "w_left", 2.0)
    for name, value in (("w_left", 99.0), ("activation", "gelu"), ("throttle", 5.0), ("bias", "x")):
        (reply,) = set_param(session, name, value)
        assert isinstance(reply, ErrorMessage), (name, value)
    (frame,) = send(session, type="step")
    assert frame.selected.neuron.weights[0] == 2.0


def test_set_param_needs_the_neuron_driver_and_a_valid_message():
    for policy in ("keyboard", "constant"):
        (reply,) = set_param(loaded(policy), "bias", 1.0)
        assert isinstance(reply, ErrorMessage) and "neuron" in reply.message
    (reply,) = set_param(LiveSession(TrackLibrary(TRACKS)), "bias", 1.0)
    assert isinstance(reply, ErrorMessage) and "load_scenario" in reply.message
    session = neuron_session()
    for command in (
        {"type": "set_param", "name": "speed", "value": 1.0},
        {"type": "set_param", "name": "bias"},
        {"type": "set_param", "value": 1.0},
    ):
        (reply,) = send(session, **command)
        assert isinstance(reply, ErrorMessage), command


def test_neuron_settings_survive_reset_and_every_scenario_change_in_a_session():
    session = neuron_session()
    set_param(session, "w_left", -7.0)
    set_param(session, "w_right", 7.0)
    set_param(session, "bias", 0.2)
    set_param(session, "activation", "sigmoid")
    set_param(session, "throttle", 0.3)
    expected = ([-7.0, 7.0], 0.2, "sigmoid", 0.3)

    def settings(frame):
        n = frame.selected.neuron
        return (n.weights, n.bias, n.activation, n.throttle)

    (frame,) = send(session, type="reset")
    assert settings(frame) == expected
    for change in (
        {"track": TRACK, "policy": "neuron", "n_cars": 4},
        {"track": TRACK, "policy": "neuron", "n_cars": 4, "seed": 3},
        {"track": TRACK, "policy": "neuron"},
    ):
        _, frame = send(session, type="load_scenario", **change)
        assert settings(frame) == expected, change
    # A detour through another driver keeps them too.
    _, frame = send(session, type="load_scenario", track=TRACK, policy="keyboard")
    assert frame.selected.neuron is None
    _, frame = send(session, type="load_scenario", track=TRACK, policy="neuron")
    assert settings(frame) == expected


def test_a_new_session_starts_its_neuron_at_zero():
    neuron_session_a = neuron_session()
    set_param(neuron_session_a, "w_left", -7.0)
    (frame,) = send(neuron_session(), type="step")  # another connection, another session
    assert frame.selected.neuron.weights == [0.0, 0.0]


def test_a_neuron_with_opposite_sign_weights_keeps_the_car_on_the_road_live():
    session = neuron_session()
    set_param(session, "w_left", -8.0)
    set_param(session, "w_right", 8.0)
    send(session, type="set_speed", speed=8.0)
    send(session, type="play")
    sent = run_clock(session, 2.0)  # 16 s of driving at 8x
    frames = [m for _, m in sent if isinstance(m, FrameMessage)]
    assert frames[-1].cars.alive == [True] and frames[-1].cars.progress[0] > 60.0
    assert not [m for _, m in sent if isinstance(m, EpisodeEndMessage)]


def test_activation_curves_endpoint_returns_python_computed_curves():
    client = TestClient(create_app())
    body = client.get("/api/activations").json()
    z = body["z"]
    assert len(z) == 101 and z[0] == -5.0 and z[-1] == 5.0 and z[50] == 0.0
    assert set(body["curves"]) == {"tanh", "sigmoid", "relu", "linear"}
    assert all(len(v) == 101 for v in body["curves"].values())
    curves = body["curves"]
    assert curves["tanh"][50] == 0.0 and curves["sigmoid"][50] == 0.5
    assert min(curves["tanh"]) > -1.0 and max(curves["tanh"]) < 1.0
    assert min(curves["sigmoid"]) > 0.0 and max(curves["sigmoid"]) < 1.0  # never negative
    assert min(curves["relu"]) == 0.0 and curves["relu"][-1] == 5.0
    assert curves["linear"] == z


def test_websocket_set_param_end_to_end():
    client = TestClient(create_app())
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_text()
        ws.send_text(json.dumps({"type": "load_scenario", "track": TRACK, "policy": "neuron"}))
        ws.receive_text(), ws.receive_text()
        ws.send_text(json.dumps({"type": "set_param", "name": "bias", "value": 1.0}))
        frame = json.loads(ws.receive_text())
        assert frame["selected"]["neuron"]["bias"] == 1.0
        ws.send_text(json.dumps({"type": "set_param", "name": "activation", "value": "sigmoid"}))
        assert json.loads(ws.receive_text())["selected"]["neuron"]["activation"] == "sigmoid"


# ── random networks: one brain per car (api v1.4) ───────────────────────────────────────────


def drive_steps(session: LiveSession, steps: int) -> FrameMessage:
    frame = None
    for _ in range(steps):
        frame = send(session, type="step")[0]
    return frame


def test_random_networks_scenario_loads_20_cars_and_reports_the_policy():
    session = LiveSession(TrackLibrary(TRACKS))
    scenario, network, frame = send(
        session, type="load_scenario", track=TRACK, policy="random_networks", n_cars=20, seed=7
    )
    assert isinstance(network, NetworkMessage) and network.car == 0
    assert scenario.policy == "random_networks" and scenario.n_cars == 20 and scenario.seed == 7
    assert len(frame.cars.x) == 20 and frame.selected.neuron is None
    (error,) = send(session, type="set_param", name="bias", value=1.0)
    assert isinstance(error, ErrorMessage) and "neuron" in error.message


def test_random_networks_drive_differently_and_every_action_is_in_range():
    session = loaded("random_networks", n_cars=20)  # no seed: all start on the line, seed 0 brains
    frame = drive_steps(session, 30)
    actions = frame.cars.action
    assert len({tuple(a) for a in actions}) >= 15  # the cars choose differently
    assert all(-1.0 <= v <= 1.0 for a in actions for v in a)
    assert len(set(frame.cars.v)) >= 5  # and the cars have different speeds after one second


def test_same_seed_same_drive_different_seed_different_drive():
    a = drive_steps(loaded("random_networks", n_cars=20, seed=5), 40)
    b = drive_steps(loaded("random_networks", n_cars=20, seed=5), 40)
    c = drive_steps(loaded("random_networks", n_cars=20, seed=6), 40)
    assert a.cars == b.cars
    assert a.cars != c.cars


def test_reset_replays_the_same_brains_and_starts():
    session = loaded("random_networks", n_cars=20, seed=2)
    first = drive_steps(session, 40)
    (reset_frame,) = send(session, type="reset")
    assert reset_frame.step == 0
    assert drive_steps(session, 40).cars == first.cars


def test_seed_none_means_seed_0_brains_and_no_random_starts():
    none = loaded("random_networks", n_cars=5, seed=None)
    zero = loaded("random_networks", n_cars=5, seed=0)
    # seed None: every car starts on the start line; seed 0: seeded random starts
    assert len(set(send(none, type="reset")[0].cars.x)) == 1
    assert len(set(send(zero, type="reset")[0].cars.x)) == 5


def test_a_single_random_network_car_works():
    frame = drive_steps(loaded("random_networks", n_cars=1, seed=3), 10)
    assert len(frame.cars.x) == 1


def test_selecting_another_car_shows_that_cars_rays():
    session = loaded("random_networks", n_cars=20, seed=4)
    drive_steps(session, 20)
    first = send(session, type="select_car", index=0)[-1].selected
    other = send(session, type="select_car", index=7)[-1].selected
    assert other.index == 7 and other.observation != first.observation


def test_the_seed_picks_the_networks_themselves():
    # Compared directly, because a seed also changes the random starts and would hide a brain bug.
    def brains(seed):
        return loaded("random_networks", n_cars=4, seed=seed)._policy.population.genomes()

    assert (brains(5) == brains(5)).all()
    assert not (brains(5) == brains(6)).all()
    assert (brains(None) == brains(0)).all()  # seed None means seed 0
    assert brains(5).shape == (4, 434)
    assert len({tuple(row[:3]) for row in brains(5)}) == 4  # four different networks


# ── the selected car's network: weights once, the trace in every frame (api v1.5) ───────────


def network_of(session: LiveSession, car: int) -> NetworkMessage:
    (message, _frame) = send(session, type="select_car", index=car)
    assert isinstance(message, NetworkMessage)
    return message


def forward_from_message(network: NetworkMessage, observation: list[float]) -> list[float]:
    """The network recomputed from the message's own numbers (what a client could check)."""
    a = np.array(observation)
    for w, b, name in zip(network.weights, network.biases, network.activations, strict=True):
        z = np.array(w) @ a + np.array(b)
        a = {"tanh": np.tanh, "linear": lambda v: v}[name](z)
    return a.tolist()


def test_network_message_comes_after_load_and_after_select_car():
    session = LiveSession(TrackLibrary(TRACKS))
    scenario, network, frame = send(
        session, type="load_scenario", track=TRACK, policy="random_networks", n_cars=20, seed=3
    )
    assert isinstance(scenario, ScenarioMessage) and isinstance(frame, FrameMessage)
    assert network.type == "network" and network.car == 0
    assert network.layer_sizes == [7, 16, 16, 2] and network.activations == ["tanh"] * 3
    assert [len(w) for w in network.weights] == [16, 16, 2]
    assert [len(w[0]) for w in network.weights] == [7, 16, 16]
    assert [len(b) for b in network.biases] == [16, 16, 2]
    other = network_of(session, 5)
    assert other.car == 5 and other.weights != network.weights  # a different brain


def test_network_message_holds_that_cars_rounded_weights():
    session = loaded("random_networks", n_cars=6, seed=1)
    message = network_of(session, 4)
    mine = session._policy.population.network(4)
    np.testing.assert_allclose(message.weights[0], mine.weights[0], atol=5e-5)
    np.testing.assert_allclose(message.biases[2], mine.biases[2], atol=5e-5)


def test_the_frame_trace_has_every_layer_and_starts_with_the_observation():
    session = loaded("random_networks", n_cars=20, seed=2)
    frame = drive_steps(session, 15)
    trace = frame.selected.trace
    assert [len(a) for a in trace.activations] == [7, 16, 16, 2]
    assert [len(z) for z in trace.weighted_sums] == [16, 16, 2]
    assert trace.activations[0] == frame.selected.observation  # the input nodes are the rays etc.
    assert all(-1.0 <= v <= 1.0 for a in trace.activations[1:] for v in a)  # tanh layers


def test_the_trace_is_the_selected_cars_forward_pass_of_its_own_network():
    session = loaded("random_networks", n_cars=20, seed=2)
    for car in (0, 9, 19):
        network = network_of(session, car)
        frame = drive_steps(session, 5)
        assert frame.selected.index == car
        recomputed = forward_from_message(network, frame.selected.observation)
        assert frame.selected.trace.activations[-1] == pytest.approx(recomputed, abs=2e-3)


def test_the_trace_follows_the_current_step_and_the_selected_car():
    session = loaded("random_networks", n_cars=20, seed=2)
    early = drive_steps(session, 2).selected.trace.activations[0]
    later = drive_steps(session, 10).selected.trace.activations[0]
    assert early != later  # the inputs changed as the car moved
    network_of(session, 6)
    (frame,) = send(session, type="pause")
    assert frame.selected.index == 6
    assert frame.selected.trace.activations[0] == frame.selected.observation


def test_other_drivers_send_no_network_and_no_trace():
    for policy in ("keyboard", "constant", "neuron"):
        session = LiveSession(TrackLibrary(TRACKS))
        replies = send(session, type="load_scenario", track=TRACK, policy=policy, n_cars=3)
        assert [type(m).__name__ for m in replies] == ["ScenarioMessage", "FrameMessage"]
        assert replies[1].selected.trace is None
        (frame,) = send(session, type="select_car", index=1)
        assert isinstance(frame, FrameMessage) and frame.selected.trace is None


def test_select_car_out_of_range_sends_only_an_error():
    session = loaded("random_networks", n_cars=3, seed=1)
    (reply,) = send(session, type="select_car", index=3)
    assert isinstance(reply, ErrorMessage)


# ── the output inspector and what-if (api v1.6) ─────────────────────────────────────────────


def explained_session(car: int = 4, steps: int = 12) -> LiveSession:
    session = loaded("random_networks", n_cars=20, seed=None)
    network_of(session, car)
    drive_steps(session, steps)
    return session


def test_every_frame_explains_the_selected_cars_outputs_with_the_explain_module():
    from neurodrive.core.nn.forward import forward
    from neurodrive.explain.contributions import neuron_contributions
    from neurodrive.explain.sensitivity import sensitivity

    session = explained_session()
    (frame,) = send(session, type="pause")
    explain = frame.selected.explain
    assert [r.output for r in explain.receipts] == [0, 1]
    assert len(explain.sensitivity) == 2 and all(len(row) == 7 for row in explain.sensitivity)
    network = session._policy.network_of(4)
    obs = session._observe()[4]
    trace = forward(network, obs)
    for m, receipt in enumerate(explain.receipts):
        r = neuron_contributions(network, trace, 2, m)
        np.testing.assert_allclose(receipt.contributions, r.contributions, atol=5e-5)
        assert receipt.bias == pytest.approx(r.bias, abs=5e-5)
        assert receipt.z == pytest.approx(r.z, abs=5e-5)
        assert receipt.value == pytest.approx(r.output, abs=5e-5)
        assert receipt.value == frame.selected.trace.activations[-1][m]  # same number as the trace
    np.testing.assert_allclose(explain.sensitivity, sensitivity(network, obs), atol=5e-5)


def test_the_receipt_adds_up_to_z():
    (frame,) = send(explained_session(), type="pause")
    for receipt in frame.selected.explain.receipts:
        assert sum(receipt.contributions) + receipt.bias == pytest.approx(receipt.z, abs=2e-3)


def test_other_drivers_get_no_explanation():
    for policy in ("keyboard", "constant", "neuron"):
        frame = drive_steps(loaded(policy, n_cars=2), 3)
        assert frame.selected.explain is None


def test_what_if_with_no_overrides_is_exactly_the_real_output():
    session = explained_session()
    (frame,) = send(session, type="pause")
    (answer,) = send(session, type="what_if", overrides={})
    assert isinstance(answer, ExplanationMessage)
    assert answer.car == 4 and answer.step == frame.step
    assert answer.observation == frame.selected.observation
    assert answer.trace.activations[-1] == answer.base_outputs  # bit-identical, then rounded
    assert answer.predicted_outputs == answer.base_outputs
    assert answer.base_outputs == pytest.approx(frame.selected.trace.activations[-1], abs=1e-4)


def test_what_if_changes_only_the_overridden_inputs():
    session = explained_session()
    (frame,) = send(session, type="pause")
    (answer,) = send(session, type="what_if", overrides={"2": 0.75, "6": -0.5})
    expected = list(frame.selected.observation)
    expected[2], expected[6] = 0.75, -0.5
    assert answer.observation == expected


def test_what_if_on_the_left_ray_moves_steer_as_the_sensitivity_predicts():
    session = explained_session()
    (frame,) = send(session, type="pause")
    s_left = frame.selected.explain.sensitivity[0][0]  # steer with respect to the far-left ray
    (answer,) = send(session, type="what_if", overrides={0: frame.selected.observation[0] + 0.1})
    change = answer.trace.activations[-1][0] - answer.base_outputs[0]
    assert answer.predicted_outputs[0] - answer.base_outputs[0] == pytest.approx(
        0.1 * s_left, abs=2e-4
    )
    assert abs(s_left) > 0.01  # this car's steer does react to the left ray
    assert np.sign(change) == np.sign(s_left)
    assert change == pytest.approx(0.1 * s_left, rel=0.5)  # roughly the predicted size


def test_what_if_changes_nothing_in_the_simulation():
    session = explained_session()
    (before,) = send(session, type="pause")
    send(session, type="what_if", overrides=dict.fromkeys(range(7), 0.0))
    (after,) = send(session, type="pause")
    assert after.step == before.step and after.cars == before.cars
    assert after.selected.trace == before.selected.trace


def test_what_if_is_refused_while_playing_and_for_other_drivers():
    session = explained_session()
    send(session, type="play")
    (error,) = send(session, type="what_if", overrides={0: 0.5})
    assert isinstance(error, ErrorMessage) and "pause" in error.message
    neuron = loaded("neuron")
    (error,) = send(neuron, type="what_if", overrides={0: 0.5})
    assert isinstance(error, ErrorMessage) and "network driver" in error.message


def test_what_if_needs_existing_inputs_and_finite_values():
    session = explained_session()
    (error,) = send(session, type="what_if", overrides={7: 0.5})
    assert isinstance(error, ErrorMessage) and "input 7" in error.message
    (error,) = send(session, type="what_if", overrides={-1: 0.5})
    assert isinstance(error, ErrorMessage)
    bad = session.handle_text('{"type": "what_if", "overrides": {"3": NaN}}')
    assert isinstance(bad[0], ErrorMessage)
    (old_form,) = send(session, type="what_if", observation=[0.5] * 7)  # the v1.6 field is gone
    assert isinstance(old_form, ErrorMessage)


# ── the what-if ghost car (api v1.7) ─────────────────────────────────────────────────────────


def ghost_for(session: LiveSession, change: dict[int, float]):
    (frame,) = send(session, type="pause")
    (answer,) = send(session, type="what_if", overrides=change)
    return frame, answer.ghost, answer


def test_ghost_paths_start_at_the_car_and_last_two_seconds():
    session = explained_session()
    frame, ghost, _ = ghost_for(session, {0: 0.9})
    i = frame.selected.index
    start = [frame.cars.x[i], frame.cars.y[i]]
    assert ghost.seconds == 2.0
    assert len(ghost.real) == len(ghost.what_if) == 61  # now + 60 steps of 1/30 s
    assert ghost.real[0] == pytest.approx(start, abs=1e-3)
    assert ghost.what_if[0] == pytest.approx(start, abs=1e-3)
    assert ghost.real_end[:2] == ghost.real[-1] and ghost.what_if_end[:2] == ghost.what_if[-1]


def test_ghost_actions_are_the_networks_answers_clipped():
    session = explained_session()
    _, ghost, answer = ghost_for(session, {0: 0.9, 6: 0.5})
    assert ghost.real_action == pytest.approx(answer.base_outputs, abs=1e-4)
    assert ghost.what_if_action == pytest.approx(answer.trace.activations[-1], abs=1e-4)
    assert all(-1.0 <= a <= 1.0 for a in ghost.real_action + ghost.what_if_action)


def test_ghost_paths_follow_the_car_physics():
    from neurodrive.env.car import CarState, step_cars

    session = explained_session()
    frame, ghost, _ = ghost_for(session, {6: -0.8})
    i = frame.selected.index
    cars = session._world.cars
    state = CarState.from_pose(cars.x[i], cars.y[i], cars.psi[i], cars.v[i])
    steer, throttle = ghost.what_if_action
    for _ in range(60):
        state = step_cars(state, steer, throttle)
    assert ghost.what_if[-1] == pytest.approx([state.x[0], state.y[0]], abs=2e-3)


def test_unchanged_inputs_give_the_same_two_paths_and_a_change_moves_the_ghost():
    session = explained_session()
    frame, same, _ = ghost_for(session, {})
    assert same.what_if == same.real
    _, moved, _ = ghost_for(session, {6: 0.9})  # heading error far to one side
    assert moved.what_if[-1] != moved.real[-1]
    assert moved.real == same.real  # the real path does not depend on the question


def test_ghost_never_moves_the_real_car():
    session = explained_session()
    before, _, _ = ghost_for(session, {0: 1.0, 4: 0.0})
    (after,) = send(session, type="pause")
    assert after.cars == before.cars and after.step == before.step


# ── decision trail and scrubbing (task 2.11, ADR-027) ────────────────────────────────────────


def trail_session(car: int = 4, steps: int = 30) -> tuple[LiveSession, dict[int, FrameMessage]]:
    """20 random networks, car ``car`` selected at step 0, then ``steps`` steps; frames by step."""
    session = loaded("random_networks", n_cars=20, seed=None)
    (_net, first) = send(session, type="select_car", index=car)
    frames = {first.step: first}
    for _ in range(steps):
        frame = send(session, type="step")[0]
        frames[frame.step] = frame
    return session, frames


def browser_trail(frames) -> dict[int, tuple[float, float, list[float]]]:
    """Rebuild the trail the way the browser does: replace, then append."""
    trail: dict[int, tuple[float, float, list[float]]] = {}
    for frame in frames:
        t = frame.selected.trail
        if t.replace:
            trail = {}
        for k, s in enumerate(t.steps):
            trail[s] = (t.x[k], t.y[k], t.action[k])
    return trail


def test_the_trail_is_the_cars_real_path_and_the_actions_it_took():
    session, frames = trail_session()
    trail = browser_trail(frames[s] for s in sorted(frames))
    assert sorted(trail) == list(range(30))  # one point per step taken, nothing else
    for t, (x, y, action) in trail.items():
        assert (x, y) == (frames[t].cars.x[4], frames[t].cars.y[4])  # where it was at step t
        assert action == frames[t + 1].cars.action[4]  # what it decided there


def test_a_new_car_gets_its_whole_trail_and_later_frames_only_new_points():
    session, frames = trail_session()
    assert frames[0].selected.trail.replace and frames[0].selected.trail.steps == []
    assert all(not frames[t].selected.trail.replace for t in range(1, 31))
    assert all(frames[t].selected.trail.steps == [t - 1] for t in range(1, 31))
    (_net, other) = send(session, type="select_car", index=7)
    assert other.selected.trail.replace and other.selected.trail.steps == list(range(30))
    assert other.selected.trail.x[5] == frames[5].cars.x[7]
    (again,) = send(session, type="pause")
    assert not again.selected.trail.replace and again.selected.trail.steps == []


def test_at_double_speed_each_frame_carries_the_two_steps_it_covers():
    session, _ = trail_session(steps=0)
    send(session, type="set_speed", speed=2)
    send(session, type="play")
    frames = [m for _, m in run_clock(session, 1.0) if isinstance(m, FrameMessage)]
    for frame in frames:
        assert frame.selected.trail.steps == [frame.step - 2, frame.step - 1]


def test_inspect_step_gives_exactly_what_the_live_frame_showed_at_that_step():
    session, frames = trail_session()
    for t in range(30):
        (moment,) = send(session, type="inspect_step", step=t)
        live = frames[t]
        assert moment.car == 4 and moment.step == t and moment.time == live.time
        assert (moment.x, moment.y, moment.psi, moment.v) == (
            live.cars.x[4], live.cars.y[4], live.cars.psi[4], live.cars.v[4]
        )  # fmt: skip
        assert moment.progress == live.cars.progress[4] and moment.status == live.cars.status[4]
        assert moment.rays == live.selected.rays
        assert moment.observation == live.selected.observation
        assert moment.trace == live.selected.trace  # bit-identical, then rounded the same way
        assert moment.explain == live.selected.explain
        assert moment.action == frames[t + 1].cars.action[4]
        assert moment.trace.activations[-1] == pytest.approx(moment.action, abs=1e-3)


def test_inspect_step_never_moves_the_simulation():
    session, frames = trail_session()
    send(session, type="inspect_step", step=3)
    (after,) = send(session, type="pause")
    assert after.step == 30 and after.cars == frames[30].cars


def test_inspect_step_needs_a_pause_and_a_remembered_step():
    session, _ = trail_session(steps=5)
    (error,) = send(session, type="inspect_step", step=5)  # step 5 has not been decided yet
    assert isinstance(error, ErrorMessage) and "steps 0 to 4" in error.message
    send(session, type="play")
    (error,) = send(session, type="inspect_step", step=2)
    assert isinstance(error, ErrorMessage) and "pause" in error.message
    send(session, type="pause")
    send(session, type="reset")
    (error,) = send(session, type="inspect_step", step=2)  # a reset forgets the old drive
    assert isinstance(error, ErrorMessage) and "nothing yet" in error.message


def test_only_the_last_trail_seconds_are_remembered(monkeypatch):
    import neurodrive.server.live as live

    monkeypatch.setattr(live, "TRAIL_SECONDS", 0.5)  # 15 steps
    session, _ = trail_session(steps=40)
    (_net, frame) = send(session, type="select_car", index=4)
    assert frame.selected.trail.steps == list(range(25, 40))
    assert isinstance(send(session, type="inspect_step", step=24)[0], ErrorMessage)
    assert send(session, type="inspect_step", step=25)[0].step == 25


def test_a_crashed_cars_trail_stops_where_it_crashed():
    session = loaded("random_networks", n_cars=20, seed=None)
    drive_steps(session, 200)
    (frame,) = send(session, type="pause")
    car = frame.cars.status.index(1)  # the first crashed car
    (_net, frame) = send(session, type="select_car", index=car)
    steps = frame.selected.trail.steps
    assert steps == list(range(len(steps))) and len(steps) < 200
    (running,) = send(session, type="inspect_step", step=steps[-1])  # its last step on the road
    (after,) = send(session, type="inspect_step", step=steps[-1] + 1)  # the crash is recorded
    assert running.status == 0 and after.status == 1


def test_moments_work_for_drivers_without_a_network():
    session = loaded("constant", n_cars=2)
    frames = [send(session, type="step")[0] for _ in range(3)]
    (moment,) = send(session, type="inspect_step", step=1)
    assert moment.trace is None and moment.explain is None
    assert (moment.x, moment.y) == (frames[0].cars.x[0], frames[0].cars.y[0])
