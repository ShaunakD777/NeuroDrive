"""Tests for live sessions: the clock logic with a hand-driven clock, then the real WebSocket."""

import json
import math
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neurodrive.server.app import create_app
from neurodrive.server.live import LiveSession, TrackLibrary
from neurodrive.server.schemas import (
    API_VERSION,
    EpisodeEndMessage,
    ErrorMessage,
    FrameMessage,
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
    assert scenario.time_limit == 90.0 and scenario.v_max == 30.0
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
    assert listing == {"tracks": [{"name": TRACK, "split": "train"}]}
    geometry = client.get(f"/api/tracks/{TRACK}").json()
    assert set(geometry) == {
        "name", "split", "width", "length", "start_s0", "centerline", "left", "right",
    }  # fmt: skip
    assert len(geometry["left"]) == len(geometry["centerline"])
    assert client.get("/api/tracks/nope").status_code == 404


def test_websocket_session_end_to_end():
    client = TestClient(create_app())
    with client.websocket_connect("/ws/live") as ws:
        assert json.loads(ws.receive_text())["api"] == API_VERSION == "v1.3"
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
