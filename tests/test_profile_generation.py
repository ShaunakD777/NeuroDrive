"""Guards for the task 7.0 profiling script (scripts/profile_generation.py): its worst case
really keeps every car driving, it measures a generation's work without changing the driving,
and its counters and timings add up."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neurodrive.env.experts import ScriptedExpert

ROOT = Path(__file__).resolve().parents[1]
CARS, SECONDS, SEED = 6, 4.0, 3  # 120 steps: fast, and the expert keeps every car on the road


@pytest.fixture(scope="module")
def prof():
    spec = importlib.util.spec_from_file_location(
        "profile_generation", ROOT / "scripts" / "profile_generation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def track(prof):
    return prof.load_track("train_loop_01")


@pytest.fixture(scope="module")
def worst(prof, track):
    actions = prof.expert_actions(track, CARS, SECONDS, SEED)
    return actions, prof.run_generation(track, CARS, SECONDS, SEED, actions)


def test_worst_case_keeps_every_car_running_to_the_time_limit(worst):
    actions, g = worst
    steps = round(SECONDS * 30)
    assert actions.shape == (steps, CARS, 2)
    assert g.steps == steps
    assert g.status == {"timeout": CARS}  # nobody crashed, stalled or finished early
    assert g.env_steps == g.sensor_rows == CARS * steps


def test_worst_case_drives_exactly_as_the_expert_does(prof, track, worst):
    # The timed pass computes sensors and networks but steers with the recorded actions, so the
    # cars end where the expert's own drive ends, bit for bit.
    world = prof.make_world(track, CARS, SECONDS, SEED)
    expert = ScriptedExpert(world)
    while not world.done:
        a = expert.label()
        world.step(a[:, 0], a[:, 1])
    cars = world.cars
    expected = np.stack([cars.x, cars.y, cars.psi, cars.v], axis=1)
    assert worst[1].final.tobytes() == expected.tobytes()


def test_the_starts_are_seeded_and_spread_over_the_track(prof, track):
    a, b = prof.make_world(track, CARS, SECONDS, SEED), prof.make_world(track, CARS, SECONDS, SEED)
    assert a.start["s0"].tobytes() == b.start["s0"].tobytes()
    other = prof.make_world(track, CARS, SECONDS, SEED + 1)
    assert not np.array_equal(a.start["s0"], other.start["s0"])
    assert np.all(np.abs(a.start["lateral"]) <= prof.START_LATERAL)
    assert np.all(np.abs(a.start["heading_offset"]) <= prof.START_HEADING)


def test_phase_times_add_up_to_the_total(worst, prof):
    g = worst[1]
    assert set(g.phase_s) == set(prof.PHASES)
    assert all(t >= 0 for t in g.phase_s.values())
    assert sum(g.phase_s.values()) == pytest.approx(g.total_s, rel=1e-9)
    assert g.ms_per_step == pytest.approx(1000 * g.total_s / g.steps)


def test_networks_steering_stop_early_and_only_running_cars_get_sensors(prof, track):
    # Fresh random networks crash or stall: fewer car-steps than the worst case, and the sensors
    # are computed for running cars only, exactly as random_search.evaluate does.
    g = prof.run_generation(track, CARS, SECONDS, SEED)
    assert g.env_steps < CARS * g.steps
    assert g.sensor_rows == g.env_steps
    assert sum(g.status.values()) == CARS


def test_the_random_search_unit_is_the_real_code(prof, track):
    unit = prof.time_random_search_unit(track, 2, SEED)
    assert unit["cars"] == 2 * unit["starts_per_network"] == 20
    assert unit["env_steps"] > 0 and unit["total_s"] > 0


def test_function_profile_names_files_without_their_folders(prof, track):
    rows = prof.function_profile(track, 2, 1.0, SEED, top=5)
    assert len(rows) == 5
    assert all("/" not in r["function"] and "\\" not in r["function"] for r in rows)
