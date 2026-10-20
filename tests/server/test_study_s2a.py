"""Guards for Study S2a (docs/studies/S2a.md): its cars are the app's cars, and its cases hold.

The study builds "Random networks, Random start, seed s" itself so it can run 2,000 brains
quickly. These tests check that it builds exactly what the live session builds, so every case in
the report can be replayed in the app, and that the cases the report explains still behave as
described.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from neurodrive.env.policies import PopulationPolicy
from neurodrive.env.sensors import edge_grid, observe
from neurodrive.server import live
from neurodrive.server.live import LiveSession, TrackLibrary

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_s2a", ROOT / "scripts" / "study_s2a.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def track(study):
    return study.load_track()


def test_the_study_uses_the_live_sessions_random_start_ranges(study):
    assert study.RANDOM_START_LATERAL == live.RANDOM_START_LATERAL
    assert study.RANDOM_START_HEADING == live.RANDOM_START_HEADING


def test_the_studys_cars_are_the_apps_cars_step_for_step(study, track):
    session = LiveSession(TrackLibrary(ROOT / "tracks"))
    command = {"type": "load_scenario", "track": study.TRACK, "policy": "random_networks"}
    session.handle_text(json.dumps({**command, "n_cars": study.N_CARS, "seed": 1}))
    world, population = study.scenario(track, 1)
    policy = PopulationPolicy(population)
    segments = edge_grid(track)
    for _ in range(40):
        action = np.clip(policy.act(observe(world, segments)), -1.0, 1.0)
        world.step(action[:, 0], action[:, 1])
        (frame,) = session.handle_text('{"type": "step"}')
        assert frame.cars.x == np.round(world.cars.x, 3).tolist()
        assert frame.cars.y == np.round(world.cars.y, 3).tolist()
        assert frame.cars.action == np.round(action, 3).tolist()


def test_the_cases_the_report_explains_still_behave_as_described(study, track):
    rec, population, dt = study.drive(track, 1)
    standstill = study.describe_car(rec, population, dt, 1, 0)
    assert standstill["stall_kind"] == "never_moved" and standstill["first_throttle"] < 0
    crash = study.describe_car(rec, population, dt, 1, 11)
    assert crash["status"] == "crashed" and crash["seconds"] == pytest.approx(10.37)
    assert crash["wall"] == "left" and crash["steering_towards_wall"]

    rec, population, dt = study.drive(track, 5)
    stopper = study.describe_car(rec, population, dt, 5, 17)
    assert stopper["stall_kind"] == "stopped_itself" and stopper["throttle_when_stopping"] < 0

    rec, population, dt = study.drive(track, 10)
    finisher = study.describe_car(rec, population, dt, 10, 10)
    assert finisher["status"] == "finished" and finisher["seconds"] == pytest.approx(83.03)
    assert finisher["right_signs"] == 4


def test_a_full_lock_circle_is_wider_than_the_space_beside_a_centred_car(study, track):
    circle = study.full_lock_circle()
    assert circle["diameter_measured"] == pytest.approx(circle["diameter_formula"], abs=0.01)
    # a car in the middle of the road has half the width to the edge, minus its own half-width
    assert circle["diameter_measured"] > track.width / 2
