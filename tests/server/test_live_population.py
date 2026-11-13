"""Tests for racing a GA generation in the live session (task 7.6, ADR-057, api v1.22).

The central check: generation g raced from evaluation start k drives exactly the drives the GA
scored, car for car and to the last digit: every car ends with the status, progress, finish time
and per-start fitness that ``evaluate_genomes`` gives for that network at that start. The ghost
(an earlier generation's champion) drives exactly that champion's scored drive too.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.nn.network import Network
from neurodrive.env.track import Track
from neurodrive.env.world import Status
from neurodrive.experiment.config import parse_config, to_ga_settings
from neurodrive.experiment.runner import train_ga_run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.evolution.ga import GASettings, evaluate_genomes, ranking
from neurodrive.server.live import LiveSession, TrackLibrary
from neurodrive.server.runs import RunLibrary

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ROOT / "tracks"
TRAIN = Track.from_json(TRACKS / "train_loop_01.json")
N, STARTS = 6, 2


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    """A runs folder with a small GA run (6 networks, 1 elite, 2 starts, 1 lap, 3 generations),
    a copy of it without saved generations (as a run trained before task 7.6) and a run of
    another method."""
    folder = tmp_path_factory.mktemp("runs")
    config = parse_config(
        {
            "method": "ga",
            "seed": 2,
            "tag": "race",
            "population": {"size": N, "elite": 1},
            "variation": {"mutation_rate": 0.2, "mutation_strength": 0.3},
            "fitness": {"start_fractions": [0.0, 0.5], "lateral_offsets": [0.0], "target_laps": 1},
            "budget": {"units": 3},
            "exams": False,
        }
    )
    store = train_ga_run(config, folder).store
    old = folder / store.path.name.replace("-race", "-old")
    shutil.copytree(store.path, old)
    shutil.rmtree(old / "extra" / "generations")
    RunStore.create(folder, "random_search", "other", {"method": "random_search"}).finish(
        "completed", "units"
    )
    return folder


def run_id(runs: Path, tag: str) -> str:
    (path,) = [p for p in runs.iterdir() if p.name.endswith(f"-{tag}")]
    return path.name


def settings_of(runs: Path) -> GASettings:
    return to_ga_settings(parse_config(RunStore.open(runs / run_id(runs, "race")).read_config()))


def generation(runs: Path, unit: int) -> np.ndarray:
    store = RunStore.open(runs / run_id(runs, "race"))
    return store.load_unit_arrays("generations", unit)["genomes"]


def line(runs: Path, unit: int) -> dict:
    store = RunStore.open(runs / run_id(runs, "race"))
    return {r["unit"]: r for r in store.read_extra_lines("population")}[unit]


def session_for(runs: Path) -> LiveSession:
    return LiveSession(TrackLibrary(TRACKS), RunLibrary(runs))


def send(session: LiveSession, **command) -> list:
    return session.handle_text(json.dumps(command))


def race(session: LiveSession, run: str, unit: int, start: int = 0, ghost: int | None = None):
    population = {"run": run, "unit": unit, "start": start, "ghost_unit": ghost}
    # the track asked for is replaced by the run's fitness track
    return send(
        session,
        type="load_scenario",
        track="hairpins_01",
        policy="population",
        population=population,
    )


def play_to_the_end(session: LiveSession):
    replies = []
    while not session._world.done:
        session._advance(500)
        replies = session._frame_and_end()
    (end,) = [m for m in replies if m.type == "episode_end"]
    return end


# ── the race is the GA's evaluation ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("unit", [1, 2, 3])
@pytest.mark.parametrize("start", [0, 1])
def test_a_raced_generation_drives_exactly_the_drives_the_ga_scored(runs, unit, start):
    settings = settings_of(runs)
    run = run_id(runs, "race")
    ghost = None if unit == 1 else 1  # generation 1's champion races generations 2 and 3
    session = session_for(runs)
    replies = race(session, run, unit, start, ghost)
    assert [m.type for m in replies] == ["scenario", "network", "frame"]
    scored = evaluate_genomes(generation(runs, unit), TRAIN, settings).evaluation
    end = play_to_the_end(session)
    world = session._world
    cars = range(N + (ghost is not None))
    expected = [(scored, i) for i in range(N)]
    if ghost is not None:
        champion = line(runs, ghost)["champion"]
        expected.append(
            (evaluate_genomes(generation(runs, ghost), TRAIN, settings).evaluation, champion)
        )
    for car, (ev, row) in zip(cars, expected, strict=True):
        assert int(world.status[car]) == int(ev.status[row, start])
        progress = np.clip(world.progress[car] / TRAIN.length, 0.0, 1.0)
        assert progress == ev.progress_frac[row, start]  # to the last digit
        if world.status[car] == Status.FINISHED:
            assert session._end_step[car] * world.dt == ev.finish_time[row, start]
        assert end.fitness[car] == round(float(ev.per_start[row, start]), 6)
    assert len(end.fitness) == len(expected) == world.n


def test_the_scenario_describes_the_generation(runs):
    run = run_id(runs, "race")
    session = session_for(runs)
    scenario = race(session, run, 3, start=1, ghost=2)[0]
    settings, saved = settings_of(runs), line(runs, 3)
    s0, lateral = settings.evaluation().starts(TRAIN.length)
    assert scenario.policy == "population" and scenario.track == "train_loop_01"
    assert scenario.n_cars == N + 1 and scenario.seed is None and scenario.target_laps == 1
    info = scenario.population
    assert (info.run, info.unit, info.last_unit, info.start, info.n_starts) == (run, 3, 3, 1, 2)
    assert info.start_s0_m == round(float(s0[1]), 3) and info.start_lateral_m == lateral[1]
    assert info.size == N and info.fitness == [float(f"{v:.6g}") for v in saved["fitness"]]
    assert info.ranks == ranking(saved["fitness"])[1].tolist()
    assert info.elite == [0] and info.champion == saved["champion"]
    assert (info.finished, info.crashed) == (saved["finished"], saved["crashed"])
    ghost_line = line(runs, 2)
    assert (info.ghost_unit, info.ghost_car) == (2, N)
    assert info.ghost_fitness == float(f"{ghost_line['fitness'][ghost_line['champion']]:.6g}")
    world = session._world
    assert np.allclose(world.s, s0[1]) and np.allclose(world.lateral_offset, lateral[1])
    first = race(session, run, 1)[0].population
    assert first.elite == [] and first.ghost_unit is None and first.ghost_car is None
    assert session._world.n == N


def test_the_colours_rank_exactly_as_the_ga_ranked(runs):
    # equal fitness keeps the lower row first (task 7.2's stable ranking), in every generation
    session, tied = session_for(runs), 0
    for unit in (1, 2, 3):
        info = race(session, run_id(runs, "race"), unit)[0].population
        fitness = line(runs, unit)["fitness"]
        assert info.ranks == ranking(fitness)[1].tolist()
        for a in range(N):
            for b in range(a + 1, N):
                if fitness[a] == fitness[b]:
                    tied += 1
                    assert info.ranks[a] < info.ranks[b]
    assert tied > 0  # the run has ties (networks at exactly 0), so the rule was really checked


def test_the_ghost_of_the_previous_generation_is_the_elite(runs):
    # with E = 1, row 0 of generation g is generation g - 1's champion: the two drive as one
    session = session_for(runs)
    race(session, run_id(runs, "race"), 3, start=0, ghost=2)
    world = session._world
    moved = False
    while not world.done:
        session._advance(1)
        cars = world.cars
        assert (cars.x[0], cars.y[0], cars.psi[0]) == (cars.x[N], cars.y[N], cars.psi[N])
        moved = moved or cars.v[0] > 0
    assert moved and world.status[0] == world.status[N]


def test_selecting_a_car_shows_its_own_network(runs):
    session = session_for(runs)
    race(session, run_id(runs, "race"), 2, ghost=1)
    genomes, spec = generation(runs, 2), settings_of(runs).spec
    for car in (0, 4):
        network = send(session, type="select_car", index=car)[0]
        assert network.type == "network" and network.car == car
        want = Network.unflatten(spec, genomes[car])
        for sent, w in zip(network.weights, want.weights, strict=True):
            assert np.array_equal(sent, np.round(w, 4))
    ghost = send(session, type="select_car", index=N)[0]
    champion = generation(runs, 1)[line(runs, 1)["champion"]]
    assert np.array_equal(
        ghost.biases[-1], np.round(Network.unflatten(spec, champion).biases[-1], 4)
    )
    frame = send(session, type="step")[0]
    assert frame.selected.trace is not None and frame.selected.explain is not None


def test_reset_races_the_same_generation_again(runs):
    session = session_for(runs)
    race(session, run_id(runs, "race"), 2, start=1)
    first = play_to_the_end(session)
    send(session, type="reset")
    assert session._world.step_count == 0 and (session._end_step == -1).all()
    assert play_to_the_end(session).fitness == first.fitness


@pytest.mark.parametrize(
    "tag, unit, start, ghost, message",
    [
        ("race", 4, 0, None, "has no saved generation 4; saved: 1 to 3"),
        ("race", 2, 2, None, "start 2 does not exist: the run has starts 0 to 1"),
        ("race", 2, 0, 2, "the ghost must come from an earlier generation than 2, got 2"),
        ("old", 1, 0, None, "has no saved generations (trained before task 7.6)"),
        ("other", 1, 0, None, "is a random_search run; population mode needs a GA run"),
    ],
)
def test_a_generation_that_cannot_race_is_refused_in_plain_words(
    runs, tag, unit, start, ghost, message
):
    session = session_for(runs)
    send(session, type="load_scenario", track="train_loop_01", policy="expert")
    (error,) = race(session, run_id(runs, tag), unit, start, ghost)
    assert error.type == "error" and message in error.message
    assert session._scenario.policy == "expert"  # the old scenario is kept


def test_other_refusals(runs):
    session = session_for(runs)
    (error,) = send(session, type="load_scenario", track="train_loop_01", policy="population")
    assert "needs population {run, unit, start}" in error.message
    (error,) = race(session, "20990101-000000-ga-nothing", 1)
    assert "unknown run" in error.message
    population = {"run": run_id(runs, "race"), "unit": 1}
    (error,) = send(
        session, type="load_scenario", track="train_loop_01", policy="expert", population=population
    )
    assert "population works only with the population driver" in error.message
    (error,) = LiveSession(TrackLibrary(TRACKS)).handle_text(
        json.dumps(
            {
                "type": "load_scenario",
                "track": "x",
                "policy": "population",
                "population": population,
            }
        )
    )
    assert "no run folders" in error.message


def test_other_drivers_send_no_race_fitness(runs):
    session = session_for(runs)
    send(session, type="load_scenario", track="train_loop_01", policy="constant", n_cars=2)
    end = play_to_the_end(session)
    assert end.fitness is None
