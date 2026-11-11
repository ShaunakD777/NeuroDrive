"""Guards for Study S5b (docs/studies/S5b.md): the worked examples of the Brief, that the
labelling wrapper changes no drive, that shift (0, 0) is the exam, and the hypothesis rules on
hand-made records (sets that must pass and sets that must fail)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.env.episode import run_episode
from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.policies import MLPPolicy
from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig
from neurodrive.experiment.exam import exam_track, shifted_starts
from neurodrive.experiment.metrics import episode_metrics

ROOT = Path(__file__).resolve().parents[2]
LOOP = Track.from_json(ROOT / "tracks" / "train_loop_01.json")


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_s5b", ROOT / "scripts" / "study_s5b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_policy(seed: int):
    network = init_network(NetworkSpec(), make_rng(seed))
    return lambda world: MLPPolicy(network)


# ── the Brief's worked examples ─────────────────────────────────────────────────────────────


def test_on_policy_error_worked_example(study):
    actions = [[0.0, 0.5], [0.1, 0.9], [0.0, 1.0]]
    labels = [[0.0, 0.4], [0.1, 0.3], [-0.2, -0.4]]
    assert study.on_policy_error(actions, labels) == pytest.approx(0.395)


def test_familiar_and_novel_errors_worked_example(study):
    novelty = np.array([0.01, 0.02, 0.5])  # only the third step is beyond tau = 0.1
    split = study.split_errors(novelty, np.array([0, 0, 0.04]), np.array([0.01, 0.36, 1.96]), 0.1)
    assert split["familiar"]["n"] == 2 and split["familiar"]["error"] == pytest.approx(0.0925)
    assert split["novel"]["n"] == 1 and split["novel"]["error"] == pytest.approx(1.0)
    assert split["novel"]["throttle"] == pytest.approx(1.96)


def test_early_and_late_rules(study):
    rules = study.early_and_late([1, 2, 3, 4], [0.10, 0.06, 0.04, 0.02], [0.0, 1.0, 1.0, 0.0])
    assert rules == {"early": 2, "late": 4}
    with pytest.raises(ValueError):
        study.early_and_late([1, 2], [0.1], [1.0, 1.0])


def test_the_grid_has_30_shifts_and_the_exam_among_them(study):
    grid = study.shifts()
    assert len(grid) == 30 and len(set(grid)) == 30
    assert grid[study.EXAM_SHIFT] == (0.0, 0.0)
    assert grid[0] == (-4.5, -30.0) and grid[-1] == (3.0, 30.0)  # sideways outer, heading inner


# ── the drives ──────────────────────────────────────────────────────────────────────────────


def test_the_labelling_wrapper_changes_no_drive(study):
    rec, labels = study.drive_shift(LOOP, random_policy(3), -3.0, 15.0)
    world = World(LOOP, WorldConfig.for_track(LOOP, 2))
    s0, lat, psi = shifted_starts(LOOP.length, -3.0, np.radians(15.0))
    world.reset(s0=s0, lateral=lat, heading_offset=psi)
    first_label = ScriptedExpert(world).label()
    plain = run_episode(world, random_policy(3)(world))
    for name in ("x", "y", "psi", "v", "observation", "action", "status"):
        assert np.array_equal(getattr(rec, name), getattr(plain, name)), name
    assert labels.shape == (rec.n_steps, 10, 2)
    assert np.array_equal(labels[0], first_label)  # the expert's label of the start state


def test_shift_zero_is_the_exam(study):
    part, _ = exam_track(LOOP, random_policy(5))
    rec, _ = study.drive_shift(LOOP, random_policy(5), 0.0, 0.0)
    metrics = episode_metrics(rec, LOOP.length)
    assert [m.status for m in metrics] == [e["status"] for e in part.episodes]
    assert [m.progress_m for m in metrics] == [e["progress_m"] for e in part.episodes]


def test_a_driver_record_counts_only_recoverable_shifted_starts(study):
    n = 30 * 10
    status = np.array(["finished"] * n, dtype="U10")
    status[:20] = "crashed"  # shift 0's starts: 20 crashes, 10 of them recoverable below
    recoverable = np.ones(n, dtype=bool)
    recoverable[10:20] = False
    first = study.EXAM_SHIFT * 10
    status[first : first + 3] = "stalled"  # the exam's own starts are not "shifted"
    steps = np.full(n, 90)
    d = {
        "status": status,
        "steps": steps,
        "end_s_m": np.zeros(n),
        "end_speed": np.zeros(n),
        "lead_novelty": np.where(status == "crashed", 0.5, np.nan),
        "sum_sq_steer": np.full(n, 0.9),
        "sum_sq_throttle": np.full(n, 0.9),
        "n_steps_alive": steps,
        "sample_start": np.repeat(np.arange(n), 2),
        "sample_t": np.tile([0, 3], n),
        "sample_novelty": np.tile([0.01, 0.5], n),
        "sample_sq_steer": np.tile([0.0, 0.2], n),
        "sample_sq_throttle": np.tile([0.0, 0.2], n),
        "drive_s": np.array(1.0),
    }
    driver = {"id": "early-s0", "kind": "early", "seed": 0, "run": "r", "unit": 2}
    r = study.driver_record(driver, d, recoverable, tau=0.1)
    shifted = n - 10 - 10  # minus the exam's own starts and the 10 unrecoverable ones
    assert r["shifted_recoverable_starts"] == shifted
    assert r["completion_shifted"] == pytest.approx((shifted - 10) / shifted)
    assert r["completion_exam_starts"] == pytest.approx(0.7)
    assert r["n_crashes"] == 10 and r["share_crashes_from_novel"] == 1.0
    assert r["on_policy"]["error"] == pytest.approx(0.01)  # (0.9 + 0.9) / (2 * 90)
    assert r["split"]["familiar"]["error"] == 0.0
    assert r["split"]["novel"]["error"] == pytest.approx(0.2)


# ── the hypothesis rules ────────────────────────────────────────────────────────────────────


def records(shifted, crash_share, fam, nov, late_fam, late_nov, n=500) -> dict:
    """Hand-made records for seeds 0-9 with only what hypotheses() reads."""
    out = {}
    for s in range(10):
        for kind, f, v in (("early", fam, nov), ("late", late_fam, late_nov)):
            out[f"{kind}-s{s}"] = {
                "seed": s,
                "completion_shifted": shifted[s],
                "completion_exam_starts": 1.0,
                "share_crashes_from_novel": crash_share[s],
                "split": {"familiar": {"n": n, "error": f[s]}, "novel": {"n": n, "error": v[s]}},
            }
    return out


def test_the_hypotheses_pass_on_a_set_made_to_pass(study):
    shifted = [0.5 + 0.03 * s for s in range(10)]
    share = [0.8, 0.9, None, 0.7, 1.0, 0.6, 0.9, 0.8, 0.75, 0.85]  # one seed without a crash
    fam = [0.01 + 0.001 * s for s in range(10)]
    nov = [0.1 + 0.01 * s for s in range(10)]
    h = study.hypotheses(
        records(shifted, share, fam, nov, [f / 2 for f in fam], [v * 3 for v in nov]), 0.1
    )
    assert h["H1"]["supported"] and h["H1"]["difference"]["ci"][1] < 0
    assert h["H2"]["supported"] and h["H2"]["early"]["seeds_with_crashes"] == 9
    assert h["H3"]["supported"] and len(h["H3"]["seeds_compared"]) == 10
    assert h["H4"]["supported"]


def test_the_hypotheses_fail_on_sets_made_to_fail(study):
    fam = [0.01] * 10
    nov = [0.1] * 10
    # H1: the clones finish every shifted start; H2: crashes mostly from familiar states.
    h = study.hypotheses(records([1.0] * 10, [0.3] * 10, fam, nov, fam, nov), 0.1)
    assert not h["H1"]["supported"] and not h["H2"]["supported"]
    # H3: one seed's novel error is not larger; H4: the late networks are no better on the data.
    nov_one_low = [0.1] * 9 + [0.005]
    h = study.hypotheses(records([0.5] * 10, [0.8] * 10, fam, nov_one_low, fam, nov), 0.1)
    assert not h["H3"]["supported"] and not h["H4"]["supported"]
    # Too few measured steps of a kind: nothing to compare, so not supported.
    h = study.hypotheses(records([0.5] * 10, [0.8] * 10, fam, nov, fam, nov, n=50), 0.1)
    assert h["H3"]["seeds_compared"] == [] and not h["H3"]["supported"]
    assert not h["H4"]["supported"]
