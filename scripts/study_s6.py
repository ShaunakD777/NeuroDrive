"""Study S6: the genetic algorithm against random search, its sweeps, its fitness and its line.

Run from the repository root:

    uv run python scripts/study_s6.py --train --only-train --workers 11   # make the runs (~4 h)
    uv run python scripts/study_s6.py                                     # analyse them (numbers)
    uv run --with matplotlib python scripts/study_s6.py --figures         # also draw the figures

Design and hypotheses: docs/studies/S6.md (written in the task 7.9 Brief, before any run).

Ten arms, every run with ``"study": "S6"``, each arm exactly one change from its shipped config
(``configs/ga.yaml``, or ``configs/random_search.yaml`` for random search), and every run the same
budget in environment steps, B = 40 M (ADR-062):

* Part A: ``ga`` (the defaults: N 50, E 2, k 3, c 0.5, p 0.02, sigma 0.05) and
  ``random_search``, seeds 0-9, examined at every checkpoint (a learning curve over steps);
* Part B: ``n20``, ``n100`` (population), ``p005``, ``p01`` (mutation rate, sigma fixed),
  ``s002``, ``s02`` (mutation strength, p fixed), seeds 0-4;
* Part C: ``finish_bonus`` (f = p + c) and ``progress`` (f = p), seeds 0-4.

Part B and C runs are trained without exams; afterwards only their chosen checkpoint is examined.
The chosen checkpoint of every run is ``best_unit``, the unit with the highest training fitness
(a training-only rule: it reads metrics.jsonl and nothing else). From its exam card: T, the
training loop's completion rate; L, its median best lap over finished drives; P, the held-out
median progress; C, the held-out completion rate.

Part E drives every chosen checkpoint (and the expert) from the exam's 10 training-loop starts
again and measures the racing line (``experiment/racing_line.py``): per run the median inside
offset at the apex e_in, the outside offsets at entry and exit, over every bend pass of its
finished drives.

Paired differences are medians over seeds of per-seed differences with bootstrap 95 % intervals
(10 000 resamples, seed 2026). Every Part B and C comparison pairs with the default GA's seeds
0-4. Numbers go to docs/studies/S6_results.json (the same bytes on every analysis of the same
runs); with --figures the pictures go to docs/studies/figures/.
"""

import os

# One thread per process: the runs go side by side, one per core, so a BLAS library spreading a
# single process over every core would only make them fight. Set before NumPy is imported.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sqlite3  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from contextlib import closing  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from neurodrive.core.rng import make_rng  # noqa: E402
from neurodrive.env.experts import ScriptedExpert  # noqa: E402
from neurodrive.env.policies import MLPPolicy  # noqa: E402
from neurodrive.env.track import Track  # noqa: E402
from neurodrive.experiment.config import load_config, parse_config, resolved  # noqa: E402
from neurodrive.experiment.db import DB_NAME, Mirror, connect, run_folders, runs_of_study  # noqa: E402
from neurodrive.experiment.exam import (  # noqa: E402
    DEFAULT_TRACKS_DIR,
    EXAM_V1,
    ExamSpec,
    exam_track,
    run_exam,
)
from neurodrive.experiment.racing_line import (  # noqa: E402
    bend_offsets,
    bends,
    drive_offsets,
    racing_line_summary,
)
from neurodrive.experiment.runner import (  # noqa: E402
    learner_from_config,
    resume,
    train_ga_run,
    train_random_search_run,
)
from neurodrive.experiment.stats import bootstrap_median_ci, summarise  # noqa: E402
from neurodrive.experiment.store import RunStore  # noqa: E402
from neurodrive.learn.supervised.dagger import ExpertLabelled  # noqa: E402
from neurodrive.learn.supervised.demos import driven_steps  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "docs" / "studies" / "S6_results.json"
FIGURES_DIR = ROOT / "docs" / "studies" / "figures"
GA_YAML = ROOT / "configs" / "ga.yaml"
RS_YAML = ROOT / "configs" / "random_search.yaml"

STUDY = "S6"
BUDGET_STEPS = 40_000_000  # B, environment steps per run (D2A)
SEEDS_A = tuple(range(10))
SEEDS_B = tuple(range(5))
BOOTSTRAP_SEED = 2026
MIN_PAIRS = 3  # a paired rule with fewer seeds is reported as not testable
REFERENCE_RUN = "20261113-123056-ga-race76"  # task 7.5's seed 0, 30 generations
REFERENCE_UNITS = 30
LIMIT_MARGIN_S = 2.0  # "finishes within 2 s of the limit"

# name -> (base config, one change (section, key, value) or None, seeds, tag, exams)
ARMS: dict[str, dict[str, Any]] = {
    "ga": dict(base=GA_YAML, change=None, seeds=SEEDS_A, tag="s6-ga", exams=True),
    "random_search": dict(base=RS_YAML, change=None, seeds=SEEDS_A, tag="s6-rs", exams=True),
    "n20": dict(base=GA_YAML, change=("population", "size", 20), seeds=SEEDS_B, tag="s6-n20"),
    "n100": dict(base=GA_YAML, change=("population", "size", 100), seeds=SEEDS_B, tag="s6-n100"),
    "p005": dict(
        base=GA_YAML, change=("variation", "mutation_rate", 0.005), seeds=SEEDS_B, tag="s6-p005"
    ),
    "p01": dict(
        base=GA_YAML, change=("variation", "mutation_rate", 0.1), seeds=SEEDS_B, tag="s6-p01"
    ),
    "s002": dict(
        base=GA_YAML, change=("variation", "mutation_strength", 0.02), seeds=SEEDS_B, tag="s6-s002"
    ),
    "s02": dict(
        base=GA_YAML, change=("variation", "mutation_strength", 0.2), seeds=SEEDS_B, tag="s6-s02"
    ),
    "finish_bonus": dict(
        base=GA_YAML, change=("fitness", "formula", "finish_bonus"), seeds=SEEDS_B, tag="s6-finish"
    ),
    "progress": dict(
        base=GA_YAML, change=("fitness", "formula", "progress"), seeds=SEEDS_B, tag="s6-progress"
    ),
}
# the longest jobs first (exams at every checkpoint, then the biggest population), so the short
# ones fill the gaps at the end
TRAIN_ORDER = (
    "random_search",
    "ga",
    "n100",
    "p01",
    "s02",
    "finish_bonus",
    "progress",
    "p005",
    "s002",
    "n20",
)
SWEEP_ARMS = ("n20", "n100", "p005", "p01", "s002", "s02")
FITNESS_ARMS = ("finish_bonus", "progress")
# the training loop's 10 exam starts moved halfway to the next start: positions the fitness never
# scored (an exploration for exploits, not a hypothesis)
UNSEEN_STARTS = ExamSpec(
    version=EXAM_V1.version,
    tracks=EXAM_V1.tracks[:1],
    start_fractions=(0.1, 0.3, 0.5, 0.7, 0.9),
    lateral_offsets=EXAM_V1.lateral_offsets,
    target_laps=EXAM_V1.target_laps,
)


# ── training ────────────────────────────────────────────────────────────────────────────────


def study_config(arm: str, seed: int):
    """The config of ``arm`` for ``seed``: the shipped YAML with the arm's one change, the
    budget B in environment steps (no unit or clock limit), its exams setting, and the tag
    ``<arm tag>-s<seed>``."""
    spec = ARMS[arm]
    data = resolved(load_config(spec["base"]))
    data["budget"] = {"units": None, "env_steps": BUDGET_STEPS, "wall_clock_s": None}
    data["exams"] = bool(spec.get("exams", False))
    if spec["change"] is not None:
        section, key, value = spec["change"]
        data[section][key] = value
    return parse_config(data, seed=seed, tag=f"{spec['tag']}-s{seed}")


def choose_checkpoint(metrics: list[dict[str, Any]]) -> int:
    """The chosen checkpoint: the unit with the highest training fitness so far at the end of the
    run (``best_unit`` of the last metrics line). It reads the training metrics only."""
    if not metrics:
        raise ValueError("the run has no completed unit")
    return int(metrics[-1]["best_unit"])


def ensure_exam(path: Path, tracks_dir: Path = DEFAULT_TRACKS_DIR) -> bool:
    """Examine the chosen checkpoint of the run at ``path`` if it has no exam card yet, and copy
    the run into the database mirror again. Returns whether an exam was taken."""
    store = RunStore.open(path)
    unit = choose_checkpoint(store.read_metrics())
    if unit in store.exam_units():
        return False
    network = store.load_checkpoint(unit)
    ref = {"kind": "checkpoint", "run": store.path.name, "unit": unit}
    store.save_exam(unit, run_exam(lambda world: MLPPolicy(network), ref, tracks_dir))
    mirror = Mirror(store.path.parent / DB_NAME)
    mirror.sync_run(store)
    mirror.close()
    return True


def study_runs(runs_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """Every S6 run folder in ``runs_dir`` by tag: [(status, path), ...]."""
    out: dict[str, list[tuple[str, Path]]] = {}
    for path in run_folders(runs_dir):
        meta = RunStore.open(path).read_meta()
        if meta.get("study") == STUDY:
            out.setdefault(meta["tag"], []).append((meta["status"], path))
    return out


def _train_job(job: tuple[str, int, str, str]) -> str:
    """Train (arm, seed), or resume its killed run, then examine its chosen checkpoint."""
    arm, seed, sweep_id, runs_dir = job
    tag = f"{ARMS[arm]['tag']}-s{seed}"
    unfinished = [p for s, p in study_runs(Path(runs_dir)).get(tag, []) if s != "completed"]
    if unfinished:
        store = RunStore.open(unfinished[0])
        resume(learner_from_config(store.read_config()), store)
    else:
        config = study_config(arm, seed)
        train = train_random_search_run if arm == "random_search" else train_ga_run
        store = train(config, runs_dir, extra_meta={"sweep": sweep_id, "study": STUDY}).store
    ensure_exam(store.path)
    metrics = store.read_metrics()
    print(
        f"{datetime.now():%H:%M:%S} {arm} seed {seed}: {store.path.name}, {len(metrics)} units, "
        f"{metrics[-1]['env_steps_total']:,} steps, {metrics[-1]['wall_clock_s'] / 60:.1f} min",
        flush=True,
    )
    return store.path.name


def train(runs_dir: Path, workers: int, arms: tuple[str, ...]) -> None:
    """Make the missing runs (and finish killed ones), several processes side by side. Every run
    depends only on its own config and seed, so running them together changes no number, only
    the wall-clock times. Completed runs only get their chosen checkpoint examined if missing."""
    found = study_runs(runs_dir)
    when = f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    jobs = []
    for arm in (a for a in TRAIN_ORDER if a in arms):
        spec = ARMS[arm]
        for seed in spec["seeds"]:
            tag = f"{spec['tag']}-s{seed}"
            done = [p for s, p in found.get(tag, []) if s == "completed"]
            if done:
                ensure_exam(done[0])
                continue
            jobs.append((arm, seed, f"{when}-{spec['tag']}", str(runs_dir)))
    print(f"{len(jobs)} runs to make with {workers} workers", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_train_job, jobs))


# ── reading the runs ────────────────────────────────────────────────────────────────────────


def arm_of(tag: str) -> str | None:
    for arm, spec in ARMS.items():
        if tag.startswith(f"{spec['tag']}-s") and tag[len(spec["tag"]) + 2 :].isdigit():
            return arm
    return None


def find_runs(runs_dir: Path) -> dict[str, Any]:
    """S6's run folders by their ``study`` field. Stops with a message when a run is missing,
    unfinished or repeated."""
    arms: dict[str, dict[int, list[Path]]] = {a: {} for a in ARMS}
    every = []
    for path in run_folders(runs_dir):
        store = RunStore.open(path)
        meta = store.read_meta()
        if meta.get("study") != STUDY:
            continue
        every.append(path)
        if meta["status"] != "completed":
            raise SystemExit(f"{path.name} is {meta['status']}; finish it (nd resume) first")
        arm = arm_of(meta["tag"])
        if arm is None:
            raise SystemExit(f"{path.name} has the S6 study tag but no S6 arm tag")
        arms[arm].setdefault(store.read_config()["seed"], []).append(path)
    for arm, by_seed in arms.items():
        want = set(ARMS[arm]["seeds"])
        if set(by_seed) != want or any(len(p) > 1 for p in by_seed.values()):
            raise SystemExit(f"S6 needs exactly one {arm} run for each of seeds {sorted(want)}")
    return {"arms": {a: {s: p[0] for s, p in by.items()} for a, by in arms.items()}, "all": every}


def read_card(store: RunStore, unit: int) -> dict[str, Any]:
    return json.loads((store.path / "exams" / f"unit_{unit}.json").read_text(encoding="utf-8"))


def card_measures(card: dict[str, Any]) -> dict[str, Any]:
    """T, L, P, C and the rest one exam card says about one network."""
    t, h = card["training"], card["heldout"]
    out: dict[str, Any] = {
        "T": t["completion_rate"],
        "L": t["median_best_lap_s"],
        "P": h["progress_frac_median"],
        "C": h["completion_rate"],
        "training_progress": t["progress_frac_median"],
        "training_crash": t["crash_rate"],
        "heldout_crash": h["crash_rate"],
        "training_smoothness": t["median_smoothness"],
        "heldout_smoothness": h["median_smoothness"],
    }
    for part in card["tracks"]:
        if part["split"] == "train":
            done = [e["time_s"] for e in part["episodes"] if e["completed"]]
            out["training_finish_times_s"] = done
            out["training_time_limit_s"] = part["time_limit_s"]
            out["training_speed"] = float(np.median([e["mean_speed"] for e in part["episodes"]]))
        else:
            out[f"P:{part['track']}"] = part["aggregate"]["progress_frac_median"]
            out[f"C:{part['track']}"] = part["aggregate"]["completion_rate"]
    return out


def run_record(path: Path, arm: str) -> dict[str, Any]:
    """What one run did: its chosen checkpoint and that checkpoint's exam measures, the fitness
    and diversity at the end, when a network first finished every start, the exam curve over
    environment steps (Part A), and the budgets in every currency."""
    store = RunStore.open(path)
    metrics = store.read_metrics()
    meta = store.read_meta()
    config = store.read_config()
    unit = choose_checkpoint(metrics)
    last = metrics[-1]
    steps_at = {0: 0, **{m["unit"]: m["env_steps_total"] for m in metrics}}
    per_unit = config["population"]["size"] if arm != "random_search" else None
    if arm == "random_search":
        networks = last["candidates_total"]
    else:
        networks = per_unit * len(metrics)
    finishers = [m["unit"] for m in metrics if m.get("finishers", 0) > 0]
    curve = {}
    if ARMS[arm].get("exams"):
        for u in store.exam_units():
            m = card_measures(read_card(store, u))
            curve[str(u)] = {"env_steps": steps_at[u], "T": m["T"], "P": m["P"], "C": m["C"]}
    t_full = [c["env_steps"] for c in curve.values() if c["T"] == 1.0]
    first_t1 = min(
        (c for c in curve.values() if c["T"] == 1.0), key=lambda c: c["env_steps"], default=None
    )
    return {
        "run_id": path.name,
        "seed": config["seed"],
        "unit": unit,
        **card_measures(read_card(store, unit)),
        "best_fitness": last["best_fitness"],
        "final_diversity": last.get("diversity"),
        "final_median_fitness": last.get("median_fitness", last.get("unit_median_fitness")),
        "first_unit_with_a_full_finisher": finishers[0] if finishers else None,
        "env_steps_at_first_full_finisher": steps_at[finishers[0]] if finishers else None,
        "env_steps_at_first_T1_exam": min(t_full) if t_full else None,
        "P_at_first_T1_exam": first_t1["P"] if first_t1 else None,
        "C_at_first_T1_exam": first_t1["C"] if first_t1 else None,
        "curve": curve,
        "budget": {
            "units": len(metrics),
            "env_steps": last["env_steps_total"],
            "env_steps_at_chosen_unit": steps_at[unit],
            "networks_scored": networks,
            "wall_clock_s": last["wall_clock_s"],
            "exam_wall_clock_s": meta["exam_wall_clock_s"],
        },
    }


# ── the racing line (Part E) and unseen starts ──────────────────────────────────────────────


def _drive_job(job: tuple[str, str, int | None]) -> dict[str, Any]:
    """Drive one network (or the expert, ``run_id`` "expert") from the training loop's 10 exam
    starts and from the 10 unseen starts; measure the racing line of the finished exam drives."""
    runs_dir, run_id, unit = job
    track = Track.from_json(DEFAULT_TRACKS_DIR / f"{EXAM_V1.tracks[0]}.json")
    if run_id == "expert":
        policy_for = ScriptedExpert
    else:
        network = RunStore.open(Path(runs_dir) / run_id).load_checkpoint(unit)

        def policy_for(world):
            return MLPPolicy(network)

    found = bends(track)
    part, rec = exam_track(track, policy_for)
    passes = []
    offsets = []  # e of every frame of every finished drive (post hoc: which side it keeps)
    finished = 0
    for car, episode in enumerate(part.episodes):
        if not episode["completed"]:
            continue
        finished += 1
        n = int(rec.alive[:, car].sum()) + 1
        d, e = drive_offsets(
            track, rec.x[:n, car], rec.y[:n, car], rec.progress[:n, car], rec.start_s0[car]
        )
        passes += bend_offsets(d, e, track.length, found)
        offsets.append(e)
    unseen, _ = exam_track(track, policy_for, UNSEEN_STARTS)
    return {
        "run_id": run_id,
        "finished_drives": finished,
        "median_offset_m": float(np.median(np.concatenate(offsets))) if offsets else None,
        "racing_line": racing_line_summary(passes, len(found)) if passes else None,
        "unseen_start_completion": unseen.aggregate["completion_rate"],
        "unseen_start_progress": unseen.aggregate["progress_frac_median"],
    }


def drive_all(
    runs_dir: Path, records: dict[str, dict[int, dict[str, Any]]], workers: int
) -> dict[str, Any]:
    """Part E for the expert and every chosen checkpoint (the same numbers in any order)."""
    jobs = [(str(runs_dir), "expert", None)] + [
        (str(runs_dir), r["run_id"], r["unit"])
        for arm in ARMS
        for _, r in sorted(records[arm].items())
    ]
    with ProcessPoolExecutor(workers) as pool:
        out = list(pool.map(_drive_job, jobs))
    return {o["run_id"]: o for o in out}


# ── statistics and the hypotheses ───────────────────────────────────────────────────────────


def band(values: list[float | None]) -> dict[str, Any]:
    kept = [v for v in values if v is not None]
    if not kept:
        return {"median": None, "q1": None, "q3": None, "n": 0}
    s = summarise(kept)
    return {"median": s.median, "q1": s.q1, "q3": s.q3, "n": s.n}


def paired_difference(a: list[float], b: list[float], seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """d_k = a_k - b_k per seed; the median of the d_k and its bootstrap 95 % interval
    (10 000 resamples of the seeds). If the interval holds 0 there is no clear difference.

    Example: a = (0.80, 0.60, 0.90), b = (0.50, 0.60, 0.40): d = (0.30, 0.00, 0.50), median 0.30,
    interval [0.00, 0.50]: with three seeds, no clear difference.
    """
    if len(a) != len(b):
        raise ValueError("paired samples need one value per seed on each side")
    d = [x - y for x, y in zip(a, b, strict=True)]
    ci = bootstrap_median_ci(d, make_rng(seed))
    return {
        "estimate": ci.estimate,
        "ci": [ci.low, ci.high],
        "n_seeds": len(d),
        "per_seed": d,
        "clear": ci.low > 0 or ci.high < 0,
    }


def paired_by_seed(
    a: dict[int, float | None], b: dict[int, float | None], direction: int = 1
) -> dict[str, Any]:
    """The paired difference a - b over the seeds both sides have a value for; ``supported``
    when its median has the expected sign (``direction`` +1: above 0, -1: below) and its interval
    excludes 0. Fewer than MIN_PAIRS seeds: not testable (supported False, ``testable`` False)."""
    seeds = sorted(s for s in a if s in b and a[s] is not None and b[s] is not None)
    if len(seeds) < MIN_PAIRS:
        return {"seeds": seeds, "n_seeds": len(seeds), "testable": False, "supported": False}
    out = paired_difference([a[s] for s in seeds], [b[s] for s in seeds])
    if direction > 0:
        out["supported"] = out["estimate"] > 0 and out["ci"][0] > 0
    else:
        out["supported"] = out["estimate"] < 0 and out["ci"][1] < 0
    return {"seeds": seeds, "testable": True, **out}


def values(records: dict[int, dict[str, Any]], key: str, seeds=None) -> dict[int, Any]:
    return {s: r[key] for s, r in records.items() if seeds is None or s in seeds}


def median_of(records: dict[int, dict[str, Any]], key: str, seeds=None) -> float | None:
    return band(list(values(records, key, seeds).values()))["median"]


def hypotheses(records: dict[str, dict[int, dict[str, Any]]], expert_e_in: float) -> dict[str, Any]:
    """The pre-registered rules G1-G2, N1, M1-M2, F1-F2 and R1-R2 (docs/studies/S6.md).

    ``records[arm][seed]`` holds T, P, L, best_fitness, final_diversity and e_in (the median
    inside offset of that seed's finished drives, None without one). Part B and C arms pair
    with the default GA's seeds 0-4."""
    ga, rs = records["ga"], records["random_search"]
    base = {s: r for s, r in ga.items() if s in SEEDS_B}
    mastered = sum(r["T"] == 1.0 for r in ga.values())
    n1 = {a: median_of(records[a], "best_fitness") for a in ("n20", "n100")}
    n50 = median_of(base, "best_fitness")
    div = {a: median_of(records[a], "final_diversity") for a in ("p005", "p01", "s002", "s02")}
    div["default"] = median_of(base, "final_diversity")
    m2 = {
        a: paired_by_seed(values(records[a], "best_fitness"), values(base, "best_fitness"), -1)
        for a in ("p01", "s02")
    }
    t_prog = median_of(records["progress"], "T")
    t_fin = median_of(records["finish_bonus"], "T")
    r1 = paired_by_seed(values(ga, "e_in"), {s: expert_e_in for s in ga})
    return {
        "G1": {
            **paired_by_seed(values(ga, "T"), values(rs, "T")),
            "mastery_seeds": mastered,
            "mastery": 2 * mastered >= len(ga),
        },
        "G2": paired_by_seed(values(ga, "P"), values(rs, "P")),
        "N1": {
            "median_best_fitness": {"n20": n1["n20"], "n50": n50, "n100": n1["n100"]},
            "supported": n50 >= n1["n20"] and n50 >= n1["n100"],
        },
        "M1": {
            "median_final_diversity": div,
            "rate": div["p005"] < div["default"] < div["p01"],
            "strength": div["s002"] < div["default"] < div["s02"],
            "supported": div["p005"] < div["default"] < div["p01"]
            and div["s002"] < div["default"] < div["s02"],
        },
        "M2": {**m2, "supported": m2["p01"]["supported"] and m2["s02"]["supported"]},
        "F1": paired_by_seed(values(records["finish_bonus"], "L"), values(base, "L")),
        "F2": {
            "median_T": {"progress": t_prog, "finish_bonus": t_fin},
            "supported": t_prog <= t_fin,
        },
        "R1": {**r1, "expert_e_in": expert_e_in},
        "R2": paired_by_seed(values(base, "e_in"), values(records["finish_bonus"], "e_in")),
    }


def against_default(records: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Every Part B and C arm against the default GA on seeds 0-4: the paired differences of T,
    P, C, L and the best fitness (the fitness only within one formula, rule 3)."""
    base = {s: r for s, r in records["ga"].items() if s in SEEDS_B}
    out = {}
    for arm in (*SWEEP_ARMS, *FITNESS_ARMS):
        keys = ("T", "P", "C", "L") + (("best_fitness",) if arm in SWEEP_ARMS else ())
        out[arm] = {k: paired_by_seed(values(records[arm], k), values(base, k)) for k in keys}
    return out


# ── exploits, a failure, and the checks that tie S6 to earlier work ─────────────────────────


def rank_correlation(x: list[float], y: list[float]) -> float:
    """Spearman's rank correlation: the correlation of the ranks (no ties in these floats).
    Example: x = (1, 2, 3), y = (30, 10, 20): ranks (0, 1, 2) and (2, 0, 1), rho = -0.5."""
    rx = np.argsort(np.argsort(np.asarray(x)))
    ry = np.argsort(np.argsort(np.asarray(y)))
    return float(np.corrcoef(rx, ry)[0, 1])


def side_of_road(records: dict[str, dict[int, dict[str, Any]]], margin: float = 1.0) -> dict:
    """POST HOC (not a hypothesis; added after seeing the per-bend offsets): which side of the
    road each chosen network keeps, by the median offset e of its finished training-loop drives:
    left of +margin, right of -margin, or in between. The loop runs anticlockwise, so its left
    side is the inside of the whole lap: the shorter way round. With the lap time L of each
    network: the counts and median L per side, and the rank correlation of the offset with L."""
    rows = [
        (arm, seed, r["median_offset_m"], r["L"])
        for arm in ARMS
        for seed, r in sorted(records[arm].items())
        if r["median_offset_m"] is not None and r["L"] is not None
    ]

    def side(e: float) -> str:
        return "left" if e > margin else "right" if e < -margin else "middle"

    out: dict[str, Any] = {"margin_m": margin, "n": len(rows)}
    for name in ("left", "middle", "right"):
        mine = [r for r in rows if side(r[2]) == name]
        out[name] = {
            "networks": len(mine),
            "L": band([r[3] for r in mine]),
            "by_arm": {a: sum(r[0] == a for r in mine) for a in ARMS},
        }
    out["offset_vs_L_rank_correlation"] = rank_correlation(
        [r[2] for r in rows], [r[3] for r in rows]
    )
    out["e_in_vs_L_rank_correlation"] = rank_correlation(
        [records[a][s]["e_in"] for a, s, _, _ in rows], [r[3] for r in rows]
    )
    return out


def exploit_search(records: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    """Chosen networks that score by a route the designer did not intend, by three flags:
    overfit (T = 1, held-out C = 0); jerky (training smoothness in the worst 10 % of all chosen
    networks); just in time (a training-loop finish within 2 s of the limit). Plus how many of
    the masters (T = 1) also lap every unseen start."""
    every = [(a, s, r) for a in ARMS for s, r in sorted(records[a].items())]
    # the jerkiest tenth of the chosen networks (at least one; 6 of S6's 60): smoothness is the
    # mean steering change per step, so larger is jerkier
    smooth = sorted((r["training_smoothness"] for _, _, r in every), reverse=True)
    cut = smooth[max(1, len(smooth) // 10) - 1]
    flagged = []
    for arm, seed, r in every:
        flags = []
        if r["T"] == 1.0 and r["C"] == 0.0:
            flags.append("overfit")
        if r["training_smoothness"] >= cut:
            flags.append("jerky")
        limit = r["training_time_limit_s"]
        if any(t > limit - LIMIT_MARGIN_S for t in r["training_finish_times_s"]):
            flags.append("just_in_time")
        if flags:
            flagged.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "run_id": r["run_id"],
                    "unit": r["unit"],
                    "flags": flags,
                    "T": r["T"],
                    "C": r["C"],
                    "P": r["P"],
                    "training_smoothness": r["training_smoothness"],
                    "unseen_start_completion": r.get("unseen_start_completion"),
                }
            )
    masters = [(a, s, r) for a, s, r in every if r["T"] == 1.0]
    return {
        "smoothness_cut": cut,
        "flagged": flagged,
        "counts": {
            f: sum(f in x["flags"] for x in flagged) for f in ("overfit", "jerky", "just_in_time")
        },
        "masters": len(masters),
        "masters_lapping_every_unseen_start": sum(
            r.get("unseen_start_completion") == 1.0 for _, _, r in masters
        ),
        "masters_by_arm": {
            a: {
                "masters": sum(x[0] == a for x in masters),
                "lap_every_unseen_start": sum(
                    x[0] == a and x[2].get("unseen_start_completion") == 1.0 for x in masters
                ),
            }
            for a in ARMS
        },
    }


def worst_episode(records: dict[int, dict[str, Any]], runs: dict[int, Path]) -> dict[str, Any]:
    """The held-out episode with the lowest progress among an arm's chosen checkpoints (ties: the
    lowest seed, the earliest track and start)."""
    best: tuple[Any, ...] | None = None
    for seed in sorted(records):
        unit = records[seed]["unit"]
        card = read_card(RunStore.open(runs[seed]), unit)
        for part in card["tracks"]:
            if part["split"] != "heldout":
                continue
            for e in part["episodes"]:
                key = (e["progress_frac"], seed, part["track"], e["start_index"])
                if best is None or key < best[0]:
                    best = (key, unit, part["track"], e)
    assert best is not None
    (_, seed, _, _), unit, track, e = best
    return {
        "seed": seed,
        "run_id": runs[seed].name,
        "unit": unit,
        "track": track,
        "start_index": e["start_index"],
        "start_s0_m": e["start_s0_m"],
        "start_lateral_m": e["start_lateral_m"],
        "status": e["status"],
        "progress_frac": e["progress_frac"],
        "steps": e["steps"],
        "end_s_m": e["end_s_m"],
        "mean_speed": e["mean_speed"],
    }


def inspect_failure(runs_dir: Path, failure: dict[str, Any]) -> dict[str, Any]:
    """Drive the failing episode again exactly as the exam did (the track's 10 starts as one
    batch) and say what the network did 1 s, 0.5 s and 1 step before it ended: the observation,
    its action and the expert's action for the same state (the Timeline replays the same drive)."""
    network = RunStore.open(runs_dir / failure["run_id"]).load_checkpoint(failure["unit"])
    track = Track.from_json(DEFAULT_TRACKS_DIR / f"{failure['track']}.json")
    labelled: list[ExpertLabelled] = []

    def policy_for(world):
        labelled.append(ExpertLabelled(MLPPolicy(network), ScriptedExpert(world)))
        return labelled[0]

    part, recording = exam_track(track, policy_for)
    car = failure["start_index"]
    n = int(driven_steps(recording)[car])
    labels = np.array(labelled[0].labels).reshape(-1, recording.n_cars, 2)
    moments = {}
    for name, back in (("one_second_before", 30), ("half_second_before", 15), ("last_step", 1)):
        t = max(0, n - back)
        moments[name] = {
            "step": t,
            "observation": [round(float(v), 4) for v in recording.observation[t, car]],
            "speed_mps": round(float(recording.v[t, car]), 3),
            "network_action": [round(float(v), 4) for v in recording.action[t, car]],
            "expert_action": [round(float(v), 4) for v in labels[t, car]],
        }
    return {
        "steps_driven": n,
        "same_as_the_card": part.episodes[car]["status"] == failure["status"],
        "moments": moments,
    }


def same_as_reference(runs_dir: Path, run: Path) -> dict[str, Any]:
    """The default GA's seed 0 against task 7.5's run (``REFERENCE_RUN``, same config and seed,
    30 generations): every population line and every metrics line but the wall-clock time, for
    the reference's generations. A free determinism check."""
    ref = runs_dir / REFERENCE_RUN
    if not ref.is_dir():
        return {"reference_present": False}
    a, b = RunStore.open(run), RunStore.open(ref)
    pop_a = a.read_extra_lines("population")[:REFERENCE_UNITS]
    pop_b = b.read_extra_lines("population")[:REFERENCE_UNITS]

    def same(x: dict[str, Any], y: dict[str, Any]) -> bool:
        """Equal on every field both lines have (later tasks may add fields), but the clock."""
        keys = (set(x) & set(y)) - {"wall_clock_s"}
        return all(x[k] == y[k] for k in keys)

    met_a = a.read_metrics()[:REFERENCE_UNITS]
    met_b = b.read_metrics()[:REFERENCE_UNITS]
    return {
        "reference_present": True,
        "reference": REFERENCE_RUN,
        "generations_compared": min(len(pop_a), len(pop_b)),
        "population_lines_equal": sum(same(x, y) for x, y in zip(pop_a, pop_b, strict=False)),
        "metrics_lines_equal": sum(same(x, y) for x, y in zip(met_a, met_b, strict=False)),
    }


def shared_history(found: dict[str, Any]) -> dict[str, Any]:
    """How long each fitness ablation evolves exactly as the default GA with the same seed: per
    seed, the first generation in which some network finishes some start (under every formula
    f = p until then) and the first generation whose finished or crashed starts differ."""
    out: dict[str, Any] = {}
    for arm in FITNESS_ARMS:
        per_seed = {}
        for seed in SEEDS_B:
            mine = RunStore.open(found["arms"][arm][seed]).read_extra_lines("population")
            base = RunStore.open(found["arms"]["ga"][seed]).read_extra_lines("population")
            first = next((x["unit"] for x in base if any(n > 0 for n in x["finished"])), None)
            differ = next(
                (
                    x["unit"]
                    for x, y in zip(base, mine, strict=False)
                    if x["finished"] != y["finished"] or x["crashed"] != y["crashed"]
                ),
                None,
            )
            per_seed[str(seed)] = {"first_finish": first, "first_difference": differ}
        out[arm] = per_seed
    return out


def database_check(runs_dir: Path, ids: list[str]) -> dict[str, Any]:
    try:
        with closing(connect(runs_dir / DB_NAME)) as conn:
            listed = runs_of_study(conn, STUDY)
            row = conn.execute(
                "SELECT n_runs, methods FROM studies WHERE study_id = ?", (STUDY,)
            ).fetchone()
    except sqlite3.Error as exc:
        return {"error": str(exc)}
    return {
        "same_runs_as_the_folders": listed == sorted(ids),
        "n_runs": row[0] if row else None,
        "methods": json.loads(row[1]) if row else None,
    }


# ── analysis ────────────────────────────────────────────────────────────────────────────────

SUMMARY_KEYS = (
    "T",
    "L",
    "P",
    "C",
    "heldout_crash",
    "training_smoothness",
    "best_fitness",
    "final_diversity",
    "e_in",
    "e_wide",
    "e_exit",
    "unseen_start_completion",
    "median_offset_m",
    "unit",
    "first_unit_with_a_full_finisher",
    "env_steps_at_first_full_finisher",
    "env_steps_at_first_T1_exam",
)
BUDGET_KEYS = (
    "units",
    "env_steps",
    "env_steps_at_chosen_unit",
    "networks_scored",
    "wall_clock_s",
    "exam_wall_clock_s",
)


def curve_band(records: dict[int, dict[str, Any]], key: str, grid: list[int]) -> dict[str, Any]:
    """An arm's exam measure against environment steps: at each step count x of ``grid``, every
    seed's last examined checkpoint at or before x, then the median and IQR over seeds."""
    out = []
    for x in grid:
        vals = []
        for r in records.values():
            seen = [c for c in r["curve"].values() if c["env_steps"] <= x]
            if seen:
                vals.append(max(seen, key=lambda c: c["env_steps"])[key])
        out.append({"env_steps": x, **band(vals)})
    return {"key": key, "points": out}


def analyse(runs_dir: Path, workers: int, inspect: bool = True) -> dict[str, Any]:
    """Read S6's runs and compute everything the report states."""
    found = find_runs(runs_dir)
    records = {
        arm: {s: run_record(p, arm) for s, p in sorted(by_seed.items())}
        for arm, by_seed in found["arms"].items()
    }
    drives = drive_all(runs_dir, records, workers)
    for by_seed in records.values():
        for r in by_seed.values():
            d = drives[r["run_id"]]
            line = d["racing_line"]
            r["finished_training_drives"] = d["finished_drives"]
            r["racing_line"] = line
            r["e_in"] = line["e_in"] if line else None
            r["e_wide"] = line["e_wide"] if line else None
            r["e_exit"] = line["e_exit"] if line else None
            r["unseen_start_completion"] = d["unseen_start_completion"]
            r["unseen_start_progress"] = d["unseen_start_progress"]
            r["median_offset_m"] = d["median_offset_m"]
            # the re-drive must agree with the card: the same finishes from the same starts
            r["redrive_matches_card"] = d["finished_drives"] == round(10 * r["T"])
    expert = drives["expert"]
    summary = {
        arm: {
            **{k: band([r[k] for r in by.values()]) for k in SUMMARY_KEYS},
            "budget": {k: band([r["budget"][k] for r in by.values()]) for k in BUDGET_KEYS},
            "T1_seeds": sum(r["T"] == 1.0 for r in by.values()),
            "n_seeds": len(by),
        }
        for arm, by in records.items()
    }
    grid = list(range(0, BUDGET_STEPS + 1, 1_000_000))
    curves = {
        arm: {k: curve_band(records[arm], k, grid) for k in ("T", "P")}
        for arm in ("ga", "random_search")
    }
    failure = worst_episode(records["ga"], found["arms"]["ga"])
    made = {
        "design": {
            "study": STUDY,
            "budget_env_steps": BUDGET_STEPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "min_pairs": MIN_PAIRS,
            "arms": {
                a: {
                    "base": Path(spec["base"]).name,
                    "change": list(spec["change"]) if spec["change"] else None,
                    "seeds": list(spec["seeds"]),
                    "tag": spec["tag"],
                    "exams_at_every_checkpoint": bool(spec.get("exams", False)),
                }
                for a, spec in ARMS.items()
            },
            "bends": [
                b.to_dict()
                for b in bends(Track.from_json(DEFAULT_TRACKS_DIR / "train_loop_01.json"))
            ],
            "unseen_starts": UNSEEN_STARTS.to_dict(),
        },
        "run_ids": {a: {str(s): p.name for s, p in by.items()} for a, by in found["arms"].items()},
        "records": {a: {str(s): r for s, r in by.items()} for a, by in records.items()},
        "expert": expert,
        "summary": summary,
        "curves": curves,
        "hypotheses": hypotheses(records, expert["racing_line"]["e_in"]),
        "against_default": against_default(records),
        "part_a_exploratory": {  # not hypotheses: what else differs between the Part A arms
            k: paired_by_seed(values(records["ga"], k), values(records["random_search"], k))
            for k in ("L", "C", "best_fitness", "env_steps_at_first_T1_exam")
        },
        # is the finish bonus's held-out gain the formula or its earlier checkpoint? the default
        # GA's first examined checkpoint that laps every training start, against finish_bonus
        "finish_bonus_vs_default_at_first_T1": paired_by_seed(
            values(records["finish_bonus"], "P"), values(records["ga"], "P_at_first_T1_exam")
        ),
        "default_first_T1_vs_chosen": {
            k: paired_by_seed(
                values(records["ga"], f"{k}_at_first_T1_exam"), values(records["ga"], k)
            )
            for k in ("P", "C")
        },
        "exploits": exploit_search(records),
        "post_hoc_side_of_road": side_of_road(records),
        "shared_history": shared_history(found),
        "same_as_task_7_5": same_as_reference(runs_dir, found["arms"]["ga"][0]),
        "database": database_check(runs_dir, [p.name for p in found["all"]]),
        "failure": {"episode": failure, "detail": None},
    }
    if inspect:
        made["failure"]["detail"] = inspect_failure(runs_dir, failure)
    return made


def _fmt(v: Any, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def print_results(made: dict[str, Any]) -> None:
    print("chosen checkpoint, median [IQR] over seeds:")
    print(f"  {'arm':14s} {'T':>18s} {'L (s)':>22s} {'P':>18s} {'C':>18s} {'e_in (m)':>20s}")
    for arm, s in made["summary"].items():
        cells = [
            f"{_fmt(s[k]['median'])} [{_fmt(s[k]['q1'], 2)}, {_fmt(s[k]['q3'], 2)}]"
            for k in ("T", "L", "P", "C", "e_in")
        ]
        print(f"  {arm:14s} " + " ".join(f"{c:>20s}" for c in cells))
    print(f"  expert e_in {made['expert']['racing_line']['e_in']:.3f} m")
    for name, h in made["hypotheses"].items():
        extra = ""
        if "ci" in h:
            extra = f"{h['estimate']:+.3f} [{h['ci'][0]:+.3f}, {h['ci'][1]:+.3f}] n={h['n_seeds']}"
        print(f"  {name}: supported={h['supported']} {extra}")
    print(
        "exploits:",
        made["exploits"]["counts"],
        "masters",
        made["exploits"]["masters"],
        "lapping every unseen start",
        made["exploits"]["masters_lapping_every_unseen_start"],
    )
    print("same as task 7.5:", made["same_as_task_7_5"])
    print("database:", made["database"])


# ── figures ─────────────────────────────────────────────────────────────────────────────────


def draw_figures(made: dict[str, Any], runs_dir: Path) -> None:
    """The figures of docs/studies/S6.md (drawn by scripts/study_s6_figures.py)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "study_s6_figures", Path(__file__).parent / "study_s6_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.draw(made, FIGURES_DIR, runs_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", action="store_true", help="make the missing runs first")
    parser.add_argument("--only-train", action="store_true", help="stop after training")
    parser.add_argument("--figures", action="store_true", help="draw the figures (matplotlib)")
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated arms to train")
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    if args.train:
        train(runs_dir, args.workers, tuple(a for a in args.arms.split(",") if a))
        if args.only_train:
            return
    made = analyse(runs_dir, args.workers)
    RESULTS_FILE.write_text(json.dumps(made, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_results(made)
    if args.figures:
        draw_figures(made, runs_dir)


if __name__ == "__main__":
    main()
