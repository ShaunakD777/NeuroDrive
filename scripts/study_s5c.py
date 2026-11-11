"""Study S5c: does the teacher matter, and does DAgger beat cloning on held-out tracks?

Run from the repository root:

    uv run python scripts/study_s5c.py --train                         # make the runs (~1.5 h)
    uv run python scripts/study_s5c.py                                 # analyse them (numbers)
    uv run --with matplotlib python scripts/study_s5c.py --figures     # also draw the figures

Design and hypotheses: docs/studies/S5c.md (written in the task 6.7 Brief, before any run).

Six arms, seeds 0-9 each, every run with ``"study": "S5c"`` (ADR-043, ADR-052):

* Part A, the teacher: ``human`` (imitation on the 22 keyboard drives, merged), ``e_match`` (the
  expert from the same 22 starts, the same frames per drive) and ``e_full`` (the expert's whole
  drives from those starts); 300 epochs, examined every 3rd epoch, as Study S5a.
* Part B, DAgger: ``dagger_wide`` (configs/dagger_wide.yaml on the expert dataset of 6.1a) and
  ``clone_wide`` (imitation on the wide expert dataset, 300 epochs examined every 30th), the two
  arms of the 6.6 follow-up (ADR-050) retrained for all ten seeds.
* Part C: ``dagger_human`` (configs/dagger_wide.yaml with the human dataset as round 1).
* The baselines, made again for S5c: random networks seeds 0-9, the hand-wired neuron, the expert.

Per run and examined checkpoint: V, the validation loss of its own method; P, the held-out
median progress; C, the held-out completion; T, the training loop's completion. Three checkpoint
rules that never read a held-out track (last, min_loss, training_exam: S5a's) pick one
checkpoint per run, and its held-out numbers are what is compared. Paired differences are
medians over seeds of per-seed differences with bootstrap 95 % intervals (10 000 resamples,
seed 2026). Numbers go to docs/studies/S5c_results.json (the same bytes on every analysis of the
same runs); with --figures the pictures go to docs/studies/figures/.
"""

import argparse
import importlib.util
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neurodrive.core.rng import make_rng
from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.policies import MLPPolicy
from neurodrive.env.track import Track
from neurodrive.experiment.baselines import make_baselines
from neurodrive.experiment.config import load_config, parse_config, resolved
from neurodrive.experiment.dataset_view import nearest_distances
from neurodrive.experiment.db import DB_NAME, connect, run_folders, runs_of_study
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, exam_track
from neurodrive.experiment.runner import train_dagger_run, train_imitation_run
from neurodrive.experiment.stats import bootstrap_median_ci, summarise
from neurodrive.experiment.store import RunStore
from neurodrive.learn.supervised.dagger import ExpertLabelled
from neurodrive.learn.supervised.demos import (
    DEFAULT_DATASETS_DIR,
    DemoDataset,
    driven_steps,
    find_dataset,
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s5a = _load("study_s5a")  # the checkpoint rules and the exam-card readers of Study S5a

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "docs" / "studies" / "S5c_results.json"
FIGURES_DIR = ROOT / "docs" / "studies" / "figures"
IMITATION = ROOT / "configs" / "imitation.yaml"
DAGGER_WIDE = ROOT / "configs" / "dagger_wide.yaml"

STUDY = "S5c"
DATASETS_DIR = DEFAULT_DATASETS_DIR
HUMAN = "20261111-073610-keyboard-s5c"  # the 22 keyboard drives, merged by `nd demo-merge`
E_MATCH = "20261111-073613-expert-s5c-match"  # the expert from the same starts, same frames
E_FULL = "20261111-073617-expert-s5c-full"  # the expert's whole drives from the same starts
WIDE = "20261110-111643-expert-wide"  # 100 drives from wide starts (6.6 follow-up)
BASE = "20261104-084617-expert-default"  # 6.1a: DAgger's round-1 demonstrations
SEEDS = tuple(range(10))
RANDOM_SEEDS = tuple(range(10))
BOOTSTRAP_SEED = 2026
FOLLOW_UP_DAGGER_TAG = "checkwide"  # the 6.6 follow-up's runs: checkwide-s<k>, checkwide-bc-s<k>
FOLLOW_UP_SEEDS = tuple(range(5))

# name -> how to train it. ``epochs``/``every`` apply to imitation arms only.
ARMS: dict[str, dict[str, Any]] = {
    "human": dict(
        method="imitation", config=IMITATION, dataset=HUMAN, tag="s5c-human", epochs=300, every=3
    ),
    "e_match": dict(
        method="imitation", config=IMITATION, dataset=E_MATCH, tag="s5c-ematch", epochs=300, every=3
    ),
    "e_full": dict(
        method="imitation", config=IMITATION, dataset=E_FULL, tag="s5c-efull", epochs=300, every=3
    ),
    "clone_wide": dict(
        method="imitation", config=IMITATION, dataset=WIDE, tag="s5c-cw", epochs=300, every=30
    ),
    "dagger_wide": dict(method="dagger", config=DAGGER_WIDE, dataset=BASE, tag="s5c-dw"),
    "dagger_human": dict(method="dagger", config=DAGGER_WIDE, dataset=HUMAN, tag="s5c-dh"),
}
RULES = ("last", "min_loss", "training_exam")
POST_HOC_RULES = ("earliest_lap",)  # added after the results; never part of a hypothesis
MEASURES = ("heldout_progress", "heldout_completion", "heldout_crash", "training_completion")
HELDOUT_TRACKS = ("hairpins_01", "monza_ccw", "sweeper_01", "train_loop_01_mirror")


# ── training ────────────────────────────────────────────────────────────────────────────────


def study_config(arm: str, seed: int):
    """The config of ``arm`` for ``seed``: the shipped YAML, the arm's dataset pinned by id,
    its length and checkpoint cadence, tag ``<arm tag>-s<seed>``."""
    spec = ARMS[arm]
    if spec["method"] == "imitation":
        data = resolved(load_config(spec["config"], epochs=spec["epochs"]))
        data["checkpoint_every"] = spec["every"]
    else:
        data = resolved(load_config(spec["config"]))
    data["data"]["dataset"] = spec["dataset"]
    return parse_config(data, seed=seed, tag=f"{spec['tag']}-s{seed}")


def _train_job(job: tuple[str, int, str, str]) -> str:
    arm, seed, sweep_id, runs_dir = job
    meta = {"sweep": sweep_id, "study": STUDY}
    train = train_dagger_run if ARMS[arm]["method"] == "dagger" else train_imitation_run
    found = train(study_config(arm, seed), runs_dir, extra_meta=meta, datasets_dir=DATASETS_DIR)
    name = found.store.path.name
    print(f"{arm} seed {seed}: {name}", flush=True)
    return name


def _baselines_job(runs_dir: str) -> list[str]:
    runs = make_baselines(runs_dir, seeds=RANDOM_SEEDS, study=STUDY)
    return [r.store.path.name for r in runs]


def existing(runs_dir: Path) -> set[tuple[str, int | None]]:
    """(arm tag, seed) of every completed S5c run already in ``runs_dir``, so a second
    ``--train`` only makes what is missing."""
    done = set()
    for path in run_folders(runs_dir):
        store = RunStore.open(path)
        meta = store.read_meta()
        if meta.get("study") == STUDY and meta.get("status") == "completed":
            done.add((meta["method"], meta["tag"]))
    return done


def train(runs_dir: Path, workers: int, arms: tuple[str, ...]) -> None:
    """Make the missing runs, several processes side by side. Every run depends only on its
    own seed and data, so running them together changes no number, only the wall-clock times."""
    done = existing(runs_dir)
    when = f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    jobs = []
    for arm in arms:  # the slow imitation arms first, so the short ones fill the gaps
        spec = ARMS[arm]
        for seed in SEEDS:
            if (spec["method"], f"{spec['tag']}-s{seed}") not in done:
                jobs.append((arm, seed, f"{when}-{spec['method']}-{spec['tag']}", str(runs_dir)))
    have_baselines = any(method == "random_network" for method, _ in done)
    with ProcessPoolExecutor(workers) as pool:
        baselines = None if have_baselines else pool.submit(_baselines_job, str(runs_dir))
        list(pool.map(_train_job, jobs))
        if baselines is not None:
            baselines.result()


# ── reading the runs ────────────────────────────────────────────────────────────────────────


def arm_of(tag: str) -> str | None:
    """The arm a run tag ``<arm tag>-s<seed>`` belongs to (None for a baseline)."""
    for arm, spec in ARMS.items():
        if tag.startswith(f"{spec['tag']}-s") and tag[len(spec["tag"]) + 2 :].isdigit():
            return arm
    return None


def find_runs(runs_dir: Path) -> dict[str, Any]:
    """S5c's run folders by their ``study`` field. Stops with a message when a run is missing,
    unfinished or repeated."""
    arms: dict[str, dict[int, list[Path]]] = {a: {} for a in ARMS}
    baselines: dict[str, dict[Any, list[Path]]] = {}
    every = []
    for path in run_folders(runs_dir):
        store = RunStore.open(path)
        meta = store.read_meta()
        if meta.get("study") != STUDY:
            continue
        every.append(path)
        if meta["status"] != "completed":
            raise SystemExit(f"{path.name} is {meta['status']}; finish it (nd resume) first")
        seed = store.read_config().get("seed")
        arm = arm_of(meta["tag"])
        if arm is not None:
            arms[arm].setdefault(seed, []).append(path)
        else:
            baselines.setdefault(meta["method"], {}).setdefault(seed, []).append(path)
    for arm, by_seed in arms.items():
        if set(by_seed) != set(SEEDS) or any(len(p) > 1 for p in by_seed.values()):
            raise SystemExit(f"S5c needs exactly one {arm} run for each of seeds {list(SEEDS)}")
    wanted = {"random_network": set(RANDOM_SEEDS), "hand_neuron": {None}, "scripted_expert": {None}}
    for method, seeds in wanted.items():
        got = baselines.get(method, {})
        if set(got) != seeds or any(len(p) > 1 for p in got.values()):
            raise SystemExit(
                f"S5c needs exactly one {method} baseline for each of {sorted(seeds, key=str)}"
            )
    return {
        "arms": {a: {s: p[0] for s, p in by.items()} for a, by in arms.items()},
        "random_network": {s: p[0] for s, p in baselines["random_network"].items()},
        "hand_neuron": baselines["hand_neuron"][None][0],
        "scripted_expert": baselines["scripted_expert"][None][0],
        "all": every,
    }


def run_record(path: Path) -> dict[str, Any]:
    """What one run did: validation loss and exam measures at every examined unit, the three
    checkpoint rules' choices (training-track fields only), the budgets in every currency."""
    store = RunStore.open(path)
    lines = store.read_metrics()
    meta = store.read_meta()
    cards = {u: s5a.measures(s5a.read_card(store, u)) for u in store.exam_units()}
    units = [u for u in sorted(cards) if u >= 1]
    val = [lines[u - 1]["val_loss"] for u in units]
    lap = [cards[u]["training_completion"] for u in units]
    rules = s5a.checkpoint_rules(units, val, lap)  # never reads a held-out number
    rules["earliest_lap"] = s5a.earliest_lap_rule(units, lap)  # POST HOC (see POST_HOC_RULES)
    full = [u for u, t in zip(units, lap, strict=True) if t == 1.0]
    last = lines[-1]
    return {
        "run_id": path.name,
        "seed": store.read_config()["seed"],
        "units": units,
        "curve": {
            str(u): {
                "val_loss": v,
                "heldout_progress": cards[u]["heldout_progress"],
                "training_completion": cards[u]["training_completion"],
            }
            for u, v in zip(units, val, strict=True)
        },
        "first_full_lap_unit": full[0] if full else None,
        "rules": {
            n: {"unit": u, "val_loss": lines[u - 1]["val_loss"], **cards[u]}
            for n, u in rules.items()
        },
        "budget": {
            "demo_frames": last["demo_frames_total"],
            "teacher_queries": last["teacher_queries_total"],
            "env_steps": last["env_steps_total"],
            "wall_clock_s": last["wall_clock_s"],
            "exam_wall_clock_s": meta["exam_wall_clock_s"],
            "train_frames_final": last.get("train_frames"),
            "dataset_frames": meta["dataset"]["demo_frames"],
        },
        "onpolicy_loss_last_round": last.get("onpolicy_loss"),
        "collect_crashed_last_round": last.get("collect_crashed"),
    }


def band(values: list[float]) -> dict[str, float]:
    s = summarise(values)
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


def label_ratio(clone_frames: int, demo_frames: int, queries: int) -> float:
    """N / (F + Q): the clone's labels over DAgger's (its demonstrations plus its teacher queries).
    Example: 196,765 / (39,338 + 170,000) = 0.94."""
    return clone_frames / (demo_frames + queries)


def selected(records: dict[int, dict[str, Any]], rule: str, measure: str) -> list[float]:
    """The measure of the checkpoint ``rule`` picked, one value per seed in seed order."""
    return [records[s]["rules"][rule][measure] for s in sorted(records)]


def hypotheses(
    records: dict[str, dict[int, dict[str, Any]]], rule: str = "training_exam"
) -> dict[str, Any]:
    """The pre-registered rules T1-T3, D-1 to D-3 and H-1 (docs/studies/S5c.md), reading the
    checkpoint ``rule`` picked. With the default they are the pre-registered tests; with the
    post hoc rule ``earliest_lap`` the same rules are an exploration, not a test."""
    P = "heldout_progress"

    def paired(a: str, b: str, rule_a: str = rule, rule_b: str = rule) -> dict[str, Any]:
        out = paired_difference(selected(records[a], rule_a, P), selected(records[b], rule_b, P))
        out["supported"] = out["estimate"] > 0 and out["ci"][0] > 0
        return out

    full_lap = sum(r["first_full_lap_unit"] is not None for r in records["human"].values())
    med_full = summarise(selected(records["e_full"], rule, P)).median
    med_match = summarise(selected(records["e_match"], rule, P)).median
    iqr = {
        a: summarise(selected(records[a], rule, P)).q3 - summarise(selected(records[a], rule, P)).q1
        for a in ("dagger_wide", "clone_wide")
    }
    return {
        "T1": paired("e_match", "human"),
        "T2": {
            "seeds_with_a_full_lap_checkpoint": full_lap,
            "n_seeds": len(records["human"]),
            "supported": 2 * full_lap >= len(records["human"]),
        },
        "T3": {
            "median_e_full": med_full,
            "median_e_match": med_match,
            "supported": med_full >= med_match,
        },
        "D1": paired("dagger_wide", "clone_wide"),
        "D2": {"iqr": iqr, "supported": iqr["dagger_wide"] < iqr["clone_wide"]},
        "D3": paired("dagger_wide", "dagger_wide", rule, "last"),
        "H1": paired("dagger_human", "human"),
    }


# ── the checks that tie S5c to earlier work ─────────────────────────────────────────────────


def same_as_follow_up(runs_dir: Path, found: dict[str, Any]) -> dict[str, Any]:
    """Seeds 0-4 of dagger_wide and clone_wide against the 6.6 follow-up's runs (tags
    checkwide-s<k> and checkwide-bc-s<k>): every metrics line but its wall-clock time and every
    exam card both runs hold, apart from the run name in the card's ``policy``."""
    old: dict[tuple[str, int], Path] = {}
    for path in run_folders(runs_dir):
        store = RunStore.open(path)
        meta, config = store.read_meta(), store.read_config()
        seed = config.get("seed")
        if meta.get("study") is None and meta.get("status") == "completed":
            if config.get("tag") == f"{FOLLOW_UP_DAGGER_TAG}-s{seed}":
                old.setdefault(("dagger_wide", seed), path)
            elif config.get("tag") == f"{FOLLOW_UP_DAGGER_TAG}-bc-s{seed}":
                old.setdefault(("clone_wide", seed), path)

    def lines(store: RunStore) -> list[dict[str, Any]]:
        return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]

    out: dict[str, Any] = {}
    for arm in ("dagger_wide", "clone_wide"):
        compared = equal_lines = cards = equal_cards = 0
        for seed in FOLLOW_UP_SEEDS:
            if (arm, seed) not in old or seed not in found["arms"][arm]:
                continue
            a, b = RunStore.open(found["arms"][arm][seed]), RunStore.open(old[(arm, seed)])
            compared += 1
            equal_lines += lines(a) == lines(b)
            for unit in sorted(set(a.exam_units()) & set(b.exam_units())):
                x, y = s5a.read_card(a, unit), s5a.read_card(b, unit)
                x.pop("policy"), y.pop("policy")
                cards += 1
                equal_cards += x == y
        out[arm] = {
            "seeds_compared": compared,
            "metrics_equal": equal_lines,
            "cards_compared": cards,
            "cards_equal": equal_cards,
        }
    return out


def baseline_values(found: dict[str, Any]) -> dict[str, Any]:
    def card(path: Path) -> dict[str, float]:
        return s5a.measures(s5a.read_card(RunStore.open(path), 0))

    return {
        "random_network": {s: card(p) for s, p in found["random_network"].items()},
        "hand_neuron": card(found["hand_neuron"]),
        "scripted_expert": card(found["scripted_expert"]),
    }


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


# ── the teachers' data, described ───────────────────────────────────────────────────────────


def describe_dataset(dataset: DemoDataset) -> dict[str, Any]:
    """What a teacher's frames look like: size, outcomes, how coarse the steering labels are,
    how much the steering changes from one step to the next, speed and 2-lap drive time."""
    steer, throttle = dataset.label[:, 0], dataset.label[:, 1]
    same_episode = dataset.episode[1:] == dataset.episode[:-1]
    change = np.abs(np.diff(steer))[same_episode]
    statuses: dict[str, int] = {}
    for e in dataset.meta["episodes"]:
        statuses[e["status"]] = statuses.get(e["status"], 0) + 1
    finished = [
        e["frames"] * dataset.meta["dt"]
        for e in dataset.meta["episodes"]
        if e["status"] == "finished"
    ]
    return {
        "frames": dataset.n_frames,
        "episodes": len(dataset.meta["episodes"]),
        "statuses": statuses,
        "steer_exactly_zero": float(np.mean(steer == 0.0)),
        "steer_full_lock": float(np.mean(np.abs(steer) == 1.0)),
        "steer_nearly_straight": float(np.mean(np.abs(steer) < 0.1)),
        "throttle_full": float(np.mean(throttle == 1.0)),
        "throttle_braking": float(np.mean(throttle < 0.0)),
        "distinct_steer_values": int(np.unique(steer).size),
        "mean_abs_steer_change_per_step": float(change.mean()) if change.size else None,
        "mean_speed_mps": float(dataset.v.mean()),
        "median_finished_drive_s": float(np.median(finished)) if finished else None,
        "n_finished": len(finished),
    }


# ── a failure, looked at ────────────────────────────────────────────────────────────────────


def worst_episode(records: dict[int, dict[str, Any]], runs: dict[int, Path]) -> dict[str, Any]:
    """The held-out episode with the lowest progress among the training-exam checkpoints of an
    arm (ties: the lowest seed, the earliest track and start)."""
    best: tuple[Any, ...] | None = None
    for seed in sorted(records):
        unit = records[seed]["rules"]["training_exam"]["unit"]
        card = s5a.read_card(RunStore.open(runs[seed]), unit)
        for part in card["tracks"]:
            if part["split"] != "heldout":
                continue
            for e in part["episodes"]:
                key = (e["progress_frac"], seed, part["track"], e["start_index"])
                if best is None or key < best[0]:
                    best = (key, unit, part["track"], e)
    assert best is not None
    _, unit, track, e = best
    seed = best[0][1]
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


def inspect_failure(
    runs_dir: Path, failure: dict[str, Any], dataset_ids: dict[str, str]
) -> dict[str, Any]:
    """Drive the failing episode again exactly as the exam did (the track's 10 starts as one
    batch) and say what the network did 1 s, 0.5 s and 1 step before it ended: the observation,
    the network's action, the expert's label for the same state, and how far the state is from
    the nearest frame of each dataset (the Timeline replays the same drive)."""
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
    frames = {
        name: DemoDataset.load(find_dataset(DATASETS_DIR, ident)).observation
        for name, ident in dataset_ids.items()
    }
    moments = {}
    for name, back in (("one_second_before", 30), ("half_second_before", 15), ("last_step", 1)):
        t = max(0, n - back)
        obs = recording.observation[t, car]
        moments[name] = {
            "step": t,
            "observation": [round(float(v), 4) for v in obs],
            "speed_mps": round(float(recording.v[t, car]), 3),
            "network_action": [round(float(v), 4) for v in recording.action[t, car]],
            "expert_label": [round(float(v), 4) for v in labels[t, car]],
            "nearest_distance": {
                k: round(float(nearest_distances(obs[None, :], f)[0]), 4) for k, f in frames.items()
            },
        }
    return {
        "steps_driven": n,
        "same_as_the_card": part.episodes[car]["status"] == failure["status"],
        "moments": moments,
    }


# ── analysis ────────────────────────────────────────────────────────────────────────────────


def analyse(runs_dir: Path, inspect: bool = True) -> dict[str, Any]:
    """Read S5c's runs and compute everything the report states. ``inspect`` also replays the
    worst human-clone failure (it drives one exam track and reads three datasets)."""
    found = find_runs(runs_dir)
    records = {
        arm: {s: run_record(p) for s, p in by_seed.items()}
        for arm, by_seed in found["arms"].items()
    }
    summary: dict[str, Any] = {}
    for arm, by_seed in records.items():
        summary[arm] = {
            "rules": {
                rule: {
                    **{m: band(selected(by_seed, rule, m)) for m in MEASURES},
                    "unit": band([by_seed[s]["rules"][rule]["unit"] for s in sorted(by_seed)]),
                    "per_track_progress": {
                        t: band(selected(by_seed, rule, f"progress:{t}")) for t in HELDOUT_TRACKS
                    },
                }
                for rule in (*RULES, *POST_HOC_RULES)
            },
            "budget": {
                k: band([by_seed[s]["budget"][k] for s in sorted(by_seed)])
                for k in (
                    "demo_frames",
                    "teacher_queries",
                    "env_steps",
                    "wall_clock_s",
                    "exam_wall_clock_s",
                    "dataset_frames",
                )
            },
            "seeds_with_a_full_lap_checkpoint": sum(
                r["first_full_lap_unit"] is not None for r in by_seed.values()
            ),
        }
    dagger, clone = records["dagger_wide"], records["clone_wide"]
    ratios = [
        label_ratio(
            clone[s]["budget"]["demo_frames"],
            dagger[s]["budget"]["demo_frames"],
            dagger[s]["budget"]["teacher_queries"],
        )
        for s in sorted(dagger)
    ]
    datasets = {
        name: describe_dataset(DemoDataset.load(find_dataset(DATASETS_DIR, ident)))
        for name, ident in (
            ("human", HUMAN),
            ("e_match", E_MATCH),
            ("e_full", E_FULL),
            ("expert_default", BASE),
        )
    }
    failure = worst_episode(records["human"], found["arms"]["human"])
    made = {
        "design": {
            "study": STUDY,
            "seeds": list(SEEDS),
            "arms": {
                a: {k: str(v) for k, v in spec.items() if k != "config"} for a, spec in ARMS.items()
            },
            "bootstrap_seed": BOOTSTRAP_SEED,
            "datasets": {
                "human": HUMAN,
                "e_match": E_MATCH,
                "e_full": E_FULL,
                "wide": WIDE,
                "base": BASE,
            },
        },
        "run_ids": {a: {str(s): p.name for s, p in by.items()} for a, by in found["arms"].items()},
        "records": {a: {str(s): r for s, r in by.items()} for a, by in records.items()},
        "summary": summary,
        "label_ratio": {"per_seed": ratios, **band(ratios)},
        "hypotheses": hypotheses(records),
        "hypotheses_post_hoc_earliest_lap": hypotheses(records, "earliest_lap"),
        "datasets": datasets,
        "baselines": baseline_values(found),
        "same_as_6_6_follow_up": same_as_follow_up(runs_dir, found),
        "database": database_check(runs_dir, [p.name for p in found["all"]]),
        "failure": {"episode": failure, "detail": None},
    }
    if inspect:
        made["failure"]["detail"] = inspect_failure(
            runs_dir,
            failure,
            {"human": HUMAN, "expert_default": BASE, "expert_match": E_MATCH},
        )
    return made


def print_results(made: dict[str, Any]) -> None:
    print("held-out median progress of the training-exam checkpoint, median [IQR] over seeds:")
    for arm, s in made["summary"].items():
        b = s["rules"]["training_exam"]["heldout_progress"]
        last = s["rules"]["last"]["heldout_progress"]
        early = s["rules"]["earliest_lap"]["heldout_progress"]
        print(
            f"  {arm:13s} selected {b['median']:.3f} [{b['q1']:.3f}, {b['q3']:.3f}]   "
            f"last {last['median']:.3f} [{last['q1']:.3f}, {last['q3']:.3f}]   "
            f"post hoc earliest-lap {early['median']:.3f} [{early['q1']:.3f}, {early['q3']:.3f}]"
        )
    for name, h in made["hypotheses"].items():
        extra = f"{h['estimate']:+.3f} [{h['ci'][0]:+.3f}, {h['ci'][1]:+.3f}]" if "ci" in h else ""
        print(f"  {name}: supported={h['supported']} {extra}")
    for name, h in made["hypotheses_post_hoc_earliest_lap"].items():
        extra = f"{h['estimate']:+.3f} [{h['ci'][0]:+.3f}, {h['ci'][1]:+.3f}]" if "ci" in h else ""
        print(f"  post hoc {name}: {extra}")
    print("label ratio N/(F+Q):", made["label_ratio"]["median"])
    print("same as 6.6 follow-up:", made["same_as_6_6_follow_up"])


# ── figures ─────────────────────────────────────────────────────────────────────────────────


def draw_figures(made: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = {
        "human": "clone: human",
        "e_match": "clone: expert (matched)",
        "e_full": "clone: expert (whole drives)",
        "clone_wide": "clone: expert (wide)",
        "dagger_wide": "DAgger (wide)",
        "dagger_human": "DAgger on human data",
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, arm in enumerate(names):
        for rule, colour, dx in (
            ("training_exam", "#1f77b4", -0.12),
            ("last", "#d62728", 0.12),
        ):
            ys = [made["records"][arm][str(s)]["rules"][rule]["heldout_progress"] for s in SEEDS]
            ax.scatter(
                [i + dx] * len(ys),
                ys,
                s=18,
                color=colour,
                alpha=0.7,
                label=rule if i == 0 else None,
            )
    for key, label in (("hand_neuron", "hand-wired neuron"), ("scripted_expert", "expert")):
        ax.axhline(made["baselines"][key]["heldout_progress"], ls="--", lw=1, color="grey")
        ax.text(
            len(names) - 0.5,
            made["baselines"][key]["heldout_progress"] + 0.01,
            label,
            ha="right",
            fontsize=8,
        )
    ax.set_xticks(range(len(names)), list(names.values()), rotation=20, ha="right")
    ax.set_ylabel("held-out median progress")
    ax.set_title("S5c: held-out progress, one point per seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "S5c_selected.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, group in zip(axes, (("human", "e_match"), ("dagger_wide", "clone_wide")), strict=True):
        for arm in group:
            units = sorted({int(u) for r in made["records"][arm].values() for u in r["curve"]})
            med, lo, hi = [], [], []
            for u in units:
                vals = [
                    r["curve"][str(u)]["heldout_progress"]
                    for r in made["records"][arm].values()
                    if str(u) in r["curve"]
                ]
                s = summarise(vals)
                med.append(s.median), lo.append(s.q1), hi.append(s.q3)
            # a DAgger unit is a round of 30 epochs: draw it at the epoch it ends on
            x = [u * 30 for u in units] if ARMS[arm]["method"] == "dagger" else units
            ax.plot(x, med, label=names[arm])
            ax.fill_between(x, lo, hi, alpha=0.2)
        ax.set_xlabel("epoch (DAgger: round x 30)")
        ax.legend()
    axes[0].set_ylabel("held-out median progress")
    fig.suptitle("S5c: held-out progress of every examined checkpoint (median and IQR over seeds)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "S5c_curves.png", dpi=130)
    plt.close(fig)
    print("figures in", FIGURES_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", action="store_true", help="make the missing runs first")
    parser.add_argument("--figures", action="store_true", help="draw the figures (matplotlib)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated arms to train")
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    if args.train:
        arms = tuple(a for a in args.arms.split(",") if a)
        train(runs_dir, args.workers, arms)
    made = analyse(runs_dir)
    RESULTS_FILE.write_text(json.dumps(made, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_results(made)
    if args.figures:
        draw_figures(made)


if __name__ == "__main__":
    main()
