"""Study S5b: where does a cloned driver fail, and are its crashes in unvisited states?

Run from the repository root:

    uv run python scripts/study_s5b.py --drive                       # drive the shifted starts
    uv run python scripts/study_s5b.py                               # analyse the cached drives
    uv run --with matplotlib python scripts/study_s5b.py --figures   # also draw the figures

Design and hypotheses: docs/studies/S5b.md (written in the task 6.5 Brief, before any drive).
Nothing is trained: the drivers are Study S5a's networks and baselines.

* Drivers (decision D2A), per S5a seed 0-9: *early*, the earliest examined epoch that finishes all
  10 training-loop starts, and *late*, the examined epoch with the lowest validation loss (S5a's
  rules, recomputed from the run folders). References: the scripted expert, the hand-wired
  neuron and the random networks for seeds 0-9 (S5a's baseline runs).
* Starts (decision D3A): the exam's 10 starts on train_loop_01, each moved by one shift
  (delta, theta), delta in LATERAL_SHIFTS metres and theta in HEADING_SHIFTS_DEG, from rest
  (``experiment/exam.py: shifted_starts``): 30 shifts x 10 starts. Each shift's 10 starts are one
  batch, as in the exam, so shift (0, 0) is every driver's exam drive on the loop, exactly.
* A start is *recoverable* when the expert finishes it. At every step the expert also labels the
  state the driver is in (the policy wrapper ``Labelled`` changes no action), which gives the
  on-policy error L_on = (1 / 2T) sum_t sum_k (y_hat_t,k - y*_t,k)^2.
* Novelty d(o), every 3rd step and 1 s before each crash: the distance in the 7 inputs to the
  nearest frame of the pinned dataset; *novel* when d > tau, the 99th percentile of the
  dataset's own other-episode distances (``experiment/dataset_view.py``).

The drives are cached per driver in <runs>/.s5b/ (git-ignored with runs/); the numbers go to
docs/studies/S5b_results.json (the same bytes on every analysis of the same drives), the
figures to docs/studies/figures/ (matplotlib only for that).
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from neurodrive.core.rng import make_rng
from neurodrive.env.car import DT
from neurodrive.env.episode import EpisodeRecording, run_episode
from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.policies import MLPPolicy, Policy
from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig
from neurodrive.experiment.dataset_view import (
    NOVELTY_PERCENTILE,
    nearest_distances,
    other_episode_distances,
)
from neurodrive.experiment.db import run_folders
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, EXAM_V1, shifted_starts
from neurodrive.experiment.metrics import episode_metrics
from neurodrive.experiment.stats import (
    bootstrap_median_ci,
    bootstrap_median_difference,
    percentile,
    summarise,
)
from neurodrive.experiment.store import RunStore
from neurodrive.explain.sensitivity import sensitivity
from neurodrive.learn.supervised.demos import DEFAULT_DATASETS_DIR, DemoDataset, find_dataset

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "docs" / "studies" / "S5b_results.json"
S5A_RESULTS = ROOT / "docs" / "studies" / "S5a_results.json"
FIGURES_DIR = ROOT / "docs" / "studies" / "figures"

# Fixed in the Brief before any drive (D1A, D2A, D3A).
STUDY = "S5b"
SOURCE_STUDY = "S5a"  # the runs whose networks are driven
DATASET = "20261104-084617-expert-default"  # the dataset S5a's networks were trained on
TRACK = "train_loop_01"
LATERAL_SHIFTS = (-4.5, -3.0, -1.5, 0.0, 1.5, 3.0)  # metres, positive = left
HEADING_SHIFTS_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)  # degrees, positive = pointing left
SEEDS = tuple(range(10))
RANDOM_SEEDS = tuple(range(10))
BOOTSTRAP_SEED = 2026
NOVELTY_EVERY = 3  # steps between novelty measurements (0.1 s)
LEAD_STEPS = 30  # "1 s before the crash"
MIN_STEPS = 100  # H3, H4: measured steps of each kind a network needs to be compared
CURVE_BEFORE_S = 5.0  # the figure's window before a crash
CURVE_AFTER_START_S = 15.0  # the figure's window after a start
CACHE_VERSION = 1


def shifts() -> list[tuple[float, float]]:
    """The 30 shifts (delta metres, theta degrees): sideways outer, heading inner. Start index
    = 10 * shift index + exam start."""
    return [(d, h) for d in LATERAL_SHIFTS for h in HEADING_SHIFTS_DEG]


N_STARTS = len(EXAM_V1.start_fractions) * len(EXAM_V1.lateral_offsets)  # 10 per shift
EXAM_SHIFT = shifts().index((0.0, 0.0))


# ── driving ─────────────────────────────────────────────────────────────────────────────────


class Labelled:
    """A policy that, at every step, also records the expert's label for the state the cars are
    in. It returns the wrapped policy's actions unchanged, so the drive is the plain drive."""

    def __init__(self, policy: Policy, expert: ScriptedExpert) -> None:
        self.policy = policy
        self.expert = expert
        self.labels: list[np.ndarray] = []

    def act(self, observations: np.ndarray) -> np.ndarray:
        self.labels.append(self.expert.label())
        return self.policy.act(observations)


def drive_shift(
    track: Track, policy_for, lateral: float, heading_deg: float
) -> tuple[EpisodeRecording, np.ndarray]:
    """One shift's 10 starts as one batch: the recording and the expert's labels (T, N, 2)."""
    world = World(track, WorldConfig.for_track(track, EXAM_V1.target_laps))
    s0, lat, psi = shifted_starts(track.length, lateral, np.radians(heading_deg))
    world.reset(s0=s0, lateral=lat, heading_offset=psi)
    policy = Labelled(policy_for(world), ScriptedExpert(world))
    recording = run_episode(world, policy, policy_ref={"kind": "s5b", "track": track.name})
    return recording, np.array(policy.labels).reshape(-1, world.n, 2)


def on_policy_error(action: np.ndarray, label: np.ndarray) -> float:
    """L_on = (1 / 2T) sum_t sum_k (action_t,k - label_t,k)^2 over T steps of (T, 2) arrays.
    Example: actions (0, 0.5), (0.1, 0.9), (0, 1) against labels (0, 0.4), (0.1, 0.3),
    (-0.2, -0.4) give 2.37 / 6 = 0.395."""
    a, y = np.asarray(action, float), np.asarray(label, float)
    return float(np.mean((a - y) ** 2))


def _policy_factory(job: dict[str, Any]):
    if job["kind"] == "expert":
        return ScriptedExpert
    network = RunStore.open(job["path"]).load_checkpoint(job["unit"])
    return lambda world: MLPPolicy(network)


def drive_driver(job: dict[str, Any]) -> dict[str, np.ndarray]:
    """Every shifted start of one driver: per start its ending, its on-policy error sums and
    its novelty 1 s before a crash; per measured step (every 3rd) its novelty and errors."""
    started = time.perf_counter()
    track = Track.from_json(DEFAULT_TRACKS_DIR / f"{TRACK}.json")
    frames = DemoDataset.load(find_dataset(DEFAULT_DATASETS_DIR, DATASET)).observation
    policy_for = _policy_factory(job)
    out: dict[str, list] = {
        k: []
        for k in (
            "status",
            "steps",
            "progress_m",
            "progress_frac",
            "end_s_m",
            "end_speed",
            "lead_novelty",
            "sum_sq_steer",
            "sum_sq_throttle",
            "n_steps_alive",
            "sample_start",
            "sample_t",
            "sample_obs",
            "sample_sq_steer",
            "sample_sq_throttle",
        )
    }
    lead_obs = []
    for c, (lateral, heading) in enumerate(shifts()):
        rec, labels = drive_shift(track, policy_for, lateral, heading)
        metrics = episode_metrics(rec, track.length)
        alive = rec.alive[: rec.n_steps]
        sq = (rec.action - labels) ** 2
        for k, m in enumerate(metrics):
            start = c * N_STARTS + k
            out["status"].append(m.status)
            out["steps"].append(m.steps)
            out["progress_m"].append(m.progress_m)
            out["progress_frac"].append(m.progress_frac)
            out["end_s_m"].append(m.end_s_m)
            out["end_speed"].append(float(rec.v[m.steps, k]))
            live = alive[:, k]
            out["sum_sq_steer"].append(float(sq[live, k, 0].sum()))
            out["sum_sq_throttle"].append(float(sq[live, k, 1].sum()))
            out["n_steps_alive"].append(int(live.sum()))
            ts = np.arange(0, m.steps, NOVELTY_EVERY)
            ts = ts[live[ts]]
            out["sample_start"].append(np.full(len(ts), start))
            out["sample_t"].append(ts)
            out["sample_obs"].append(rec.observation[ts, k])
            out["sample_sq_steer"].append(sq[ts, k, 0])
            out["sample_sq_throttle"].append(sq[ts, k, 1])
            lead_obs.append(rec.observation[max(m.steps - LEAD_STEPS, 0), k])
    lead = nearest_distances(np.array(lead_obs), frames)
    crashed = np.array(out["status"]) == "crashed"
    obs = np.concatenate(out.pop("sample_obs"))
    arrays = {k: np.concatenate(v) for k, v in out.items() if k.startswith("sample_")}
    arrays |= {k: np.array(v) for k, v in out.items() if not k.startswith("sample_")}
    arrays["lead_novelty"] = np.where(crashed, lead, np.nan)
    arrays["sample_novelty"] = nearest_distances(obs, frames)
    arrays["drive_s"] = np.array(time.perf_counter() - started)
    return arrays


# ── finding the drivers ─────────────────────────────────────────────────────────────────────


def read_card(store: RunStore, unit: int) -> dict[str, Any]:
    return json.loads((store.path / "exams" / f"unit_{unit}.json").read_text(encoding="utf-8"))


def early_and_late(epochs: list[int], val_loss: list[float], lap: list[float]) -> dict[str, int]:
    """S5a's two rules among the examined ``epochs``: early = the earliest epoch with the highest
    training-loop completion ``lap``; late = the lowest validation loss (the earliest of equals).
    Example: losses (0.10, 0.06, 0.04, 0.02) and completions (0, 1, 1, 0) for epochs 1-4 give
    early 2 and late 4."""
    if not epochs or not len(epochs) == len(val_loss) == len(lap):
        raise ValueError("need one loss and one completion per epoch")
    best = max(lap)
    late = min(range(len(epochs)), key=lambda i: (val_loss[i], epochs[i]))
    return {
        "early": min(e for e, t in zip(epochs, lap, strict=True) if t == best),
        "late": epochs[late],
    }


def find_drivers(runs_dir: Path) -> list[dict[str, Any]]:
    """S5a's runs (by the ``study`` field of meta.json) turned into S5b's drivers."""
    found: dict[str, dict[Any, Path]] = {}
    for path in run_folders(runs_dir):
        store = RunStore.open(path)
        meta = store.read_meta()
        if meta.get("study") == SOURCE_STUDY and meta["status"] == "completed":
            seed = store.read_config().get("seed")
            if seed in found.setdefault(meta["method"], {}):
                raise SystemExit(f"two {meta['method']} runs for seed {seed} in study S5a")
            found[meta["method"]][seed] = path
    need = {
        "imitation": set(SEEDS),
        "random_network": set(RANDOM_SEEDS),
        "hand_neuron": {None},
        "scripted_expert": {None},
    }
    for method, seeds in need.items():
        if set(found.get(method, {})) != seeds:
            raise SystemExit(f"S5b needs S5a's {method} runs for {sorted(seeds, key=str)}")
    drivers = []
    for seed in SEEDS:
        path = found["imitation"][seed]
        store = RunStore.open(path)
        lines = store.read_metrics()
        epochs = [u for u in store.exam_units() if u >= 1]
        val = [lines[e - 1]["val_loss"] for e in epochs]
        lap = [read_card(store, e)["training"]["completion_rate"] for e in epochs]
        for kind, unit in early_and_late(epochs, val, lap).items():
            line = lines[unit - 1]
            drivers.append(
                {
                    "id": f"{kind}-s{seed}",
                    "kind": kind,
                    "seed": seed,
                    "path": str(path),
                    "run": path.name,
                    "unit": unit,
                    "val_loss": line["val_loss"],
                    "val_loss_steer": line["val_loss_steer"],
                    "val_loss_throttle": line["val_loss_throttle"],
                }
            )
    for kind, method, unit in (
        ("expert", "scripted_expert", None),
        ("neuron", "hand_neuron", 0),
    ):
        path = found[method][None]
        drivers.append(
            {
                "id": kind,
                "kind": kind,
                "seed": None,
                "path": str(path),
                "run": path.name,
                "unit": unit,
            }
        )
    for seed in RANDOM_SEEDS:
        path = found["random_network"][seed]
        drivers.append(
            {
                "id": f"random-s{seed}",
                "kind": "random",
                "seed": seed,
                "path": str(path),
                "run": path.name,
                "unit": 0,
            }
        )
    return drivers


# ── the cache ───────────────────────────────────────────────────────────────────────────────


def _cache_key(driver: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "run": driver["run"],
        "unit": driver["unit"],
        "track": TRACK,
        "dataset": DATASET,
        "shifts": shifts(),
        "novelty_every": NOVELTY_EVERY,
        "lead_steps": LEAD_STEPS,
    }


def cache_path(runs_dir: Path, driver: dict[str, Any]) -> Path:
    return runs_dir / ".s5b" / f"{driver['id']}.npz"


def _drive_job(job: tuple[dict[str, Any], str]) -> str:
    driver, runs_dir = job
    arrays = drive_driver(driver)
    path = cache_path(Path(runs_dir), driver)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, key=np.array(json.dumps(_cache_key(driver))), **arrays)
    return driver["id"]


def drive_all(runs_dir: Path, drivers: list[dict[str, Any]], workers: int) -> None:
    with ProcessPoolExecutor(workers) as pool:
        for name in pool.map(_drive_job, [(d, str(runs_dir)) for d in drivers]):
            print(f"drove {name}")


def load_drives(runs_dir: Path, driver: dict[str, Any]) -> dict[str, np.ndarray]:
    path = cache_path(runs_dir, driver)
    if not path.is_file():
        raise SystemExit(f"no drives cached for {driver['id']}: run with --drive first")
    with np.load(path) as data:
        if json.loads(str(data["key"])) != json.loads(json.dumps(_cache_key(driver))):
            raise SystemExit(f"the cached drives of {driver['id']} are stale: run with --drive")
        return {k: data[k] for k in data.files if k != "key"}


# ── analysis ────────────────────────────────────────────────────────────────────────────────


def _band(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    s = summarise(values)
    return {"median": s.median, "q1": s.q1, "q3": s.q3, "n": s.n}


def split_errors(novelty: np.ndarray, sq_steer: np.ndarray, sq_throttle: np.ndarray, tau: float):
    """The on-policy error on familiar (d <= tau) and on novel (d > tau) measured steps, each as
    the mean of (sq_steer + sq_throttle) / 2, with their steer and throttle parts and counts.
    Example: errors (0, 0.01), (0, 0.36), (0.04, 1.96) with only the third step novel give
    familiar 0.37 / 4 = 0.0925 and novel 2.0 / 2 = 1.0."""
    out = {}
    for name, mask in (("familiar", novelty <= tau), ("novel", novelty > tau)):
        n = int(mask.sum())
        out[name] = {
            "n": n,
            "error": float(np.mean((sq_steer[mask] + sq_throttle[mask]) / 2)) if n else None,
            "steer": float(np.mean(sq_steer[mask])) if n else None,
            "throttle": float(np.mean(sq_throttle[mask])) if n else None,
        }
    return out


def driver_record(
    driver: dict[str, Any], d: dict[str, np.ndarray], recoverable: np.ndarray, tau: float
) -> dict[str, Any]:
    """What one driver did on the shifted starts (recoverable starts only, unless said)."""
    status = d["status"]
    finished, crashed = status == "finished", status == "crashed"
    cell = np.arange(len(status)) // N_STARTS
    shifted = recoverable & (cell != EXAM_SHIFT)
    exam = cell == EXAM_SHIFT
    by_shift = []
    for c, (lateral, heading) in enumerate(shifts()):
        inside = recoverable & (cell == c)
        n = int(inside.sum())
        by_shift.append(
            {
                "lateral_m": lateral,
                "heading_deg": heading,
                "recoverable": n,
                "finished": int((finished & inside).sum()),
                "crashed": int((crashed & inside).sum()),
                "completion": float(finished[inside].mean()) if n else None,
            }
        )
    crash_rows = [
        {
            "start": int(i),
            "lateral_m": shifts()[i // N_STARTS][0],
            "heading_deg": shifts()[i // N_STARTS][1],
            "exam_start": int(i % N_STARTS),
            "steps": int(d["steps"][i]),
            "end_s_m": float(d["end_s_m"][i]),
            "end_speed": float(d["end_speed"][i]),
            "lead_novelty": float(d["lead_novelty"][i]),
        }
        for i in np.flatnonzero(crashed & recoverable)
    ]
    lead = np.array([r["lead_novelty"] for r in crash_rows])
    keep = recoverable[d["sample_start"]]
    nov = d["sample_novelty"][keep]
    sq_s, sq_t = d["sample_sq_steer"][keep], d["sample_sq_throttle"][keep]
    alive_steps = int(d["n_steps_alive"][recoverable].sum())
    sum_s, sum_t = (
        float(d["sum_sq_steer"][recoverable].sum()),
        float(d["sum_sq_throttle"][recoverable].sum()),
    )
    start_finished = finished[d["sample_start"]][keep]
    return {
        "id": driver["id"],
        "kind": driver["kind"],
        "seed": driver["seed"],
        "run": driver["run"],
        "unit": driver["unit"],
        **{
            k: driver[k] for k in ("val_loss", "val_loss_steer", "val_loss_throttle") if k in driver
        },
        "completion_shifted": float(finished[shifted].mean()),
        "crash_rate_shifted": float(crashed[shifted].mean()),
        "completion_exam_starts": float(finished[exam].mean()),
        "shifted_recoverable_starts": int(shifted.sum()),
        "completion_all_starts": float(finished.mean()),
        # POST HOC (added after the first analysis): starts 5-9 of shift delta are the same poses
        # as starts 0-4 of shift delta + 1.5 m, so central offsets count twice; this counts each
        # recoverable shifted pose once.
        "completion_shifted_distinct_poses": distinct_completion(finished, shifted),
        # POST HOC: how long the cars drove before crashing.
        "crash_times_s": [r["steps"] * DT for r in crash_rows],
        "by_shift": by_shift,
        "crashes": crash_rows,
        "n_crashes": len(crash_rows),
        "share_crashes_from_novel": float(np.mean(lead > tau)) if len(lead) else None,
        "on_policy": {
            "steps": alive_steps,
            "error": (sum_s + sum_t) / (2 * alive_steps),
            "steer": sum_s / alive_steps,
            "throttle": sum_t / alive_steps,
        },
        "split": split_errors(nov, sq_s, sq_t, tau),
        "novel_share_of_steps": {
            "finished_drives": float(np.mean(nov[start_finished] > tau))
            if start_finished.any()
            else None,
            "crashed_drives": float(np.mean(nov[~start_finished] > tau))
            if (~start_finished).any()
            else None,
        },
        "novelty_median": float(np.median(nov)),
        "drive_s": float(d["drive_s"]),
        "env_steps": int(d["steps"].sum()),
        "expert_labels": int(d["n_steps_alive"].sum()),
    }


def novelty_curves(
    drives: list[dict[str, np.ndarray]], recoverable: np.ndarray
) -> dict[str, list[dict[str, float] | None]]:
    """Median and quartiles of the measured novelty against time, pooled over ``drives``, in
    bins of 0.1 s (one measurement): before each crash, counted back from the crash step, and
    after the start, for drives that crash and for drives that finish. Recoverable starts only."""
    return {
        name: [
            _band([v for d in drives for v in _curve_values(d, recoverable, name, b)])
            for b in range(_curve_len(name))
        ]
        for name in ("before_crash", "after_start_crashed", "after_start_finished")
    }


def _curve_len(name: str) -> int:
    window = CURVE_BEFORE_S if name == "before_crash" else CURVE_AFTER_START_S
    return round(window / (NOVELTY_EVERY * DT))


def _curve_values(d: dict[str, np.ndarray], recoverable: np.ndarray, name: str, b: int) -> list:
    """The novelty values in bin ``b`` of curve ``name`` for one driver. Before a crash, bin b
    holds the measurements (n - b) * 0.1 s before it, n bins in all, so the last bin is 0.1 s."""
    start, t, nov, status = d["sample_start"], d["sample_t"], d["sample_novelty"], d["status"]
    keep = recoverable[start]
    if name == "before_crash":
        mask = keep & (status == "crashed")[start]
        before = np.round((d["steps"][start] - t) / NOVELTY_EVERY).astype(int)
        return nov[mask & (before == _curve_len(name) - b)].tolist()
    wanted = "crashed" if name == "after_start_crashed" else "finished"
    mask = keep & (status == wanted)[start]
    return nov[mask & (t == b * NOVELTY_EVERY)].tolist()


def exam_cell_matches_card(driver: dict[str, Any], d: dict[str, np.ndarray]) -> bool:
    """Shift (0, 0) is the exam: every car's status and progress equal its card's, exactly."""
    store = RunStore.open(Path(driver["path"]))
    card = read_card(store, 0 if driver["unit"] is None else driver["unit"])
    (part,) = [t for t in card["tracks"] if t["track"] == TRACK]
    first = EXAM_SHIFT * N_STARTS
    return all(
        d["status"][first + e["start_index"]] == e["status"]
        and float(d["progress_m"][first + e["start_index"]]) == e["progress_m"]
        for e in part["episodes"]
    )


def same_as_s5a(drivers: list[dict[str, Any]]) -> dict[str, Any]:
    """The early and late epochs against S5a's published rules (earliest_lap, min_loss)."""
    if not S5A_RESULTS.is_file():
        return {"found": False}
    s5a = {s["seed"]: s["rules"] for s in json.loads(S5A_RESULTS.read_text("utf-8"))["seeds"]}
    rule = {"early": "earliest_lap", "late": "min_loss"}
    pairs = [
        (d["unit"], s5a[d["seed"]][rule[d["kind"]]]["epoch"]) for d in drivers if d["kind"] in rule
    ]
    return {"found": True, "compared": len(pairs), "equal": sum(a == b for a, b in pairs)}


def hypotheses(records: dict[str, dict[str, Any]], tau: float) -> dict[str, Any]:
    """The pre-registered rules H1-H4 (docs/studies/S5b.md) on the per-driver records."""

    def rng():
        return make_rng(BOOTSTRAP_SEED)

    early = [records[f"early-s{s}"] for s in SEEDS]
    late = [records[f"late-s{s}"] for s in SEEDS]
    d1 = bootstrap_median_difference(
        [r["completion_shifted"] for r in early],
        [r["completion_exam_starts"] for r in early],
        rng(),
        paired=True,
    )
    h1 = {
        "early_completion_shifted": _band([r["completion_shifted"] for r in early]),
        "early_completion_exam_starts": _band([r["completion_exam_starts"] for r in early]),
        "difference": {"estimate": d1.estimate, "ci": [d1.low, d1.high]},
        "supported": d1.estimate < 0 and d1.high < 0,
    }
    h2: dict[str, Any] = {}
    for name, group in (("early", early), ("late", late)):
        shares = [r["share_crashes_from_novel"] for r in group]
        known = [s for s in shares if s is not None]
        ci = bootstrap_median_ci(known, rng()) if known else None
        h2[name] = {
            "shares": shares,
            "seeds_with_crashes": len(known),
            "median": ci.estimate if ci else None,
            "ci": [ci.low, ci.high] if ci else None,
            "supported": bool(ci and ci.estimate > 0.5),
        }
    h2["supported"] = h2["early"]["supported"] and h2["late"]["supported"]

    def comparable(r: dict[str, Any]) -> bool:
        return r["split"]["familiar"]["n"] >= MIN_STEPS and r["split"]["novel"]["n"] >= MIN_STEPS

    qualifying = [r for r in early if comparable(r)]
    h3 = {
        "seeds_compared": [r["seed"] for r in qualifying],
        "ratios_novel_to_familiar": [
            r["split"]["novel"]["error"] / r["split"]["familiar"]["error"] for r in qualifying
        ],
        "supported": bool(qualifying)
        and all(r["split"]["novel"]["error"] > r["split"]["familiar"]["error"] for r in qualifying),
    }
    pairs = [(e, lt) for e, lt in zip(early, late, strict=True) if comparable(e) and comparable(lt)]
    h4: dict[str, Any] = {"seeds_compared": [e["seed"] for e, _ in pairs]}
    for kind in ("familiar", "novel"):
        if len(pairs) < 2:
            h4[kind] = None
            continue
        d = bootstrap_median_difference(
            [lt["split"][kind]["error"] for _, lt in pairs],
            [e["split"][kind]["error"] for e, _ in pairs],
            rng(),
            paired=True,
        )
        h4[kind] = {"estimate": d.estimate, "ci": [d.low, d.high]}
    h4["supported"] = bool(
        h4["familiar"]
        and h4["novel"]
        and h4["familiar"]["estimate"] < 0
        and h4["familiar"]["ci"][1] < 0
        and h4["novel"]["estimate"] > 0
        and h4["novel"]["ci"][0] > 0
    )
    return {"H1": h1, "H2": h2, "H3": h3, "H4": h4}


def pose_of(start: int) -> tuple[int, float, float]:
    """(exam start fraction index, sideways offset, heading) of a start index."""
    lateral, heading = shifts()[start // N_STARTS]
    k = start % N_STARTS
    n_fractions = len(EXAM_V1.start_fractions)
    return k % n_fractions, EXAM_V1.lateral_offsets[k // n_fractions] + lateral, heading


def distinct_completion(finished: np.ndarray, counted: np.ndarray) -> float:
    """The share of the distinct poses among the ``counted`` starts that are finished (twin
    starts of one pose always end alike, since the drive is deterministic)."""
    poses = {pose_of(int(i)): bool(finished[i]) for i in np.flatnonzero(counted)}
    return float(np.mean(list(poses.values())))


def summarise_kind(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Median and IQR over the seeds of one kind of driver (early or late or random)."""

    def band(get) -> dict[str, float] | None:
        return _band([v for r in records if (v := get(r)) is not None])

    by_shift = [
        _band(
            [
                r["by_shift"][c]["completion"]
                for r in records
                if r["by_shift"][c]["completion"] is not None
            ]
        )
        for c in range(len(shifts()))
    ]
    return {
        "completion_shifted": band(lambda r: r["completion_shifted"]),
        "crash_rate_shifted": band(lambda r: r["crash_rate_shifted"]),
        "completion_exam_starts": band(lambda r: r["completion_exam_starts"]),
        "share_crashes_from_novel": band(lambda r: r["share_crashes_from_novel"]),
        "n_crashes": band(lambda r: r["n_crashes"]),
        "on_policy_error": band(lambda r: r["on_policy"]["error"]),
        "on_policy_throttle": band(lambda r: r["on_policy"]["throttle"]),
        "on_policy_steer": band(lambda r: r["on_policy"]["steer"]),
        "val_loss": band(lambda r: r.get("val_loss")),
        "familiar_error": band(lambda r: r["split"]["familiar"]["error"]),
        "novel_error": band(lambda r: r["split"]["novel"]["error"]),
        "familiar_steps": band(lambda r: r["split"]["familiar"]["n"]),
        "novel_steps": band(lambda r: r["split"]["novel"]["n"]),
        "novel_share_finished_drives": band(lambda r: r["novel_share_of_steps"]["finished_drives"]),
        "novel_share_crashed_drives": band(lambda r: r["novel_share_of_steps"]["crashed_drives"]),
        "unit": band(lambda r: r["unit"]),
        "completion_by_shift": by_shift,
        "completion_shifted_distinct_poses": band(lambda r: r["completion_shifted_distinct_poses"]),
        # POST HOC: crash times pooled over the seeds' crashes (not a per-seed median).
        "crash_time_s_pooled": _band([x for r in records for x in r["crash_times_s"]]),
        "share_crashes_within_2_s_pooled": _share(
            [x < 2.0 for r in records for x in r["crash_times_s"]]
        ),
        "share_crashes_after_5_s_pooled": _share(
            [x > 5.0 for r in records for x in r["crash_times_s"]]
        ),
        "lead_novelty_crashes_after_5_s_pooled": _band(
            [c["lead_novelty"] for r in records for c in r["crashes"] if c["steps"] * DT > 5.0]
        ),
    }


def _share(flags: list[bool]) -> float | None:
    return float(np.mean(flags)) if flags else None


def novelty_reference(dataset: DemoDataset) -> tuple[float, dict[str, float]]:
    """tau and the shape of the other-episode distances it comes from."""
    dist = other_episode_distances(dataset.observation, dataset.episode)
    tau = percentile(dist, NOVELTY_PERCENTILE)
    return tau, {p: percentile(dist, float(p)) for p in (50, 90, 99, 100)}


def analyse(runs_dir: Path, inspect: list[dict[str, Any]] = ()) -> dict[str, Any]:
    drivers = find_drivers(runs_dir)
    dataset = DemoDataset.load(find_dataset(DEFAULT_DATASETS_DIR, DATASET))
    tau, distances = novelty_reference(dataset)
    drives = {d["id"]: load_drives(runs_dir, d) for d in drivers}
    recoverable = drives["expert"]["status"] == "finished"
    records = {d["id"]: driver_record(d, drives[d["id"]], recoverable, tau) for d in drivers}
    curves = {
        kind: novelty_curves([drives[f"{kind}-s{s}"] for s in SEEDS], recoverable)
        for kind in ("early", "late")
    }
    curves["expert"] = novelty_curves([drives["expert"]], recoverable)
    checks = {
        "exam_shift_equals_exam_card": {
            d["id"]: exam_cell_matches_card(d, drives[d["id"]]) for d in drivers
        },
        "epochs_equal_s5a_rules": same_as_s5a(drivers),
    }
    checks["every_exam_shift_equal"] = all(checks["exam_shift_equals_exam_card"].values())
    budget = {
        "env_steps_driven": sum(r["env_steps"] for r in records.values()),
        "expert_labels": sum(r["expert_labels"] for r in records.values() if r["kind"] != "expert"),
        "drive_s": sum(r["drive_s"] for r in records.values()),
        "training_env_steps": 0,
        "teacher_queries_for_learning": 0,
    }
    expert = records["expert"]
    return {
        "study": STUDY,
        "design": {
            "source_study": SOURCE_STUDY,
            "dataset": DATASET,
            "track": TRACK,
            "lateral_shifts_m": list(LATERAL_SHIFTS),
            "heading_shifts_deg": list(HEADING_SHIFTS_DEG),
            "starts_per_driver": len(shifts()) * N_STARTS,
            "novelty_every_steps": NOVELTY_EVERY,
            "lead_steps": LEAD_STEPS,
            "min_steps": MIN_STEPS,
            "bootstrap": {"resamples": 10_000, "seed": BOOTSTRAP_SEED},
            "novelty_percentile": NOVELTY_PERCENTILE,
        },
        "novelty": {"tau": tau, "other_episode_distance_percentiles": distances},
        "recoverable": {
            "starts": int(recoverable.sum()),
            "of": len(recoverable),
            "by_shift": [b["recoverable"] for b in expert["by_shift"]],
            "expert_status_counts": {
                s: int(np.sum(drives["expert"]["status"] == s))
                for s in sorted(set(drives["expert"]["status"].tolist()))
            },
        },
        "checks": checks,
        "budget": budget,
        "kinds": {
            kind: summarise_kind([r for r in records.values() if r["kind"] == kind])
            for kind in ("early", "late", "random")
        },
        "drivers": records,
        "curves": curves,
        "hypotheses": hypotheses(records, tau),
        "inspected": [inspect_crash(drivers, spec, dataset, tau) for spec in inspect],
    }


# ── one crash, looked at closely ────────────────────────────────────────────────────────────


def inspect_crash(
    drivers: list[dict[str, Any]], spec: dict[str, Any], dataset: DemoDataset, tau: float
) -> dict[str, Any]:
    """Re-drive one shift of one driver and describe car ``exam_start`` 1 s before its crash:
    its state, the network's inputs and outputs, the expert's label, the sensitivities, and the
    nearest dataset frame. Also the expert's own drive from the same start."""
    (driver,) = [d for d in drivers if d["id"] == spec["driver"]]
    track = Track.from_json(DEFAULT_TRACKS_DIR / f"{TRACK}.json")
    lateral, heading = spec["lateral_m"], spec["heading_deg"]
    rec, labels = drive_shift(track, _policy_factory(driver), lateral, heading)
    k = spec["exam_start"]
    (m,) = [x for i, x in enumerate(episode_metrics(rec, track.length)) if i == k]
    network = RunStore.open(Path(driver["path"])).load_checkpoint(driver["unit"])
    expert_rec, _ = drive_shift(track, ScriptedExpert, lateral, heading)
    expert_m = episode_metrics(expert_rec, track.length)[k]
    plain_rec, _ = drive_shift(track, _policy_factory(driver), 0.0, 0.0)
    plain_m = episode_metrics(plain_rec, track.length)[k]
    moments = {}
    for name, before in (("1 s before", LEAD_STEPS), ("2 s before", 2 * LEAD_STEPS), ("crash", 1)):
        t = max(m.steps - before, 0)
        obs = rec.observation[t, k]
        dist = np.linalg.norm(dataset.observation - obs, axis=1)
        j = int(np.argmin(dist))
        moments[name] = {
            "step": t,
            "time_s": t * DT,
            "x": float(rec.x[t, k]),
            "y": float(rec.y[t, k]),
            "speed": float(rec.v[t, k]),
            "observation": obs.tolist(),
            "action": rec.action[t, k].tolist(),
            "expert_label": labels[t, k].tolist(),
            "novelty": float(dist[j]),
            "novel": bool(dist[j] > tau),
            "sensitivity": sensitivity(network, obs).tolist(),
            "nearest_frame": {
                "row": j,
                "episode": int(dataset.episode[j]),
                "step": int(dataset.step[j]),
                "observation": dataset.observation[j].tolist(),
                "label": dataset.label[j].tolist(),
                "x": float(dataset.x[j]),
                "y": float(dataset.y[j]),
            },
        }
    start = shifts().index((lateral, heading)) * N_STARTS + k
    return {
        **spec,
        "start": start,
        "run": driver["run"],
        "unit": driver["unit"],
        "status": m.status,
        "steps": m.steps,
        "time_s": m.time_s,
        "end_s_m": m.end_s_m,
        "progress_frac": m.progress_frac,
        "start_pose": {
            "s0_m": float(rec.start_s0[k]),
            "lateral_m": float(rec.start_lateral[k]),
            "heading_offset_rad": float(rec.start_heading_offset[k]),
        },
        "moments": moments,
        "expert_from_same_start": {"status": expert_m.status, "time_s": expert_m.time_s},
        "same_network_unshifted": {"status": plain_m.status, "time_s": plain_m.time_s},
        "novelty_last_5_s": [
            float(np.linalg.norm(dataset.observation - rec.observation[t, k], axis=1).min())
            for t in range(max(m.steps - 150, 0), m.steps + 1, NOVELTY_EVERY)
        ],
    }


# ── output ──────────────────────────────────────────────────────────────────────────────────


def _rounded(obj: Any) -> Any:
    """Floats to 6 significant digits, recursively (NaN as None; keys sorted when written)."""
    if isinstance(obj, float | np.floating):
        return None if np.isnan(obj) else float(f"{float(obj):.6g}")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_rounded(v) for v in obj]
    return obj


def _fmt(v: float | None, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def print_tables(a: dict[str, Any]) -> None:
    nov, checks = a["novelty"], a["checks"]
    print(f"tau = {nov['tau']:.4f}; distances {nov['other_episode_distance_percentiles']}")
    r = a["recoverable"]
    print(f"recoverable starts: {r['starts']} of {r['of']}; expert {r['expert_status_counts']}")
    print(f"every (0, 0) cell = its exam card: {checks['every_exam_shift_equal']}")
    print(f"epochs = S5a's rules: {checks['epochs_equal_s5a_rules']}")
    print(f"budget: {a['budget']}")
    print(
        "\nper driver: unit, completion shifted / exam starts, crashes, novel share of crashes,"
        " L_on, familiar / novel error (steps), val loss"
    )
    for rec in a["drivers"].values():
        if rec["kind"] == "random":
            continue
        sp = rec["split"]
        print(
            f"  {rec['id']:<9} unit {str(rec['unit']):>4}  {rec['completion_shifted']:.3f} / "
            f"{rec['completion_exam_starts']:.2f}  crashes {rec['n_crashes']:>3}  novel "
            f"{_fmt(rec['share_crashes_from_novel'], 2)}  L_on {rec['on_policy']['error']:.4f} "
            f"(thr {rec['on_policy']['throttle']:.4f})  fam {_fmt(sp['familiar']['error'], 4)} "
            f"({sp['familiar']['n']})  nov {_fmt(sp['novel']['error'], 4)} ({sp['novel']['n']})  "
            f"val {_fmt(rec.get('val_loss'), 4)}"
        )
    for kind, s in a["kinds"].items():
        cs = s["completion_shifted"]
        print(
            f"{kind}: completion shifted median {cs['median']:.3f} [{cs['q1']:.3f}, {cs['q3']:.3f}]"
        )
    for h, v in a["hypotheses"].items():
        print(f"{h}: {json.dumps(_rounded(v))}")


def candidates(a: dict[str, Any], kind: str = "early") -> None:
    """Crashes of ``kind`` networks from novel states, for choosing the failure to inspect."""
    tau = a["novelty"]["tau"]
    for rec in a["drivers"].values():
        if rec["kind"] != kind:
            continue
        for c in rec["crashes"]:
            if c["lead_novelty"] > tau:
                print(
                    f"{rec['id']} shift ({c['lateral_m']}, {c['heading_deg']}) start "
                    f"{c['exam_start']}: crash at {c['steps']} steps, {c['end_s_m']:.0f} m, "
                    f"{c['end_speed']:.1f} m/s, lead novelty {c['lead_novelty']:.3f}"
                )


# ── figures ─────────────────────────────────────────────────────────────────────────────────

BLUE = "#2a78d6"  # categorical slot 1 of the dataviz reference palette (light): early networks
ORANGE = "#eb6834"  # slot 2: late networks
AQUA = "#1baf7a"  # slot 3: the scripted expert
INK = "#111111"
SEQUENTIAL = (
    "#f7fbff",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)


def figures(a: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tau = a["novelty"]["tau"]
    n_lat, n_head = len(LATERAL_SHIFTS), len(HEADING_SHIFTS_DEG)

    # 1. Completion by shift: one grid per kind of driver. Sideways shift down, heading across.
    ramp = LinearSegmentedColormap.from_list("completion", SEQUENTIAL)
    panels = [
        (
            "scripted expert:\nstarts it finishes (of 10)",
            [b["recoverable"] / 10 for b in a["drivers"]["expert"]["by_shift"]],
            [str(b["recoverable"]) for b in a["drivers"]["expert"]["by_shift"]],
        ),
        (
            "early networks (epochs 1-5):\nshare finished, 10 seeds pooled",
            *_pooled_grid(a, "early"),
        ),
        (
            "late networks (lowest loss):\nshare finished, 10 seeds pooled",
            *_pooled_grid(a, "late"),
        ),
        (
            "hand-wired neuron:\ncompletion",
            *_grid_values(
                [
                    {"median": b["completion"]} if b["completion"] is not None else None
                    for b in a["drivers"]["neuron"]["by_shift"]
                ]
            ),
        ),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    for ax, (title, values, texts) in zip(axes, panels, strict=True):
        grid = np.array([np.nan if v is None else v for v in values], dtype=float).reshape(
            n_lat, n_head
        )
        img = ax.imshow(grid, cmap=ramp, vmin=0, vmax=1, aspect="auto")
        for i in range(n_lat):
            for j in range(n_head):
                v = grid[i, j]
                ax.text(
                    j,
                    i,
                    texts[i * n_head + j],
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#ffffff" if (not np.isnan(v) and v > 0.55) else INK,
                )
        ax.set_xticks(range(n_head), [f"{h:+.0f}°" if h else "0°" for h in HEADING_SHIFTS_DEG])
        ax.set_yticks(range(n_lat), [f"{d:+.1f} m" if d else "0 m" for d in LATERAL_SHIFTS])
        ax.set_xlabel("heading shift (left +)")
        ax.set_title(title, fontsize=10)
        ax.add_patch(
            plt.Rectangle(
                (HEADING_SHIFTS_DEG.index(0.0) - 0.5, LATERAL_SHIFTS.index(0.0) - 0.5),
                1,
                1,
                fill=False,
                ec=ORANGE,
                lw=2,
            )
        )
    axes[0].set_ylabel("sideways shift (left +)")
    fig.colorbar(
        img, ax=axes, fraction=0.02, pad=0.02, label="share of recoverable starts finished"
    )
    fig.suptitle(
        "S5b: the exam's 10 starts on train_loop_01, shifted; outlined cell = shift (0, 0),"
        " the exam's own starts",
        y=1.06,
    )
    fig.savefig(FIGURES_DIR / "s5b_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 2. Novelty against time: before each crash, and after the start.
    step_s = NOVELTY_EVERY * DT
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    ax = axes[0]
    for kind, colour, label in (
        ("early", BLUE, "early networks"),
        ("late", ORANGE, "late networks"),
    ):
        c = a["curves"][kind]["before_crash"]
        x = -np.arange(len(c), 0, -1) * step_s
        _band_line(ax, x, c, colour, label)
    ax.axvline(-LEAD_STEPS * DT, color="#888888", lw=1, ls=":")
    ax.text(
        -LEAD_STEPS * DT,
        0.02,
        " 1 s before",
        color="#555555",
        fontsize=8,
        transform=ax.get_xaxis_transform(),
    )
    ax.set_xlabel("time before the crash (s)")
    ax.set_title("Before each crash from a recoverable start (median, IQR)", fontsize=10)
    ax = axes[1]
    for kind, name, colour, style, label in (
        ("early", "after_start_crashed", BLUE, "-", "early networks, drives that crash"),
        ("early", "after_start_finished", BLUE, "--", "early networks, drives that finish"),
        ("expert", "after_start_finished", AQUA, "-", "scripted expert (finishes all)"),
    ):
        c = a["curves"][kind][name]
        _band_line(ax, np.arange(len(c)) * step_s, c, colour, label, style)
    ax.set_xlabel("time since the start (s)")
    ax.set_title("After a shifted start (median, IQR)", fontsize=10)
    for ax in axes:
        ax.axhline(tau, color=INK, lw=1.2, ls="-.", label=f"τ = {tau:.3f} (novel above)")
        ax.set_yscale("log")
        ax.grid(True, color="#e5e5e5", lw=0.8)
        ax.legend(frameon=True, framealpha=0.9, fontsize=8)
    axes[0].set_ylabel("novelty: distance to the nearest\ndataset frame (log scale)")
    fig.suptitle("S5b: how far the car's state is from everything the expert showed")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s5b_novelty_time.png", dpi=130)
    plt.close(fig)

    # 3. On-policy error on familiar and novel steps, per seed, next to the validation loss.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, kind, colour, title in (
        (axes[0], "early", BLUE, "early networks"),
        (axes[1], "late", ORANGE, "late networks"),
    ):
        for s in SEEDS:
            r = a["drivers"][f"{kind}-s{s}"]
            ys = [r.get("val_loss"), r["split"]["familiar"]["error"], r["split"]["novel"]["error"]]
            xs = [0, 1, 2]
            pts = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None]
            ax.plot(
                *zip(*pts, strict=True),
                color=colour,
                lw=1,
                alpha=0.6,
                marker="o",
                ms=5,
                mec="#ffffff",
                mew=0.8,
            )
        ax.set_xticks(
            [0, 1, 2],
            [
                "validation loss\n(the expert's own\nheld-back drives)",
                "on-policy error,\nfamiliar states\n(d ≤ τ)",
                "on-policy error,\nnovel states\n(d > τ)",
            ],
        )
        ax.set_yscale("log")
        ax.set_title(f"{title}: one line per seed", fontsize=10)
        ax.grid(True, axis="y", color="#e5e5e5", lw=0.8)
    axes[0].set_ylabel("mean squared error\n(steer and throttle, log scale)")
    fig.suptitle(
        "S5b: the error against the expert's label, on the teacher's states and on the"
        " network's own"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s5b_error.png", dpi=130)
    plt.close(fig)


def _grid_values(bands: list[dict[str, float] | None]) -> tuple[list[float | None], list[str]]:
    values = [None if b is None else b["median"] for b in bands]
    return values, ["-" if v is None else f"{v:.2f}" for v in values]


def _pooled_grid(a: dict[str, Any], kind: str) -> tuple[list[float | None], list[str]]:
    """Per shift: finished / recoverable starts summed over the seeds of ``kind``."""
    records = [a["drivers"][f"{kind}-s{s}"] for s in SEEDS]
    values = []
    for c in range(len(shifts())):
        n = sum(r["by_shift"][c]["recoverable"] for r in records)
        values.append(sum(r["by_shift"][c]["finished"] for r in records) / n if n else None)
    return values, ["-" if v is None else f"{v:.2f}" for v in values]


MIN_CURVE_N = 20  # a curve's bin is drawn only when it holds at least this many measurements


def _band_line(ax, x, curve, colour, label, style="-") -> None:
    def get(key: str) -> np.ndarray:
        return np.array([np.nan if p is None or p["n"] < MIN_CURVE_N else p[key] for p in curve])

    med, q1, q3 = get("median"), get("q1"), get("q3")
    ax.fill_between(x, q1, q3, color=colour, alpha=0.13, lw=0)
    ax.plot(x, med, color=colour, lw=2, ls=style, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drive", action="store_true", help="drive every shifted start first")
    parser.add_argument("--figures", action="store_true", help="draw the figures (matplotlib)")
    parser.add_argument(
        "--candidates", action="store_true", help="list early crashes from novel states"
    )
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"), help="folder of run folders")
    parser.add_argument("--workers", type=int, default=6, help="drivers driven at once")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    if args.drive:
        drive_all(runs_dir, find_drivers(runs_dir), args.workers)
    analysis = analyse(runs_dir, INSPECT)
    print_tables(analysis)
    if args.candidates:
        candidates(analysis)
    text = json.dumps(_rounded(analysis), indent=1, sort_keys=True) + "\n"
    RESULTS_FILE.write_text(text, encoding="utf-8", newline="\n")
    print(f"\nwrote {RESULTS_FILE.relative_to(ROOT).as_posix()}")
    if args.figures:
        figures(analysis)
        print(f"wrote figures to {FIGURES_DIR.relative_to(ROOT).as_posix()}")


# The crashes looked at in the failure analysis, chosen after the results (an inspection, not a
# hypothesis): an early network that crashes 1.4 s after a start the expert recovers from, and a
# late network that survives its start and crashes 5 s later at speed (it finishes the same
# exam start unshifted).
INSPECT: list[dict[str, Any]] = [
    {"driver": "early-s0", "lateral_m": -3.0, "heading_deg": -30.0, "exam_start": 0},
    {"driver": "late-s9", "lateral_m": 1.5, "heading_deg": 15.0, "exam_start": 8},
]


if __name__ == "__main__":
    main()
