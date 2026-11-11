"""Task 6.6's verification: does DAgger make the cloned driver more robust?

Written down in the task Brief (2026-11-10) before any DAgger run:

- **V1:** in every one of seeds 0-4, DAgger's final network (round 10, 300 epochs) finishes a
  larger share of Study S5b's recoverable shifted starts than S5a's epoch-300 network of the
  same seed (same dataset, split, starting weights and number of epochs);
- **V2:** the median over the 5 seeds of that share is at least 0.70, the median of S5a's early
  clones in S5b (the most robust clones found);
- **V3:** in every seed, the on-policy error of round 10's collection drive is below round 2's.

Reported as well, not tested: every round's training-loop and held-out exam numbers (never used
to choose anything), the teacher queries and the wall-clock time.

How: ``--train`` trains ``configs/dagger.yaml`` for seeds 0-4 (tag ``check-s<k>``, on S5a's
pinned dataset, one process per seed). ``--drive`` drives, from S5b's 300 shifted starts on
``train_loop_01``, each seed's DAgger network after round 1 and after the last round, and S5a's
epoch-300 network, into the cache ``runs/.dagger_check/`` (git-ignored), reusing
``scripts/study_s5b.py`` (its driving, its cached drives of the expert and of S5a's early and
late networks, and its per-driver record). Without options it only analyses; the results go to
``runs/.dagger_check/results.json`` and are printed. This is a check for the task Report, not a
study: the comparison of DAgger with plain cloning on held-out tracks is Study S5c (6.7).

Reproduce: ``uv run python scripts/check_dagger.py --train --drive`` (about 13 minutes with
5 workers on 8 cores).

**The follow-up (``--wide``, ADR-050; F1A, F2B, F3A chosen by the user).** Written down before
any of its runs: DAgger collecting from wide starts (``configs/dagger_wide.yaml``: ±3 m,
±0.35 rad), and as a control plain cloning on an expert dataset recorded from the same wide
starts (``nd demo-record --random-starts 90 --start-lateral 3 --start-heading 0.35 --tag
wide``; 300 epochs, examined every 30), seeds 0-4, scored on the same 300 shifted starts:

- **W1:** DAgger-wide's median shifted-start completion is at least 0.70, and above 6.6's
  DAgger (``check-s<k>``) in at least 4 of the 5 seeds;
- **W2:** DAgger-wide finishes more shifted starts than the wide clone at epoch 300 in at
  least 4 of the 5 seeds;
- **W3:** DAgger-wide laps the training loop from all 10 exam starts at round 10 in every seed.

Reproduce: record the wide dataset (above), then
``uv run python scripts/check_dagger.py --wide --train --drive``; results in
``runs/.dagger_check/results_wide.json``.
"""

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neurodrive.experiment.config import load_config, parse_config, resolved
from neurodrive.experiment.dataset_view import dataset_view
from neurodrive.experiment.db import run_folders
from neurodrive.experiment.runner import train_dagger_run, train_imitation_run
from neurodrive.experiment.stats import summarise
from neurodrive.experiment.store import RunStore
from neurodrive.learn.supervised.demos import DEFAULT_DATASETS_DIR, DemoDataset, find_dataset

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
CACHE_DIR = RUNS_DIR / ".dagger_check"
CONFIG = ROOT / "configs" / "dagger.yaml"
SEEDS = tuple(range(5))
TAG = "check"
V2_BAR = 0.70  # S5b: the median completion of S5a's early clones on the shifted starts


def _load_s5b():
    spec = importlib.util.spec_from_file_location("study_s5b", ROOT / "scripts" / "study_s5b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s5b = _load_s5b()


# ── training ────────────────────────────────────────────────────────────────────────────────


def check_config(seed: int):
    """configs/dagger.yaml on S5a's dataset (pinned by its id), with seed and tag."""
    data = resolved(load_config(CONFIG))
    data["data"]["dataset"] = s5b.DATASET
    return parse_config(data, seed=seed, tag=f"{TAG}-s{seed}")


def _train_seed(job: tuple[int, str]) -> str:
    seed, sweep_id = job
    found = train_dagger_run(
        check_config(seed),
        RUNS_DIR,
        extra_meta={"sweep": sweep_id},
        datasets_dir=DEFAULT_DATASETS_DIR,
    )
    print(f"seed {seed}: {found.store.path.name} {found.status}, {found.teacher_queries} queries")
    return found.store.path.name


def train(workers: int) -> None:
    sweep_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-dagger-{TAG}"
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_train_seed, [(s, sweep_id) for s in SEEDS]))


# ── finding the networks ────────────────────────────────────────────────────────────────────


def dagger_runs() -> dict[int, Path]:
    """The newest completed DAgger run of each seed with tag check-s<k>."""
    found: dict[int, Path] = {}
    for path in run_folders(RUNS_DIR):
        store = RunStore.open(path)
        meta, config = store.read_meta(), store.read_config()
        if meta.get("method") != "dagger" or meta.get("status") != "completed":
            continue
        seed = config.get("seed")
        if config.get("tag") == f"{TAG}-s{seed}" and seed in SEEDS:
            found[seed] = max(found.get(seed, path), path)
    if set(found) != set(SEEDS):
        raise SystemExit(f"need DAgger runs for seeds {list(SEEDS)}: run with --train first")
    return found


def s5a_runs() -> dict[int, Path]:
    found = {}
    for path in run_folders(RUNS_DIR):
        store = RunStore.open(path)
        meta = store.read_meta()
        if meta.get("study") == "S5a" and meta.get("method") == "imitation":
            found[store.read_config()["seed"]] = path
    return found


def new_drivers() -> list[dict[str, Any]]:
    """Per seed: DAgger after round 1 and after its last round, and S5a's epoch 300."""
    dagger, s5a = dagger_runs(), s5a_runs()
    drivers = []
    for seed in SEEDS:
        last = RunStore.open(dagger[seed]).checkpoint_units()[-1]
        for kind, path, unit in (
            ("dagger_r1", dagger[seed], 1),
            ("dagger", dagger[seed], last),
            ("s5a_e300", s5a[seed], 300),
        ):
            drivers.append(
                {
                    "id": f"{kind}-s{seed}",
                    "kind": kind,
                    "seed": seed,
                    "path": str(path),
                    "run": path.name,
                    "unit": unit,
                }
            )
    return drivers


def cache_path(driver: dict[str, Any]) -> Path:
    return CACHE_DIR / f"{driver['id']}.npz"


def _key(driver: dict[str, Any]) -> str:
    return json.dumps({**s5b._cache_key(driver), "check": 1}, sort_keys=True)


def _drive_job(driver: dict[str, Any]) -> str:
    arrays = s5b.drive_driver(driver)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path(driver), key=np.array(_key(driver)), **arrays)
    return driver["id"]


def drive(workers: int) -> None:
    with ProcessPoolExecutor(workers) as pool:
        for name in pool.map(_drive_job, new_drivers()):
            print(f"drove {name}")


def load_new(driver: dict[str, Any]) -> dict[str, np.ndarray]:
    path = cache_path(driver)
    if not path.is_file():
        raise SystemExit(f"no drives cached for {driver['id']}: run with --drive first")
    with np.load(path) as data:
        if str(data["key"]) != _key(driver):
            raise SystemExit(f"the cached drives of {driver['id']} are stale: run with --drive")
        return {k: data[k] for k in data.files if k != "key"}


# ── analysis ────────────────────────────────────────────────────────────────────────────────


def round_table(path: Path) -> list[dict[str, Any]]:
    """Per round of a DAgger run: losses, the collection drive and both exam parts."""
    store = RunStore.open(path)
    rows = []
    for m in store.read_metrics():
        card = s5b.read_card(store, m["unit"])
        rows.append(
            {
                "round": m["unit"],
                "val_loss": m["val_loss"],
                "loss": m["loss"],
                "onpolicy_loss": m["onpolicy_loss"],
                "onpolicy_loss_throttle": m["onpolicy_loss_throttle"],
                "collect_frames": m["collect_frames"],
                "collect_finished": m["collect_finished"],
                "train_frames": m["train_frames"],
                "teacher_queries_total": m["teacher_queries_total"],
                "env_steps_total": m["env_steps_total"],
                "wall_clock_s": m["wall_clock_s"],
                "training_completion": card["training"]["completion_rate"],
                "heldout_completion": card["heldout"]["completion_rate"],
                "heldout_progress": card["heldout"]["progress_frac_median"],
            }
        )
    return rows


def analyse() -> dict[str, Any]:
    old = {d["id"]: d for d in s5b.find_drivers(RUNS_DIR)}
    expert = s5b.load_drives(RUNS_DIR, old["expert"])
    recoverable = expert["status"] == "finished"
    dataset = DemoDataset.load(find_dataset(DEFAULT_DATASETS_DIR, s5b.DATASET))
    tau, _ = s5b.novelty_reference(dataset)
    records: dict[str, dict[str, Any]] = {}
    for driver in new_drivers():
        records[driver["id"]] = s5b.driver_record(driver, load_new(driver), recoverable, tau)
    for seed in SEEDS:
        for kind in ("early", "late"):
            d = old[f"{kind}-s{seed}"]
            records[d["id"]] = s5b.driver_record(d, s5b.load_drives(RUNS_DIR, d), recoverable, tau)
    runs = dagger_runs()
    s5a = s5a_runs()
    same_as_s5a_epoch_30 = {
        seed: RunStore.open(runs[seed])
        .load_checkpoint(1)
        .equals(RunStore.open(s5a[seed]).load_checkpoint(30))
        for seed in SEEDS
    }
    rounds = {seed: round_table(runs[seed]) for seed in SEEDS}

    def share(kind: str, seed: int) -> float:
        return records[f"{kind}-s{seed}"]["completion_shifted"]

    v1 = {seed: share("dagger", seed) > share("s5a_e300", seed) for seed in SEEDS}
    dagger_median = summarise([share("dagger", s) for s in SEEDS]).median
    v3 = {
        seed: rounds[seed][-1]["onpolicy_loss"] < rounds[seed][1]["onpolicy_loss"] for seed in SEEDS
    }
    kinds = ("dagger", "dagger_r1", "s5a_e300", "early", "late")
    summary = {
        kind: {
            name: s5b._band([records[f"{kind}-s{s}"][name] for s in SEEDS])
            for name in ("completion_shifted", "crash_rate_shifted", "completion_exam_starts")
        }
        | {
            "on_policy_error": s5b._band(
                [records[f"{kind}-s{s}"]["on_policy"]["error"] for s in SEEDS]
            )
        }
        for kind in kinds
    }
    return {
        "seeds": list(SEEDS),
        "runs": {seed: runs[seed].name for seed in SEEDS},
        "tau": tau,
        "recoverable_starts": int(recoverable.sum()),
        "round1_equals_s5a_epoch30": same_as_s5a_epoch_30,
        "V1": {"per_seed": v1, "holds": all(v1.values())},
        "V2": {"median": dagger_median, "bar": V2_BAR, "holds": dagger_median >= V2_BAR},
        "V3": {"per_seed": v3, "holds": all(v3.values())},
        "summary": summary,
        "per_seed": {
            seed: {
                kind: {
                    k: records[f"{kind}-s{seed}"][k]
                    for k in (
                        "unit",
                        "completion_shifted",
                        "crash_rate_shifted",
                        "completion_exam_starts",
                        "on_policy",
                        "split",
                    )
                }
                for kind in kinds
            }
            for seed in SEEDS
        },
        "rounds": rounds,
        "by_shift": {kind: [records[f"{kind}-s{s}"]["by_shift"] for s in SEEDS] for kind in kinds},
    }


def _fmt(v: float | None, digits: int = 3) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def print_results(a: dict[str, Any]) -> None:
    print(f"recoverable starts: {a['recoverable_starts']} of 300; tau {a['tau']:.4f}")
    print(f"round 1 equals S5a's epoch 30: {a['round1_equals_s5a_epoch30']}")
    print("\nshifted-start completion (recoverable, exam shift left out) per seed:")
    print("seed  dagger(unit)  dagger_r1  s5a_e300  s5a early  s5a late")
    for seed in a["seeds"]:
        p = a["per_seed"][seed]
        print(
            f"{seed:>4}  {_fmt(p['dagger']['completion_shifted'])} ({p['dagger']['unit']:>2})"
            f"     {_fmt(p['dagger_r1']['completion_shifted'])}      "
            f"{_fmt(p['s5a_e300']['completion_shifted'])}     "
            f"{_fmt(p['early']['completion_shifted'])}      {_fmt(p['late']['completion_shifted'])}"
        )
    print("\nmedians [IQR] over the seeds:")
    for kind, s in a["summary"].items():
        line = "  ".join(
            f"{name} {_fmt(b['median'])} [{_fmt(b['q1'])}, {_fmt(b['q3'])}]"
            for name, b in s.items()
            if b is not None
        )
        print(f"  {kind:<10} {line}")
    print("\nper round (seed: on-policy error / training-loop completion / held-out completion):")
    for seed in a["seeds"]:
        r = a["rounds"][seed]
        print(
            f"  seed {seed}: "
            + "  ".join(
                f"{x['round']}:{_fmt(x['onpolicy_loss'])}/{_fmt(x['training_completion'], 1)}"
                f"/{_fmt(x['heldout_completion'], 2)}"
                for x in r
            )
        )
        last = r[-1]
        print(
            f"          teacher queries {last['teacher_queries_total']}, training frames "
            f"{last['train_frames']}, val loss {_fmt(last['val_loss'], 4)}, "
            f"training clock {last['wall_clock_s']:.0f} s"
        )
    for name in ("V1", "V2", "V3"):
        print(f"{name}: {'holds' if a[name]['holds'] else 'does not hold'}  {a[name]}")


# ── the follow-up: wide collection starts and the wide clone (ADR-050) ─────────────────────

WIDE_CONFIG = ROOT / "configs" / "dagger_wide.yaml"
CLONE_CONFIG = ROOT / "configs" / "imitation.yaml"
WIDE_TAG = "checkwide"
WIDE_DATASET = "expert-wide"  # the newest expert dataset tagged wide
CLONE_EPOCHS = 300
CLONE_EVERY = 30
W1_BAR = 0.70
MOST = 4  # "in at least 4 of the 5 seeds"


def wide_config(seed: int):
    data = resolved(load_config(WIDE_CONFIG))
    data["data"]["dataset"] = s5b.DATASET  # the same demonstrations as 6.6's DAgger
    return parse_config(data, seed=seed, tag=f"{WIDE_TAG}-s{seed}")


def clone_config(seed: int):
    data = resolved(load_config(CLONE_CONFIG, epochs=CLONE_EPOCHS))
    data["data"]["dataset"] = find_dataset(DEFAULT_DATASETS_DIR, WIDE_DATASET).name
    data["checkpoint_every"] = CLONE_EVERY
    return parse_config(data, seed=seed, tag=f"{WIDE_TAG}-bc-s{seed}")


def _train_wide(job: tuple[str, int, str]) -> str:
    kind, seed, sweep_id = job
    meta = {"sweep": sweep_id}
    if kind == "dagger":
        found = train_dagger_run(
            wide_config(seed), RUNS_DIR, extra_meta=meta, datasets_dir=DEFAULT_DATASETS_DIR
        )
    else:
        found = train_imitation_run(
            clone_config(seed), RUNS_DIR, extra_meta=meta, datasets_dir=DEFAULT_DATASETS_DIR
        )
    print(f"{kind} seed {seed}: {found.store.path.name} {found.status}")
    return found.store.path.name


def train_wide(workers: int) -> None:
    find_dataset(DEFAULT_DATASETS_DIR, WIDE_DATASET)  # recorded first, or stop here
    when = f"{datetime.now(UTC):%Y%m%d-%H%M%S}"
    jobs = [("dagger", s, f"{when}-dagger-{WIDE_TAG}") for s in SEEDS]
    jobs += [("clone", s, f"{when}-imitation-{WIDE_TAG}-bc") for s in SEEDS]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_train_wide, jobs))


def tagged_runs(method: str, tag: str) -> dict[int, Path]:
    """The newest completed run of each seed with ``method`` and tag ``<tag>-s<k>``."""
    found: dict[int, Path] = {}
    for path in run_folders(RUNS_DIR):
        store = RunStore.open(path)
        meta, config = store.read_meta(), store.read_config()
        seed = config.get("seed")
        if (
            meta.get("method") == method
            and meta.get("status") == "completed"
            and config.get("tag") == f"{tag}-s{seed}"
            and seed in SEEDS
        ):
            found[seed] = max(found.get(seed, path), path)
    if set(found) != set(SEEDS):
        raise SystemExit(f"need {method} runs tagged {tag}-s<k> for seeds {list(SEEDS)}")
    return found


def wide_drivers() -> list[dict[str, Any]]:
    dagger, clone = tagged_runs("dagger", WIDE_TAG), tagged_runs("imitation", f"{WIDE_TAG}-bc")
    drivers = []
    for seed in SEEDS:
        last = RunStore.open(dagger[seed]).checkpoint_units()[-1]
        for kind, path, unit in (
            ("dagger_wide", dagger[seed], last),
            ("clone_wide", clone[seed], CLONE_EPOCHS),
        ):
            drivers.append(
                {
                    "id": f"{kind}-s{seed}",
                    "kind": kind,
                    "seed": seed,
                    "path": str(path),
                    "run": path.name,
                    "unit": unit,
                }
            )
    return drivers


def drive_wide(workers: int) -> None:
    with ProcessPoolExecutor(workers) as pool:
        for name in pool.map(_drive_job, wide_drivers()):
            print(f"drove {name}")


def coverage(dataset: DemoDataset) -> dict[str, dict[str, float]]:
    """Each observation's covered range c and dense coverage d (the Datasets tab's numbers)."""
    view = dataset_view(dataset)
    return {h.name: {"covered": h.covered, "dense": h.dense} for h in view.observations}


def analyse_wide() -> dict[str, Any]:
    old = {d["id"]: d for d in s5b.find_drivers(RUNS_DIR)}
    recoverable = s5b.load_drives(RUNS_DIR, old["expert"])["status"] == "finished"
    base = DemoDataset.load(find_dataset(DEFAULT_DATASETS_DIR, s5b.DATASET))
    tau, _ = s5b.novelty_reference(base)
    records = {}
    for driver in [*wide_drivers(), *new_drivers()]:
        records[driver["id"]] = s5b.driver_record(driver, load_new(driver), recoverable, tau)
    for seed in SEEDS:
        d = old[f"early-s{seed}"]
        records[d["id"]] = s5b.driver_record(d, s5b.load_drives(RUNS_DIR, d), recoverable, tau)
    dagger_wide = tagged_runs("dagger", WIDE_TAG)
    clone_wide = tagged_runs("imitation", f"{WIDE_TAG}-bc")

    def share(kind: str, seed: int) -> float:
        return records[f"{kind}-s{seed}"]["completion_shifted"]

    beats_dagger = {s: share("dagger_wide", s) > share("dagger", s) for s in SEEDS}
    beats_clone = {s: share("dagger_wide", s) > share("clone_wide", s) for s in SEEDS}
    median = summarise([share("dagger_wide", s) for s in SEEDS]).median
    laps = {
        s: s5b.read_card(RunStore.open(p), RunStore.open(p).checkpoint_units()[-1])["training"][
            "completion_rate"
        ]
        for s, p in dagger_wide.items()
    }
    kinds = ("dagger_wide", "clone_wide", "dagger", "s5a_e300", "early")
    summary = {
        kind: {
            name: s5b._band([records[f"{kind}-s{s}"][name] for s in SEEDS])
            for name in ("completion_shifted", "crash_rate_shifted", "completion_exam_starts")
        }
        | {
            "on_policy_error": s5b._band(
                [records[f"{kind}-s{s}"]["on_policy"]["error"] for s in SEEDS]
            )
        }
        for kind in kinds
    }
    clone_cards = {
        s: {
            u: s5b.read_card(RunStore.open(p), u)["training"]["completion_rate"]
            for u in RunStore.open(p).exam_units()
        }
        for s, p in clone_wide.items()
    }
    wide_data = DemoDataset.load(find_dataset(DEFAULT_DATASETS_DIR, WIDE_DATASET))
    dagger_sets = {
        s: RunStore.open(p).read_meta().get("dagger_dataset") for s, p in dagger_wide.items()
    }
    sets = {
        "expert_default": coverage(base),
        "expert_wide": coverage(wide_data),
        "dagger_wide_s0": coverage(DemoDataset.load(DEFAULT_DATASETS_DIR / dagger_sets[0])),
        "dagger_s0": coverage(
            DemoDataset.load(
                DEFAULT_DATASETS_DIR / RunStore.open(dagger_runs()[0]).read_meta()["dagger_dataset"]
            )
        ),
    }
    return {
        "seeds": list(SEEDS),
        "runs": {
            "dagger_wide": {s: p.name for s, p in dagger_wide.items()},
            "clone_wide": {s: p.name for s, p in clone_wide.items()},
        },
        "wide_dataset": {"id": wide_data.meta["dataset_id"], "frames": wide_data.n_frames},
        "dagger_wide_frames": {
            s: RunStore.open(p).read_metrics()[-1]["teacher_queries_total"]
            for s, p in dagger_wide.items()
        },
        "W1": {
            "median": median,
            "bar": W1_BAR,
            "beats_dagger": beats_dagger,
            "holds": median >= W1_BAR and sum(beats_dagger.values()) >= MOST,
        },
        "W2": {"beats_clone": beats_clone, "holds": sum(beats_clone.values()) >= MOST},
        "W3": {"training_completion": laps, "holds": all(v == 1.0 for v in laps.values())},
        "summary": summary,
        "per_seed": {
            s: {k: records[f"{k}-s{s}"]["completion_shifted"] for k in kinds} for s in SEEDS
        },
        "clone_training_completion_by_epoch": clone_cards,
        "rounds": {s: round_table(p) for s, p in dagger_wide.items()},
        "by_shift": {k: [records[f"{k}-s{s}"]["by_shift"] for s in SEEDS] for k in kinds},
        "coverage": sets,
    }


def print_wide(a: dict[str, Any]) -> None:
    print(f"wide expert dataset {a['wide_dataset']}; DAgger-wide queries {a['dagger_wide_frames']}")
    print("\nshifted-start completion per seed:")
    print("seed  " + "  ".join(f"{k:>11}" for k in a["summary"]))
    for s in a["seeds"]:
        print(f"{s:>4}  " + "  ".join(f"{_fmt(v):>11}" for v in a["per_seed"][s].values()))
    print("\nmedians [IQR] over the seeds:")
    for kind, s in a["summary"].items():
        line = "  ".join(
            f"{name} {_fmt(b['median'])} [{_fmt(b['q1'])}, {_fmt(b['q3'])}]"
            for name, b in s.items()
            if b is not None
        )
        print(f"  {kind:<12} {line}")
    print("\nper round of DAgger-wide (on-policy error / training-loop / held-out completion):")
    for s, rows in a["rounds"].items():
        print(
            f"  seed {s}: "
            + "  ".join(
                f"{x['round']}:{_fmt(x['onpolicy_loss'])}/{_fmt(x['training_completion'], 1)}"
                f"/{_fmt(x['heldout_completion'], 2)}"
                for x in rows
            )
        )
    print("\nwide clone, training-loop completion by examined epoch:")
    for s, cards in a["clone_training_completion_by_epoch"].items():
        print(f"  seed {s}: " + " ".join(f"{u}:{v:.1f}" for u, v in cards.items()))
    print("\nheading-error coverage (covered / dense):")
    for name, cov in a["coverage"].items():
        h = cov["heading_error"]
        print(f"  {name:<15} {h['covered']:.3f} / {h['dense']:.3f}")
    for name in ("W1", "W2", "W3"):
        print(f"{name}: {'holds' if a[name]['holds'] else 'does not hold'}  {a[name]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", action="store_true", help="train seeds 0-4 first")
    parser.add_argument("--drive", action="store_true", help="drive the shifted starts first")
    parser.add_argument("--wide", action="store_true", help="the follow-up (ADR-050) instead")
    parser.add_argument("--workers", type=int, default=5, help="processes at once")
    args = parser.parse_args()
    if args.wide:
        if args.train:
            train_wide(args.workers)
        if args.drive:
            drive_wide(args.workers)
        results = analyse_wide()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        text = json.dumps(results, indent=1, sort_keys=True, default=str)
        (CACHE_DIR / "results_wide.json").write_text(text + "\n", encoding="utf-8")
        print_wide(results)
        return
    if args.train:
        train(args.workers)
    if args.drive:
        drive(args.workers)
    results = analyse()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    text = json.dumps(results, indent=1, sort_keys=True, default=str)
    (CACHE_DIR / "results.json").write_text(text + "\n", encoding="utf-8")
    print_results(results)


if __name__ == "__main__":
    main()
