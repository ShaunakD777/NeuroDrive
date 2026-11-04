"""Study S5a: does a lower imitation loss mean better driving?

Run from the repository root:

    uv run python scripts/study_s5a.py --train                       # make the runs (~25 min)
    uv run python scripts/study_s5a.py                               # analyse them (numbers)
    uv run --with matplotlib python scripts/study_s5a.py --figures   # also draw the figures

Design and hypotheses: docs/studies/S5a.md (written in the task 6.4 Brief, before any run).

* Imitation (configs/imitation.yaml: [7, 16, 16, 2] tanh, Adam 0.003, batches of 256, 20 % of
  the episodes held back) on the 6.1a expert dataset, pinned by its id, for seeds 0-9, 300 epochs
  each, checkpointed and examined every 3rd epoch and on the usual schedule (decision D2B). One
  sweep; every run has study S5a.
* The baselines, made again for S5a (ADR-043): random network seeds 0-9, hand-wired neuron,
  scripted expert (the teacher). The exam is deterministic, so their cards equal task 5.6's.

Measured, for seed k and examined epoch e: V_k(e), the validation loss; P_k(e), the held-out
median progress; C_k(e), the held-out completion rate; T_k(e), the training loop's completion
rate. Per seed: the pairwise agreement A_k of V with P and with T (``experiment/stats.py``), and
the held-out scores of three checkpoint rules that never look at a held-out track: the last
epoch, the lowest validation loss, and the best training-loop exam (ties: the lowest validation
loss, decision D3A). Over the seeds: median, IQR and bootstrap intervals (10 000 resamples,
seed 2026). Numbers go to docs/studies/S5a_results.json (the same bytes on every analysis of the
same runs); with --figures the pictures go to docs/studies/figures/ (matplotlib only for that).
"""

import argparse
import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neurodrive.core.rng import make_rng
from neurodrive.experiment.baselines import make_baselines
from neurodrive.experiment.config import load_config, parse_config, resolved
from neurodrive.experiment.db import DB_NAME, connect, run_folders, runs_of_study
from neurodrive.experiment.runner import train_imitation_run
from neurodrive.experiment.stats import (
    bootstrap_median_ci,
    bootstrap_median_difference,
    pairwise_agreement,
    summarise,
)
from neurodrive.experiment.store import RunStore
from neurodrive.learn.supervised.demos import DEFAULT_DATASETS_DIR

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "docs" / "studies" / "S5a_results.json"
FIGURES_DIR = ROOT / "docs" / "studies" / "figures"
CONFIG = ROOT / "configs" / "imitation.yaml"

# Fixed in the Brief before any run (D1A, D2B, D3A).
STUDY = "S5a"
TAG = "s5a"
DATASET = "20261104-084617-expert-default"  # the 6.1a expert dataset, pinned by its id
DATASETS_DIR = DEFAULT_DATASETS_DIR
SEEDS = tuple(range(10))
EPOCHS = 300
CHECKPOINT_EVERY = 3
RANDOM_SEEDS = tuple(range(10))
BOOTSTRAP_SEED = 2026
EARLY = 10  # H3: some epoch up to this one laps the training loop from all 10 starts
LATE_SHARE = 0.8  # H1: the lowest validation loss is at or after this share of the run
H1_FROM = 30  # H1: V(EPOCHS) < V(H1_FROM)
TASK_6_3_RUN = "20261104-150909-imitation-default"  # seed 0, 100 epochs, every epoch examined


# ── training ────────────────────────────────────────────────────────────────────────────────


def study_config(seed: int):
    """configs/imitation.yaml with S5a's dataset, length, checkpoints, tag and ``seed``."""
    data = resolved(load_config(CONFIG, tag=TAG, epochs=EPOCHS))
    data["data"]["dataset"] = DATASET
    data["checkpoint_every"] = CHECKPOINT_EVERY
    return parse_config(data, seed=seed, tag=f"{TAG}-s{seed}")


def _train_seed(job: tuple[int, str, str]) -> str:
    seed, sweep_id, runs_dir = job
    meta = {"sweep": sweep_id, "study": STUDY}
    found = train_imitation_run(
        study_config(seed), runs_dir, extra_meta=meta, datasets_dir=DATASETS_DIR
    )
    return found.store.path.name


def _make_baselines(runs_dir: str) -> list[str]:
    return [r.store.path.name for r in make_baselines(runs_dir, seeds=RANDOM_SEEDS, study=STUDY)]


def train(runs_dir: Path, workers: int) -> list[str]:
    """Make S5a's runs: the baselines in one worker, each imitation seed in another.

    Every run depends only on its own seed, so running them side by side changes no number
    (only the wall-clock times). The database mirror takes one writer at a time (WAL, 5 s wait).
    """
    sweep_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-imitation-{TAG}"
    with ProcessPoolExecutor(workers) as pool:
        baselines = pool.submit(_make_baselines, str(runs_dir))
        seeds = list(pool.map(_train_seed, [(s, sweep_id, str(runs_dir)) for s in SEEDS]))
        return [*seeds, *baselines.result()]


# ── reading the runs ────────────────────────────────────────────────────────────────────────


def read_card(store: RunStore, unit: int) -> dict[str, Any]:
    return json.loads((store.path / "exams" / f"unit_{unit}.json").read_text(encoding="utf-8"))


def measures(card: dict[str, Any]) -> dict[str, float]:
    """The numbers S5a reads from one exam card (held-out = the 40 held-out episodes pooled)."""
    t, h = card["training"], card["heldout"]
    out = {
        "heldout_progress": h["progress_frac_median"],
        "heldout_completion": h["completion_rate"],
        "heldout_crash": h["crash_rate"],
        "training_progress": t["progress_frac_median"],
        "training_completion": t["completion_rate"],
        "training_crash": t["crash_rate"],
    }
    for part in card["tracks"]:
        if part["split"] == "train":
            out["training_speed"] = float(np.median([e["mean_speed"] for e in part["episodes"]]))
        else:
            out[f"progress:{part['track']}"] = part["aggregate"]["progress_frac_median"]
            out[f"completion:{part['track']}"] = part["aggregate"]["completion_rate"]
    return out


def find_runs(runs_dir: Path) -> dict[str, Any]:
    """S5a's run folders, found by the ``study`` field of their meta.json (the master copy).

    Returns {"imitation": {seed: path}, "random_network": {seed: path}, "hand_neuron": path,
    "scripted_expert": path, "all": every S5a folder}. Stops with a message if a run is missing,
    unfinished or repeated.
    """
    found: dict[str, dict[int | None, list[Path]]] = {}
    every: list[Path] = []
    for path in run_folders(runs_dir):
        store = RunStore.open(path)
        meta = store.read_meta()
        if meta.get("study") != STUDY:
            continue
        every.append(path)
        if meta["status"] != "completed":
            raise SystemExit(f"{path.name} is {meta['status']}; finish it (nd resume) first")
        seed = store.read_config().get("seed")
        found.setdefault(meta["method"], {}).setdefault(seed, []).append(path)
    expected = {
        "imitation": set(SEEDS),
        "random_network": set(RANDOM_SEEDS),
        "hand_neuron": {None},
        "scripted_expert": {None},
    }
    for method, wanted in expected.items():
        by_seed = found.get(method, {})
        if set(by_seed) != wanted or any(len(p) > 1 for p in by_seed.values()):
            raise SystemExit(
                f"S5a needs exactly one {method} run for each of {sorted(wanted, key=str)}"
            )
    return {
        "imitation": {s: found["imitation"][s][0] for s in SEEDS},
        "random_network": {s: found["random_network"][s][0] for s in RANDOM_SEEDS},
        "hand_neuron": found["hand_neuron"][None][0],
        "scripted_expert": found["scripted_expert"][None][0],
        "all": every,
    }


def checkpoint_rules(epochs: list[int], loss: list[float], lap: list[float]) -> dict[str, int]:
    """The epoch each checkpoint rule picks among the examined ``epochs`` (none looks at a
    held-out track).

    ``loss[i]`` is the validation loss and ``lap[i]`` the training loop's completion rate of
    ``epochs[i]``. last: the last epoch. min_loss: the lowest loss (the earliest of equals).
    training_exam: the highest training-loop completion; among those the lowest loss (D3A),
    then the earliest. Example: losses (0.10, 0.06, 0.04, 0.02, 0.01) and completions
    (1, 1, 1, 0, 0) for epochs 1-5 give last 5, min_loss 5, training_exam 3.
    """
    if not epochs or not len(epochs) == len(loss) == len(lap):
        raise ValueError("need one loss and one completion per epoch")
    order = range(len(epochs))
    return {
        "last": max(epochs),
        "min_loss": epochs[min(order, key=lambda i: (loss[i], epochs[i]))],
        "training_exam": epochs[min(order, key=lambda i: (-lap[i], loss[i], epochs[i]))],
    }


def earliest_lap_rule(epochs: list[int], lap: list[float]) -> int:
    """POST HOC (added after the results, not pre-registered): the earliest examined epoch with
    the highest training-loop completion, i.e. decision D3's option B. Still blind to the
    held-out tracks. Example: completions (0, 1, 1, 0) for epochs 1-4 give epoch 2."""
    best = max(lap)
    return min(e for e, t in zip(epochs, lap, strict=True) if t == best)


def _ranks(values: list[float]) -> np.ndarray:
    """Ranks 1..n, equal values sharing the mean of their ranks."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size)
    ranks[order] = np.arange(1, v.size + 1)
    for value in np.unique(v):
        same = v == value
        ranks[same] = ranks[same].mean()
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    """Spearman's rank correlation (the Pearson correlation of the ranks); None if a side is
    constant. Reported next to the pairwise agreement for comparison only."""
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def seed_record(path: Path) -> dict[str, Any]:
    """What one imitation run did: losses every epoch, exam measures at every examined epoch,
    the agreements and the checkpoint rules' choices."""
    store = RunStore.open(path)
    lines = store.read_metrics()
    meta = store.read_meta()
    cards = {u: measures(read_card(store, u)) for u in store.exam_units()}
    epochs = [u for u in sorted(cards) if u >= 1]
    val = [lines[e - 1]["val_loss"] for e in epochs]
    held = [cards[e]["heldout_progress"] for e in epochs]
    lap = [cards[e]["training_completion"] for e in epochs]
    laps_full = [e for e, t in zip(epochs, lap, strict=True) if t == 1.0]
    first_lap = laps_full[0] if laps_full else None
    lost = [e for e, t in zip(epochs, lap, strict=True) if first_lap and e > first_lap and t < 1.0]
    rules = checkpoint_rules(epochs, val, lap)
    rules["earliest_lap"] = earliest_lap_rule(epochs, lap)  # post hoc

    def share_full(lo: int, hi: int) -> float:
        inside = [t for e, t in zip(epochs, lap, strict=True) if lo <= e <= hi]
        return sum(t == 1.0 for t in inside) / len(inside)

    def agreement(score: list[float]) -> dict[str, Any]:
        a = pairwise_agreement(val, score)
        return {
            "share": a.share,
            "agree": a.agree,
            "disagree": a.disagree,
            "left_out": a.left_out,
            "spearman": spearman(val, score),
        }

    thirds = [(1, EPOCHS // 3), (EPOCHS // 3 + 1, 2 * EPOCHS // 3), (2 * EPOCHS // 3 + 1, EPOCHS)]
    return {
        "run_id": path.name,
        "seed": store.read_config()["seed"],
        "val_episodes": meta["dataset"]["val_episodes"],
        "wall_clock_s": lines[-1]["wall_clock_s"],
        "exam_wall_clock_s": meta["exam_wall_clock_s"],
        "epochs": epochs,
        "val_loss_every_epoch": [line["val_loss"] for line in lines],
        "loss_every_epoch": [line["loss"] for line in lines],
        "loss_parts": {
            str(e): {
                k: lines[e - 1][k]
                for k in ("loss_steer", "loss_throttle", "val_loss_steer", "val_loss_throttle")
            }
            for e in sorted({1, H1_FROM, min(100, EPOCHS), EPOCHS})
        },
        "cards": cards,
        "agreement_heldout_progress": agreement(held),
        "agreement_training_completion": agreement(lap),
        "min_val_loss_epoch_all": 1 + int(np.argmin([line["val_loss"] for line in lines])),
        "first_full_lap_epoch": first_lap,
        "first_lost_lap_epoch": lost[0] if lost else None,
        "last_full_lap_epoch": laps_full[-1] if laps_full else None,
        "full_lap_share_by_third": {f"{lo}-{hi}": share_full(lo, hi) for lo, hi in thirds},
        "rules": {
            name: {"epoch": e, "val_loss": lines[e - 1]["val_loss"], **cards[e]}
            for name, e in rules.items()
        },
        "best_heldout_epoch_for_reference": max(
            epochs, key=lambda e: (cards[e]["heldout_progress"], -e)
        ),
    }


def baseline_values(paths: dict[str, Any]) -> dict[str, Any]:
    """Each baseline's measures: the random networks per seed, the neuron, the expert."""

    def card(path: Path) -> dict[str, float]:
        return measures(read_card(RunStore.open(path), 0))

    return {
        "random_network": {s: card(p) for s, p in paths["random_network"].items()},
        "hand_neuron": card(paths["hand_neuron"]),
        "scripted_expert": card(paths["scripted_expert"]),
    }


def same_as_task_5_6(runs_dir: Path, paths: dict[str, Any]) -> dict[str, int]:
    """Compare every S5a baseline card with the same baseline's card from task 5.6 (no study):
    everything but the ``policy`` reference (which names the run) must be equal."""
    older: dict[tuple[str, int | None], Path] = {}
    for path in run_folders(runs_dir):
        store = RunStore.open(path)
        meta = store.read_meta()
        if meta.get("study") is None and meta["tag"].startswith("baseline"):
            older.setdefault((meta["method"], store.read_config().get("seed")), path)
    pairs = [(("random_network", s), p) for s, p in paths["random_network"].items()]
    pairs += [(("hand_neuron", None), paths["hand_neuron"])]
    pairs += [(("scripted_expert", None), paths["scripted_expert"])]
    compared = equal = 0
    for key, path in pairs:
        if key in older:
            a, b = read_card(RunStore.open(path), 0), read_card(RunStore.open(older[key]), 0)
            a.pop("policy"), b.pop("policy")
            compared += 1
            equal += a == b
    return {"compared": compared, "equal": equal}


def same_as_task_6_3(runs_dir: Path, seed0: Path) -> dict[str, Any]:
    """Seed 0's first 100 epochs against the 6.3 run (seed 0, 100 epochs, every epoch examined):
    every metrics line but its wall-clock time, and every exam card both runs hold, apart from
    the run name in its ``policy``."""
    old_path = runs_dir / TASK_6_3_RUN
    if not (old_path / "meta.json").is_file():
        return {"found": False}
    old, new = RunStore.open(old_path), RunStore.open(seed0)

    def lines(store: RunStore) -> list[dict[str, Any]]:
        return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]

    shared = sorted(set(old.exam_units()) & set(new.exam_units()))
    equal_cards = 0
    for unit in shared:
        a, b = read_card(old, unit), read_card(new, unit)
        a.pop("policy"), b.pop("policy")
        equal_cards += a == b
    old_lines = lines(old)
    return {
        "found": True,
        "metrics_lines_equal": lines(new)[: len(old_lines)] == old_lines,
        "lines_compared": len(old_lines),
        "shared_exam_cards": len(shared),
        "equal_exam_cards": equal_cards,
    }


def database_check(runs_dir: Path, ids: list[str]) -> dict[str, Any]:
    """Every S5a run folder through the database: its ``study`` column and its ``studies`` row."""
    try:
        with closing(connect(runs_dir / DB_NAME)) as conn:
            listed = runs_of_study(conn, STUDY)
            row = conn.execute(
                "SELECT n_runs, methods, sweeps FROM studies WHERE study_id = ?", (STUDY,)
            ).fetchone()
    except sqlite3.Error as exc:
        return {"error": str(exc)}
    return {
        "same_runs_as_the_folders": listed == sorted(ids),
        "n_runs": row[0] if row else None,
        "methods": json.loads(row[1]) if row else None,
        "n_sweeps": len(json.loads(row[2])) if row else None,
    }


# ── analysis ────────────────────────────────────────────────────────────────────────────────


def _band(values: list[float]) -> dict[str, float]:
    s = summarise(values)
    return {"median": s.median, "q1": s.q1, "q3": s.q3}


RULES = ("last", "min_loss", "training_exam")
POST_HOC_RULES = ("earliest_lap",)  # added after the results; never part of a hypothesis
RULE_MEASURES = ("heldout_progress", "heldout_completion", "heldout_crash", "training_completion")


def hypotheses(
    records: list[dict[str, Any]], agreement_ci: dict[str, Any], difference: dict[str, Any]
) -> dict[str, Any]:
    """The pre-registered rules H1-H4 (docs/studies/S5a.md) on the per-seed records."""
    late = LATE_SHARE * EPOCHS
    h1 = {
        "val_loss_end_below_epoch_30_in_every_seed": all(
            r["val_loss_every_epoch"][-1] < r["val_loss_every_epoch"][H1_FROM - 1] for r in records
        ),
        "min_examined_loss_epoch_late_in_every_seed": all(
            r["rules"]["min_loss"]["epoch"] >= late for r in records
        ),
    }
    shares = [r["agreement_heldout_progress"]["share"] for r in records]
    h2 = {
        "median_agreement": agreement_ci["median"],
        "median_below_half": agreement_ci["median"] < 0.5,
        "seeds_without_a_value": sum(s is None for s in shares),
    }
    early = [r["first_full_lap_epoch"] for r in records]
    h3 = {
        "first_full_lap_epoch_per_seed": early,
        "every_seed_laps_by_epoch_10": all(e is not None and e <= EARLY for e in early),
    }
    h4 = {
        "estimate": difference["estimate"],
        "ci": difference["ci"],
        "supported": difference["estimate"] > 0 and difference["ci"][0] > 0,
    }
    h1["supported"] = all(h1.values())
    h2["supported"] = h2["median_below_half"] and h2["seeds_without_a_value"] == 0
    h3["supported"] = h3["every_seed_laps_by_epoch_10"]
    return {"H1": h1, "H2": h2, "H3": h3, "H4": h4}


def analyse(runs_dir: Path) -> dict[str, Any]:
    paths = find_runs(runs_dir)
    records = [seed_record(paths["imitation"][s]) for s in SEEDS]
    base = baseline_values(paths)
    epochs = records[0]["epochs"]
    if any(r["epochs"] != epochs for r in records):
        raise SystemExit("the seeds were examined at different epochs")

    def rng():
        return make_rng(BOOTSTRAP_SEED)

    # Agreement of the validation loss with each driving score, over the seeds.
    agreement = {}
    for key in ("agreement_heldout_progress", "agreement_training_completion"):
        shares = [r[key]["share"] for r in records if r[key]["share"] is not None]
        ci = bootstrap_median_ci(shares, rng())
        agreement[key] = {
            "median": ci.estimate,
            "ci": [ci.low, ci.high],
            **_band(shares),
            "spearman": _band(
                [r[key]["spearman"] for r in records if r[key]["spearman"] is not None]
            ),
        }

    # The checkpoint rules: each rule's held-out scores, and paired differences between rules.
    rules = {
        rule: {
            m: {**_band(v := [r["rules"][rule][m] for r in records]), "seeds": v}
            for m in (*RULE_MEASURES, "training_speed", "val_loss", "epoch")
        }
        for rule in (*RULES, *POST_HOC_RULES)
    }
    differences = {}
    pairs = (("training_exam", "min_loss"), ("training_exam", "last"))
    for a, b in (*pairs, ("earliest_lap", "training_exam")):
        for m in ("heldout_progress", "heldout_completion"):
            d = bootstrap_median_difference(
                [r["rules"][a][m] for r in records],
                [r["rules"][b][m] for r in records],
                rng(),
                paired=True,
            )
            differences[f"{a}-{b}:{m}"] = {"estimate": d.estimate, "ci": [d.low, d.high]}

    # Curves over the examined epochs (the same epochs for every seed), and the loss every epoch.
    names = ("heldout_progress", "heldout_completion", "training_completion", "training_speed")
    curves = {
        n: [{**_band(v := [r["cards"][e][n] for r in records]), "seeds": v} for e in [0, *epochs]]
        for n in names
    }
    val_curve = [_band([r["val_loss_every_epoch"][i] for r in records]) for i in range(EPOCHS)]

    random_band = {
        n: _band([base["random_network"][s][n] for s in RANDOM_SEEDS]) for n in base["hand_neuron"]
    }
    ids = sorted(p.name for p in paths["all"])
    metas = [RunStore.open(p).read_meta() for p in paths["imitation"].values()]

    def one_or_all(values: list[Any]) -> Any:
        distinct = sorted(set(values), key=str)
        return distinct[0] if len(distinct) == 1 else distinct

    seeds = []
    for r in records:
        seeds.append(
            {
                k: v
                for k, v in r.items()
                if k not in ("cards", "val_loss_every_epoch", "loss_every_epoch", "epochs")
            }
        )
        seeds[-1]["examined"] = {
            str(e): {"val_loss": r["val_loss_every_epoch"][e - 1], **r["cards"][e]} for e in epochs
        }
        seeds[-1]["val_loss_every_epoch"] = r["val_loss_every_epoch"]

    return {
        "study": STUDY,
        "design": {
            "config": "configs/imitation.yaml",
            "dataset": DATASET,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "checkpoint_every": CHECKPOINT_EVERY,
            "examined_epochs": epochs,
            "random_network_seeds": list(RANDOM_SEEDS),
            "bootstrap": {"resamples": 10_000, "seed": BOOTSTRAP_SEED},
            # The short form: a full 40-digit hash looks like a secret to the secret scanners.
            "git_commit": one_or_all([(m["git_commit"] or "")[:7] or None for m in metas]),
            "git_dirty": one_or_all([m["git_dirty"] for m in metas]),
            "sweep": one_or_all([m["sweep"] for m in metas]),
            "dataset_sha256_prefix": one_or_all([m["dataset"]["sha256"][:12] for m in metas]),
        },
        "run_ids": ids,
        "database": database_check(runs_dir, ids),
        "baselines_equal_task_5_6": same_as_task_5_6(runs_dir, paths),
        "seed0_equals_task_6_3": same_as_task_6_3(runs_dir, paths["imitation"][0]),
        "baselines": {
            "random_network": {"band": random_band, "seeds": base["random_network"]},
            "hand_neuron": base["hand_neuron"],
            "scripted_expert": base["scripted_expert"],
        },
        "agreement": agreement,
        "rules": rules,
        "rule_differences": differences,
        "curves": curves,
        "val_loss_curve": val_curve,
        "seeds": seeds,
        "hypotheses": hypotheses(
            records,
            agreement["agreement_heldout_progress"],
            differences["training_exam-min_loss:heldout_progress"],
        ),
    }


def _rounded(obj: Any) -> Any:
    """Floats to 6 significant digits, recursively (keys sorted when written)."""
    if isinstance(obj, float):
        return float(f"{obj:.6g}")
    if isinstance(obj, dict):
        return {str(k): _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_rounded(v) for v in obj]
    return obj


def print_tables(a: dict[str, Any]) -> None:
    print(f"S5a runs analysed: {len(a['run_ids'])}; database: {a['database']}")
    print(f"baseline cards equal to task 5.6: {a['baselines_equal_task_5_6']}")
    print(f"seed 0 against the 6.3 run: {a['seed0_equals_task_6_3']}")
    print(
        "\nper seed: lowest-loss epoch (examined / all), first lap, first lost, last lap,"
        " full-lap share by third, A(V,P), A(V,T)"
    )
    for s in a["seeds"]:
        ap, at = s["agreement_heldout_progress"], s["agreement_training_completion"]
        thirds = " ".join(f"{v:.2f}" for v in s["full_lap_share_by_third"].values())
        print(
            f"  seed {s['seed']}: {s['rules']['min_loss']['epoch']:>3} / "
            f"{s['min_val_loss_epoch_all']:>3}  lap {s['first_full_lap_epoch']} lost "
            f"{s['first_lost_lap_epoch']} last "
            f"{s['last_full_lap_epoch']}  [{thirds}]  A(V,P) {ap['share']:.3f}  A(V,T) "
            f"{at['share'] if at['share'] is None else round(at['share'], 3)}  val episodes "
            f"{s['val_episodes']}"
        )
    for key, g in a["agreement"].items():
        print(
            f"{key}: median {g['median']:.3f} CI {g['ci']} IQR [{g['q1']:.3f}, {g['q3']:.3f}]"
            f"  spearman median {g['spearman']['median']:.3f}"
        )
    print("\ncheckpoint rules (median [IQR] over the seeds):")
    for rule, m in a["rules"].items():
        print(
            f"  {rule:<14} epoch {m['epoch']['median']:.0f}  held-out progress "
            f"{m['heldout_progress']['median']:.3f} [{m['heldout_progress']['q1']:.3f}, "
            f"{m['heldout_progress']['q3']:.3f}]  completion "
            f"{m['heldout_completion']['median']:.3f}  training laps "
            f"{m['training_completion']['median']:.2f}  val loss "
            f"{m['val_loss']['median']:.5f}"
        )
    for key, d in a["rule_differences"].items():
        print(f"  {key}: {d['estimate']:.3f} CI {d['ci']}")
    rn = a["baselines"]["random_network"]["band"]
    for name in ("heldout_progress", "heldout_completion"):
        print(
            f"baselines {name}: random networks median {rn[name]['median']:.3f}, neuron "
            f"{a['baselines']['hand_neuron'][name]:.3f}, expert "
            f"{a['baselines']['scripted_expert'][name]:.3f}"
        )
    for h, v in a["hypotheses"].items():
        print(f"{h}: {v}")


# ── figures ─────────────────────────────────────────────────────────────────────────────────

BLUE = "#2a78d6"  # categorical slot 1 of the dataviz reference palette (light), as in S3 and S4
SEED_GREY = "#9a9a96"
BASELINE_STYLE = {
    "random_network": ("#888888", ":", "random networks (median of 10)"),
    "hand_neuron": ("#c2410c", "--", "hand-wired neuron"),
    "scripted_expert": ("#111111", "-.", "scripted expert (the teacher)"),
}
# The reference palette's blue ramp, steps 250 to 700: one hue, light (early) to dark (late).
EPOCH_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")


def _level(a: dict[str, Any], baseline: str, name: str) -> float:
    if baseline == "random_network":
        return a["baselines"]["random_network"]["band"][name]["median"]
    return a["baselines"][baseline][name]


def figures(a: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    epochs = np.array([0, *a["design"]["examined_epochs"]], dtype=float)

    # 1. The loss (top, the learner's own number) and the exam (below), against the epoch.
    #    Separate panels: the loss and the exam never share an axis (protocol rule 3).
    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=True)
    every = np.arange(1, EPOCHS + 1)
    for s in a["seeds"]:
        axes[0].plot(every, s["val_loss_every_epoch"], color=SEED_GREY, lw=0.7, alpha=0.6)
    axes[0].plot(
        every,
        [p["median"] for p in a["val_loss_curve"]],
        color=BLUE,
        lw=2.2,
        label="median of 10 seeds",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("validation loss (MSE)\nlog scale")
    axes[0].set_title("The learner's own score: validation loss after every epoch (thin = seeds)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    for ax, name, label in (
        (axes[1], "heldout_progress", "held-out median progress\n(share of 2 laps)"),
        (axes[2], "training_completion", "training loop: share of the\n10 starts that finish"),
    ):
        c = a["curves"][name]
        med, q1, q3 = (np.array([p[k] for p in c]) for k in ("median", "q1", "q3"))
        for i in range(len(a["seeds"])):
            ax.plot(epochs, [p["seeds"][i] for p in c], color=SEED_GREY, lw=0.6, alpha=0.45)
        ax.fill_between(epochs, q1, q3, color=BLUE, alpha=0.15, lw=0)
        ax.plot(epochs, med, color=BLUE, lw=2.2, label="median, IQR band")
        for baseline, (colour, style, text) in BASELINE_STYLE.items():
            if baseline == "random_network" and name == "training_completion":
                continue  # 0, on the axis
            ax.axhline(_level(a, baseline, name), color=colour, ls=style, lw=1.2, label=text)
        ax.set_ylim(-0.03, 1.05)
        ax.set_ylabel(label)
    axes[1].set_title("The Driving Exam of the same networks (every 3rd epoch and the schedule)")
    axes[1].legend(frameon=True, fontsize=8, ncol=2, loc="upper right", framealpha=0.9)
    axes[2].set_xlabel("epoch")
    for ax in axes:
        ax.grid(True, color="#e5e5e5", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s5a_loss_and_exam.png", dpi=130)
    plt.close(fig)

    # 2. Validation loss against the driving score: every seed, every examined epoch.
    ramp = LinearSegmentedColormap.from_list("epochs", EPOCH_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    xs, ys_p, ys_t, cs = [], [], [], []
    for s in a["seeds"]:
        for e, card in s["examined"].items():
            xs.append(card["val_loss"])
            ys_p.append(card["heldout_progress"])
            ys_t.append(card["training_completion"])
            cs.append(int(e))
    for ax, ys, label in (
        (axes[0], ys_p, "held-out median progress (share of 2 laps)"),
        (axes[1], ys_t, "training loop: share of the 10 starts that finish"),
    ):
        dots = ax.scatter(
            xs,
            ys,
            c=cs,
            cmap=ramp,
            s=16,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            vmin=1,
            vmax=EPOCHS,
        )
        ax.set_xscale("log")
        ax.invert_xaxis()  # lower loss to the right: "better" points right on both axes
        ax.set_xlabel("validation loss (log scale, decreasing to the right)")
        ax.set_ylabel(label)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(True, color="#e5e5e5", lw=0.8)
    fig.colorbar(dots, ax=axes, label="epoch", fraction=0.025, pad=0.02)
    fig.suptitle(
        "S5a: if lower loss meant better driving, the points would rise to the right "
        f"(10 seeds x {len(a['design']['examined_epochs'])} examined epochs)"
    )
    fig.savefig(FIGURES_DIR / "s5a_loss_vs_driving.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 3. The checkpoint rules on the held-out tracks, next to the baselines; the post-hoc rule
    #    is drawn apart (hatched, lighter) so it is never read as a pre-registered result.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    labels = {
        "last": "last epoch\n(300)",
        "min_loss": "lowest\nvalidation loss",
        "training_exam": "best training-loop\nexam (D3A)",
        "earliest_lap": "earliest full lap\n(post hoc)",
    }
    shown = (*RULES, *POST_HOC_RULES)
    for ax, name, label in (
        (axes[0], "heldout_progress", "held-out median progress"),
        (axes[1], "heldout_completion", "held-out completion rate"),
    ):
        x = np.arange(len(shown))
        med = np.array([a["rules"][r][name]["median"] for r in shown])
        err = np.array(
            [
                [med[i] - a["rules"][r][name]["q1"] for i, r in enumerate(shown)],
                [a["rules"][r][name]["q3"] - med[i] for i, r in enumerate(shown)],
            ]
        )
        pre = len(RULES)
        ax.bar(
            x[:pre],
            med[:pre],
            0.55,
            yerr=err[:, :pre],
            color=BLUE,
            capsize=4,
            label="median, IQR (10 seeds)",
        )
        ax.bar(
            x[pre:],
            med[pre:],
            0.55,
            yerr=err[:, pre:],
            color="#86b6ef",
            hatch="//",
            edgecolor="#1c5cab",
            capsize=4,
            label="post hoc (not a hypothesis)",
        )
        for i, r in enumerate(shown):
            seeds = a["rules"][r][name]["seeds"]
            ax.scatter(
                np.full(len(seeds), i) + np.linspace(-0.18, 0.18, len(seeds)),
                seeds,
                s=12,
                color="#ffffff",
                edgecolors="#104281",
                linewidths=0.8,
                zorder=3,
                label="one seed" if i == 0 else None,
            )
        for baseline, (colour, style, text) in BASELINE_STYLE.items():
            ax.axhline(_level(a, baseline, name), color=colour, ls=style, lw=1.2, label=text)
        ax.set_xticks(x, [labels[r] for r in shown])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", color="#e5e5e5", lw=0.8)
    axes[1].legend(
        frameon=True, framealpha=0.9, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.93)
    )
    fig.suptitle("S5a: which checkpoint to keep? (none of the rules looks at a held-out track)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s5a_checkpoint_rules.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", action="store_true", help="make S5a's runs first (~25 min)")
    parser.add_argument("--figures", action="store_true", help="draw the figures (matplotlib)")
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"), help="folder of run folders")
    parser.add_argument("--workers", type=int, default=6, help="runs trained at once")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    if args.train:
        for name in train(runs_dir, args.workers):
            print(f"run folder: {name}")
    analysis = analyse(runs_dir)
    print_tables(analysis)
    text = json.dumps(_rounded(analysis), indent=1, sort_keys=True) + "\n"
    RESULTS_FILE.write_text(text, encoding="utf-8", newline="\n")
    print(f"\nwrote {RESULTS_FILE.relative_to(ROOT).as_posix()}")
    if args.figures:
        figures(analysis)
        print(f"wrote figures to {FIGURES_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
