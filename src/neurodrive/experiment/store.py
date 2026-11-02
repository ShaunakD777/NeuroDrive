"""Run folders, version 1 (ADR-030, ADR-037): everything one training run leaves on disk.

A run folder is the master record of a run (ADR-007). Version 1 (task 5.5) adds ``exams/``,
``state/`` and the meta fields ``budget``, ``exams``, ``stop_reason``, ``resumed_at`` and
``exam_wall_clock_s`` to version 0 (ADR-030), which is a strict subset and is still read:

    runs/<YYYYMMDD-HHMMSS>-<method>-<tag>/
      config.yaml        the fully resolved config, including the seed
      meta.json          run_version, status, start/end time (UTC), git commit + dirty flag,
                         Python / NumPy / NeuroDrive versions, and ``sweep`` (the sweep id)
                         for runs made by ``nd sweep``
      metrics.jsonl      one JSON object per unit (unit, budget totals, wall_clock_s, the
                         learner's native metrics)
      checkpoints/       unit_<n>.npz (network files, core/nn/io.py) + index.json {"units": [...]}
      exams/             unit_<n>.json: Driving Exam cards of examined checkpoints (exam v1)
      state/             latest.npz: the learner's state after the latest unit, for resuming
      extra/             method-specific arrays, e.g. the toy dataset (extra/dataset.npz)

``status`` is one of running, completed, diverged, failed. The state file is replaced
atomically (written to ``latest.tmp.npz``, then renamed), so a kill never leaves half a state.
Nothing personal is stored: no absolute paths, no user names. ``runs/`` is git-ignored.
"""

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import neurodrive
from neurodrive.core.nn.io import load_network, save_network
from neurodrive.core.nn.network import Network
from neurodrive.experiment.exam import ExamResult
from neurodrive.learn.base import LearnerState

RUN_VERSION = 1
STATUSES = ("running", "completed", "diverged", "failed")
_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")


def _check_name(kind: str, value: str) -> None:
    if not value or not set(value) <= _NAME_CHARS:
        raise ValueError(f"{kind} must be letters, digits, '_' or '-', got {value!r}")


def _git_info() -> dict[str, Any]:
    """Commit hash and dirty flag of the source checkout, or None for both outside a git repo."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunStore:
    """Reads and writes one run folder. Create a new run with :meth:`create`, open one with
    :meth:`open`."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # --- creating and finishing ---

    @classmethod
    def create(
        cls,
        runs_dir: str | Path,
        method: str,
        tag: str,
        config: dict[str, Any],
        when: datetime | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> "RunStore":
        """Make ``<runs_dir>/<YYYYMMDD-HHMMSS>-<method>-<tag>/`` and write config and meta.

        ``when`` (default: now, UTC) sets the timestamp in the name. If that folder already
        exists, ``-2``, ``-3``, ... is appended. The run starts with status "running".
        ``extra_meta`` adds fields to ``meta.json``, e.g. ``{"sweep": <sweep id>}`` (ADR-033).
        """
        _check_name("method", method)
        _check_name("tag", tag)
        when = when or datetime.now(UTC)
        base = Path(runs_dir) / f"{when:%Y%m%d-%H%M%S}-{method}-{tag}"
        path, n = base, 1
        while path.exists():
            n += 1
            path = base.with_name(f"{base.name}-{n}")
        (path / "checkpoints").mkdir(parents=True)
        (path / "extra").mkdir()
        store = cls(path)
        with open(path / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        meta = {
            "run_version": RUN_VERSION,
            "run_id": path.name,
            "method": method,
            "tag": tag,
            "status": "running",
            "started_at": _now(),
            "ended_at": None,
            **_git_info(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "neurodrive_version": neurodrive.__version__,
            **(extra_meta or {}),
        }
        store._write_json("meta.json", meta)
        store._write_json("checkpoints/index.json", {"units": []})
        (path / "metrics.jsonl").touch()
        return store

    @classmethod
    def open(cls, path: str | Path) -> "RunStore":
        """Open an existing run folder (checks that meta.json is there)."""
        store = cls(Path(path))
        if not (store.path / "meta.json").is_file():
            raise FileNotFoundError(f"not a run folder (no meta.json): {store.path.name}")
        return store

    def finish(self, status: str, stop_reason: str | None = None) -> None:
        """Record the final status, why the run stopped (run v1) and the end time."""
        if status not in STATUSES or status == "running":
            raise ValueError(f"final status must be one of {STATUSES[1:]}, got {status!r}")
        meta = self.read_meta()
        meta["status"], meta["ended_at"] = status, _now()
        if stop_reason is not None:
            meta["stop_reason"] = stop_reason
        self._write_json("meta.json", meta)

    def update_meta(self, **fields: Any) -> None:
        """Set (or replace) fields of meta.json."""
        self._write_json("meta.json", {**self.read_meta(), **fields})

    # --- writing during the run ---

    def append_metrics(self, record: dict[str, Any]) -> None:
        """Add one unit's line to metrics.jsonl (``record`` must contain ``unit``)."""
        if "unit" not in record:
            raise ValueError("a metrics record needs a 'unit' entry")
        with open(self.path / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

    def save_checkpoint(self, unit: int, network: Network) -> None:
        """Save ``network`` as checkpoints/unit_<unit>.npz and list it in index.json."""
        save_network(network, self.path / "checkpoints" / f"unit_{unit}.npz")
        units = sorted(set(self.checkpoint_units()) | {unit})
        self._write_json("checkpoints/index.json", {"units": units})

    def save_exam(self, unit: int, result: ExamResult) -> None:
        """Save a checkpoint's exam card as exams/unit_<unit>.json."""
        result.save(self.path / "exams" / f"unit_{unit}.json")

    def save_state(self, state: LearnerState, runner: dict[str, Any]) -> None:
        """Replace state/latest.npz with the learner's ``state`` and the runner's bookkeeping.

        Arrays are stored under their own names with an ``a_`` prefix; ``state.info`` and
        ``runner`` (unit, budget totals, clocks) go into one JSON text entry. The file is written
        under a temporary name and then renamed over the old one: a rename is all-or-nothing, so
        a kill leaves either the old state or the new one, never half of one.
        """
        folder = self.path / "state"
        folder.mkdir(exist_ok=True)
        text = json.dumps({"learner": state.info, "runner": runner}, allow_nan=False)
        arrays = {f"a_{name}": value for name, value in state.arrays.items()}
        tmp = folder / "latest.tmp.npz"
        np.savez(tmp, __info__=np.array(text), **arrays)
        os.replace(tmp, folder / "latest.npz")

    def load_state(self) -> tuple[LearnerState, dict[str, Any]] | None:
        """The saved learner state and runner bookkeeping, or None if no unit saved one yet."""
        path = self.path / "state" / "latest.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as data:
            info = json.loads(str(data["__info__"]))
            arrays = {key[2:]: data[key] for key in data.files if key.startswith("a_")}
        return LearnerState(arrays=arrays, info=info["learner"]), info["runner"]

    def cut_back(self, unit: int) -> None:
        """Forget everything after ``unit``: later metrics lines, checkpoints and exam cards, and a
        state file a kill left half-written. Used before resuming from a state of ``unit``."""
        kept = [m for m in self.read_metrics() if m["unit"] <= unit]
        text = "".join(json.dumps(m, allow_nan=False) + "\n" for m in kept)
        tmp = self.path / "metrics.tmp.jsonl"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path / "metrics.jsonl")
        later = [u for u in self.checkpoint_units() if u > unit]
        for u in later:
            (self.path / "checkpoints" / f"unit_{u}.npz").unlink(missing_ok=True)
        self._write_json(
            "checkpoints/index.json", {"units": [u for u in self.checkpoint_units() if u <= unit]}
        )
        for u in self.exam_units():
            if u > unit:
                (self.path / "exams" / f"unit_{u}.json").unlink()
        (self.path / "state" / "latest.tmp.npz").unlink(missing_ok=True)

    def save_extra(self, name: str, **arrays: np.ndarray) -> None:
        """Save named arrays to extra/<name>.npz."""
        _check_name("extra name", name)
        np.savez(self.path / "extra" / f"{name}.npz", **arrays)

    # --- reading ---

    def read_config(self) -> dict[str, Any]:
        with open(self.path / "config.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def read_meta(self) -> dict[str, Any]:
        return json.loads((self.path / "meta.json").read_text(encoding="utf-8"))

    def read_metrics(self) -> list[dict[str, Any]]:
        lines = (self.path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def checkpoint_units(self) -> list[int]:
        index = json.loads((self.path / "checkpoints/index.json").read_text(encoding="utf-8"))
        return list(index["units"])

    def load_checkpoint(self, unit: int) -> Network:
        if unit not in self.checkpoint_units():
            raise KeyError(f"no checkpoint for unit {unit}")
        return load_network(self.path / "checkpoints" / f"unit_{unit}.npz")

    def exam_units(self) -> list[int]:
        """Units that have an exam card, in order."""
        folder = self.path / "exams"
        if not folder.is_dir():
            return []
        return sorted(int(p.stem.removeprefix("unit_")) for p in folder.glob("unit_*.json"))

    def load_exam(self, unit: int) -> ExamResult:
        return ExamResult.load(self.path / "exams" / f"unit_{unit}.json")

    def load_extra(self, name: str) -> dict[str, np.ndarray]:
        with np.load(self.path / "extra" / f"{name}.npz") as data:
            return {key: data[key] for key in data.files}

    def _write_json(self, relative: str, obj: dict[str, Any]) -> None:
        text = json.dumps(obj, indent=2, allow_nan=False) + "\n"
        (self.path / relative).write_text(text, encoding="utf-8")
