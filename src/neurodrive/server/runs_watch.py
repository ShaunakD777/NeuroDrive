"""Launching training runs and following them while they train (task 5.8, ADR-041, api v1.13).

Training never happens inside the server. :class:`Launcher` starts each run as a separate
process running the same ``nd train`` command a terminal would, and that process only writes its
run folder. :class:`RunsWatcher` looks at the run folders every half second and turns what is new
into ``/ws/runs`` messages; :class:`RunsHub` hands them to the connected browsers. Because the
watcher reads only the run folder contract (docs/ARCHITECTURE.md §6.13), a run started with
``nd train`` in a terminal shows up too, and every later learning method is followed for free.

**Following a growing file ("tailing").** For each running run the watcher keeps the offset o,
the number of bytes of ``metrics.jsonl`` it has already read. A poll reads the bytes from o to
the file's current size n and keeps only the complete lines (those ending in a newline); o moves
forward by exactly their length. A line still being written is left for the next poll, so it is
never read in half. Example: two 40-byte lines read (o = 80); the trainer has written 25 bytes of
a third (n = 105): no newline after byte 80, nothing is sent, o stays 80; next poll n = 120: one
complete line, sent, o = 120.

**Stopping** (ADR-041): ``POST /api/runs/{id}/stop`` writes the empty file ``stop_requested``
into the run folder; the runner ends the run after its current unit (stop reason "stopped").
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from neurodrive import __version__
from neurodrive.experiment.config import (
    ConfigError,
    RandomSearchRunConfig,
    ToyRunConfig,
    load_config,
)
from neurodrive.experiment.store import STOP_FILE, RunStore
from neurodrive.server.runs import RUN_ID
from neurodrive.server.schemas import (
    CheckpointMessage,
    ConfigInfo,
    ErrorMessage,
    ExamDoneMessage,
    ExamHeadline,
    HelloMessage,
    LaunchRequest,
    RunMetricsMessage,
    RunStatusMessage,
    WatchRun,
    WireModel,
)

POLL_SECONDS = 0.5  # how often the watcher looks at the run folders
MAX_ACTIVE_RUNS = 4  # training processes this server runs at once (one CPU core each)
LAUNCH_TIMEOUT_S = 60.0  # how long a new process may take to create its run folder
LAUNCH_DIR = ".launches"  # <runs dir>/.launches/<time>.log: what each launched process printed
_RUN_FOLDER_LINE = re.compile(r"^run folder: (.+?)\s*$", re.MULTILINE)

# ── launching ──────────────────────────────────────────────────────────────────────────────────


class LaunchError(Exception):
    """A launch or stop that cannot be done; ``status`` is the HTTP status code to answer with
    (404 unknown run, 409 not possible now, 422 bad request, 500 the process did not start) and
    ``detail`` what to tell the browser."""

    def __init__(self, status: int, detail: str | list[str]) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def _own_process_group() -> dict[str, Any]:
    """Popen settings that keep a training process out of the server's console group, so that
    stopping the server (Ctrl+C) or closing its window does not stop the training."""
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        return {"creationflags": flags}
    return {"start_new_session": True}


@dataclass
class _Launch:
    process: subprocess.Popen
    log: Path
    run_id: str | None = None


class Launcher:
    """Starts ``nd train`` processes for configs in ``configs_dir``, writing into ``runs_dir``.

    Thread-safe: launches arrive from the server's worker threads while the watcher asks for
    exit codes from the event loop.
    """

    def __init__(
        self,
        runs_dir: Path,
        configs_dir: Path,
        max_active: int = MAX_ACTIVE_RUNS,
        launch_timeout_s: float = LAUNCH_TIMEOUT_S,
    ) -> None:
        # Absolute, because the training process runs in another working folder.
        self.runs_dir, self.configs_dir = Path(runs_dir).resolve(), Path(configs_dir).resolve()
        self.max_active, self.launch_timeout_s = max_active, launch_timeout_s
        self._lock = threading.Lock()
        self._launches: list[_Launch] = []

    def configs(self) -> list[ConfigInfo]:
        """The training configs: every ``*.yaml`` directly in ``configs_dir`` that ``nd train``
        accepts (files in sub-folders, such as ``configs/baselines/``, are not training configs)."""
        found = []
        for path in sorted(self.configs_dir.glob("*.yaml")):
            try:
                config = load_config(path)
            except ConfigError:
                continue
            toy = isinstance(config, ToyRunConfig)
            by_units = isinstance(config, RandomSearchRunConfig)
            found.append(
                ConfigInfo(
                    name=path.name,
                    method=config.method,
                    task=config.task if toy else None,
                    seed=config.seed,
                    tag=config.tag,
                    length_field="units" if by_units else "epochs",
                    length=config.budget.units if by_units else config.training.epochs,
                )
            )
        return found

    def launch(self, request: LaunchRequest) -> str:
        """Start training ``request.config`` with its overrides; returns the new run's id.

        Steps: (1) the config must be one :meth:`configs` lists, and it must validate with the
        overrides exactly as ``nd train`` would (422 otherwise, nothing started); (2) at most
        ``max_active`` processes at once (409); (3) start ``python -m neurodrive train ...`` in its
        own process group, its output going to ``<runs dir>/.launches/<time>.log``; (4) wait for
        the line ``run folder: <path>``, which ``nd train`` prints as soon as the folder exists
        (500 if the process ends first or takes longer than ``launch_timeout_s``).
        """
        names = [c.name for c in self.configs()]
        if request.config not in names:
            raise LaunchError(
                422, [f"config: unknown config {request.config!r}; choose from {', '.join(names)}"]
            )
        path = self.configs_dir / request.config
        overrides = {
            "seed": request.seed,
            "tag": request.tag,
            "epochs": request.epochs,
            "units": request.units,
        }
        try:
            load_config(path, **overrides)
        except ConfigError as exc:
            raise LaunchError(422, exc.problems) from None
        argv = [sys.executable, "-m", "neurodrive", "train", "--config", str(path)]
        for name, value in overrides.items():
            if value is not None:
                argv += [f"--{name}", str(value)]
        # Run in the runs folder's parent and name the folder alone, so the process prints
        # "run folder: runs/<id>" and no absolute path (with a user name) reaches the log.
        argv += ["--runs-dir", self.runs_dir.name]
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
        with self._lock:
            if self._active() >= self.max_active:
                raise LaunchError(
                    409,
                    f"{self.max_active} launched runs are already training; "
                    "wait for one to finish or stop one",
                )
            log = self._new_log()
            with open(log, "wb") as out:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    cwd=self.runs_dir.parent,
                    env=env,
                    **_own_process_group(),
                )
            launch = _Launch(process, log)
            self._launches.append(launch)
        run_id = self._wait_for_run_id(launch)
        with self._lock:
            launch.run_id = run_id
        return run_id

    def stop(self, store: RunStore) -> None:
        """Ask the run of ``store`` to stop after its current unit (the stop file, ADR-041).

        Works for any running run, also one started from a terminal. 409 if the run has already
        ended, or if this server started it and its process is gone (it was killed: the request
        would never be read; ``nd resume`` continues such a run).
        """
        status = store.read_meta()["status"]
        if status != "running":
            raise LaunchError(409, f"the run is already {status}")
        code = self.exit_codes().get(store.path.name)
        if code is not None:
            raise LaunchError(
                409,
                f"its training process has ended (exit code {code}) without finishing the run; "
                "nd resume can continue it",
            )
        store.request_stop()

    def exit_codes(self) -> dict[str, int | None]:
        """Each launched run's process exit code (None while it still runs)."""
        with self._lock:
            return {
                launch.run_id: launch.process.poll()
                for launch in self._launches
                if launch.run_id is not None
            }

    def _active(self) -> int:
        return sum(launch.process.poll() is None for launch in self._launches)

    def _new_log(self) -> Path:
        folder = self.runs_dir / LAUNCH_DIR
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        return folder / f"{stamp}.log"

    def _wait_for_run_id(self, launch: _Launch) -> str:
        deadline = time.monotonic() + self.launch_timeout_s
        while True:
            text = launch.log.read_text(encoding="utf-8", errors="replace")
            match = _RUN_FOLDER_LINE.search(text)
            if match:
                return Path(match.group(1)).name
            if launch.process.poll() is not None:
                tail = "\n".join(text.strip().splitlines()[-15:])
                raise LaunchError(500, f"the training process ended before starting:\n{tail}")
            if time.monotonic() > deadline:
                launch.process.kill()
                raise LaunchError(
                    500, f"the training process did not start within {self.launch_timeout_s:.0f} s"
                )
            time.sleep(0.05)


# ── watching ───────────────────────────────────────────────────────────────────────────────────


def _file_key(path: Path) -> tuple[int, int] | None:
    """Modification time and size: a cheap sign that a file changed (None if it is missing)."""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _read_json(path: Path) -> Any | None:
    """The file's JSON, or None while it is missing or half-written (read it again later)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _headline(part: dict[str, Any]) -> ExamHeadline:
    return ExamHeadline(
        completion_rate=part["completion_rate"],
        crash_rate=part["crash_rate"],
        progress_frac_median=part["progress_frac_median"],
    )


@dataclass
class _Tracked:
    """What the watcher has read of one running run: everything it has sent about it so far."""

    path: Path
    resumes: int  # len(meta["resumed_at"]) when tracking started; a change means a resume
    offset: int = 0  # bytes of metrics.jsonl read (only complete lines)
    records: list[dict[str, Any]] = field(default_factory=list)
    index_key: tuple[int, int] | None = None
    checkpoints: list[int] = field(default_factory=list)
    exams: dict[int, ExamDoneMessage] = field(default_factory=dict)

    def read_new(self, run_id: str, replace: bool = False) -> list[WireModel]:
        """Read what is new in the folder; the messages for it (metrics, checkpoint, exams).

        ``replace`` (or a metrics file shorter than what was read: a resume cut it back) starts
        again from nothing, and the metrics message then carries every line with ``replace``.
        Checkpoints and exam cards are listed *before* the metrics are read: the runner writes a
        unit's metrics line before its checkpoint and card, so every card seen here has its line
        among the records read just after. The messages go out metrics first.
        """
        if replace or self._was_cut():
            self._forget()
            replace = True
        checkpoint = self._read_checkpoints(run_id)
        exams = self._read_exams(run_id)
        new = self._tail()
        out: list[WireModel] = []
        if replace:
            out.append(RunMetricsMessage(run_id=run_id, replace=True, records=list(self.records)))
        elif new:
            out.append(RunMetricsMessage(run_id=run_id, replace=False, records=new))
        if checkpoint is not None:
            out.append(checkpoint)
        return out + exams

    def history(self, run_id: str) -> list[WireModel]:
        """Everything read so far, as the messages a newly watching browser needs."""
        out: list[WireModel] = [
            RunMetricsMessage(run_id=run_id, replace=True, records=list(self.records)),
            CheckpointMessage(run_id=run_id, units=list(self.checkpoints)),
        ]
        return out + [self.exams[u] for u in sorted(self.exams)]

    def _was_cut(self) -> bool:
        """The metrics file is shorter than what was read: a resume cut it back (ADR-037)."""
        key = _file_key(self.path / "metrics.jsonl")
        return key is not None and key[1] < self.offset

    def _forget(self) -> None:
        self.offset, self.records, self.index_key = 0, [], None
        self.checkpoints, self.exams = [], {}

    def _tail(self) -> list[dict[str, Any]]:
        """The new complete lines of metrics.jsonl (from the offset to the last newline)."""
        path = self.path / "metrics.jsonl"
        key = _file_key(path)
        if key is None or key[1] <= self.offset:
            return []
        with open(path, "rb") as f:
            f.seek(self.offset)
            data = f.read(key[1] - self.offset)
        end = data.rfind(b"\n") + 1  # everything after the last newline is still being written
        new = [json.loads(line) for line in data[:end].splitlines() if line.strip()]
        self.offset += end
        self.records.extend(new)
        return new

    def _read_checkpoints(self, run_id: str) -> CheckpointMessage | None:
        path = self.path / "checkpoints" / "index.json"
        key = _file_key(path)
        if key is None or key == self.index_key:
            return None
        index = _read_json(path)
        if index is None:
            return None
        self.index_key = key
        units = [int(u) for u in index["units"]]
        if units == self.checkpoints:
            return None
        self.checkpoints = units
        return CheckpointMessage(run_id=run_id, units=units)

    def _read_exams(self, run_id: str) -> list[ExamDoneMessage]:
        folder = self.path / "exams"
        if not folder.is_dir():
            return []
        out = []
        for card_path in sorted(folder.glob("unit_*.json")):
            unit = int(card_path.stem.removeprefix("unit_"))
            if unit in self.exams:
                continue
            card = _read_json(card_path)
            if card is None:
                continue
            message = ExamDoneMessage(
                run_id=run_id,
                unit=unit,
                training=_headline(card["training"]),
                heldout=_headline(card["heldout"]),
            )
            self.exams[unit] = message
            out.append(message)
        return sorted(out, key=lambda m: m.unit)


class RunsWatcher:
    """Turns changes in the run folders under ``runs_dir`` into ``/ws/runs`` messages.

    :meth:`poll` is plain synchronous code (no timers), so tests call it directly. The first poll
    only takes stock: it starts tracking the runs that are running and sends nothing.
    """

    def __init__(self, runs_dir: Path, launcher: Launcher | None = None) -> None:
        self.runs_dir = Path(runs_dir)
        self.launcher = launcher
        self._meta: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}
        self._tracked: dict[str, _Tracked] = {}
        self._status: dict[str, RunStatusMessage] = {}  # the last status sent per run
        self._primed = False

    def poll(self) -> list[WireModel]:
        """Look at every run folder once; the messages for whatever changed since the last poll.

        Steps per folder: (1) re-read ``meta.json`` only if its time or size changed; (2) a run
        whose status is running is tracked (again from scratch after a resume); (3) a tracked run's
        new metrics lines, checkpoints and exam cards become messages; (4) its status message is
        recomputed and sent if it differs from the last one; (5) a run that has ended is read one
        last time and no longer tracked.
        """
        exit_codes = self.launcher.exit_codes() if self.launcher else {}
        events: list[WireModel] = []
        present = set()
        for path in self._folders():
            run_id = path.name
            present.add(run_id)
            changed, meta = self._read_meta(path)
            if meta is None:
                continue
            tracked = self._tracked.get(run_id)
            resumes = len(meta.get("resumed_at") or [])
            start = meta["status"] == "running" and (tracked is None or resumes != tracked.resumes)
            if start:  # a new run, a resumed one, or one seen at the first poll
                tracked = self._tracked[run_id] = _Tracked(path, resumes)
            if tracked is not None:
                # Tracking that starts sends the whole history (replace): a browser may hold an
                # older one, e.g. of a failed run that has just been resumed.
                events += tracked.read_new(run_id, replace=start)
                if meta["status"] != "running":
                    del self._tracked[run_id]
            if tracked is not None or run_id in exit_codes or (changed and self._primed):
                status = self._status_message(path, meta, tracked, changed, exit_codes)
                if status != self._status.get(run_id):
                    self._status[run_id] = status
                    events.append(status)
        for gone in set(self._meta) - present:  # a folder deleted by hand
            for table in (self._meta, self._tracked, self._status):
                table.pop(gone, None)
        if not self._primed:
            self._primed = True
            return []
        return events

    def statuses(self) -> list[RunStatusMessage]:
        """What a newly connected browser is told first: every running run and every run this
        server launched, by run id."""
        return [s for _, s in sorted(self._status.items()) if s.status == "running" or s.launched]

    def history(self, run_id: str) -> list[WireModel]:
        """Everything about ``run_id`` so far, for a browser that starts watching it.

        A tracked run answers from what was already read, so the next poll's messages continue
        exactly where this history ends. A running run that is not tracked yet starts being
        tracked here, with nothing read: its lines then all arrive with the next poll. A finished
        run is read from its folder (nothing will follow). ``KeyError`` for an unknown run.
        """
        path = self.runs_dir / run_id
        if not RUN_ID.fullmatch(run_id) or not (path / "meta.json").is_file():
            raise KeyError(run_id)
        tracked = self._tracked.get(run_id)
        if tracked is None:
            _, meta = self._read_meta(path)
            if meta is None:
                raise KeyError(run_id)
            resumes = len(meta.get("resumed_at") or [])
            if meta["status"] == "running":
                tracked = self._tracked[run_id] = _Tracked(path, resumes)
            else:
                finished = _Tracked(path, resumes)
                finished.read_new(run_id)
                return finished.history(run_id)
        return tracked.history(run_id)

    def _folders(self) -> list[Path]:
        if not self.runs_dir.is_dir():
            return []
        return sorted(
            p
            for p in self.runs_dir.iterdir()
            if RUN_ID.fullmatch(p.name) and (p / "meta.json").is_file()
        )

    def _read_meta(self, path: Path) -> tuple[bool, dict[str, Any] | None]:
        """(changed, meta): meta.json is read again only when its time or size changed."""
        key = _file_key(path / "meta.json")
        seen = self._meta.get(path.name)
        if key is None or (seen is not None and seen[0] == key):
            return False, seen[1] if seen else None
        meta = _read_json(path / "meta.json")
        if meta is None:  # half-written: keep the old one and look again next time
            return False, seen[1] if seen else None
        self._meta[path.name] = (key, meta)
        return True, meta

    def _status_message(
        self,
        path: Path,
        meta: dict[str, Any],
        tracked: _Tracked | None,
        changed: bool,
        exit_codes: dict[str, int | None],
    ) -> RunStatusMessage:
        run_id = path.name
        previous = self._status.get(run_id)
        if tracked is not None:
            units_done = len(tracked.records)
        elif previous is not None and not changed:
            units_done = previous.units_done
        else:  # a run seen for the first time already finished: count its lines once
            data = (path / "metrics.jsonl").read_bytes()
            units_done = sum(1 for line in data.splitlines() if line.strip())
        return RunStatusMessage(
            run_id=run_id,
            method=meta["method"],
            tag=meta["tag"],
            status=meta["status"],
            stop_reason=meta.get("stop_reason"),
            units_done=units_done,
            budget_units=(meta.get("budget") or {}).get("units"),
            stop_requested=meta["status"] == "running" and (path / STOP_FILE).is_file(),
            launched=run_id in exit_codes,
            exit_code=exit_codes.get(run_id),
        )


# ── the /ws/runs connections ───────────────────────────────────────────────────────────────────


@dataclass
class RunsClient:
    """One ``/ws/runs`` connection: the run it watches and its outgoing messages, in order."""

    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    watching: str | None = None

    def put(self, message: WireModel) -> None:
        self.queue.put_nowait(message)


class RunsHub:
    """The connected browsers and the poll loop that feeds them.

    Every message is put into a connection's queue synchronously, on the event loop, and one
    writer per connection sends the queue in order. So a history put in by ``watch`` can never
    be overtaken by a later poll's messages. The loop runs only while someone is connected.
    """

    def __init__(self, watcher: RunsWatcher, poll_seconds: float = POLL_SECONDS) -> None:
        self.watcher, self.poll_seconds = watcher, poll_seconds
        self._clients: list[RunsClient] = []
        self._task: asyncio.Task | None = None

    def connect(self) -> RunsClient:
        """A new connection: catch up with the folders, then hello and the current statuses."""
        polling = self._task is not None and not self._task.done()
        if not polling:
            self._dispatch(self.watcher.poll())  # changes while nobody was connected
        client = RunsClient()
        client.put(HelloMessage(version=__version__))
        for status in self.watcher.statuses():
            client.put(status)
        self._clients.append(client)
        if not polling:
            self._task = asyncio.get_running_loop().create_task(self._run())
        return client

    def disconnect(self, client: RunsClient) -> None:
        self._clients.remove(client)
        if not self._clients and self._task is not None:
            self._task.cancel()
            self._task = None

    def handle_text(self, client: RunsClient, text: str) -> None:
        """A message from the browser; the only command is ``watch {run_id}``."""
        try:
            data = json.loads(text)
            if not isinstance(data, dict) or data.get("type") != "watch":
                raise ValueError("unknown command; the only one is watch {run_id}")
            command = WatchRun.model_validate(data)
        except (ValueError, ValidationError) as exc:
            client.put(ErrorMessage(message=str(exc)))
            return
        client.watching = command.run_id
        if command.run_id is None:
            return
        try:
            history = self.watcher.history(command.run_id)
        except KeyError:
            client.watching = None
            client.put(ErrorMessage(message=f"unknown run {command.run_id!r}"))
            return
        for message in history:
            client.put(message)

    def _dispatch(self, events: list[WireModel]) -> None:
        for event in events:
            for client in self._clients:
                if isinstance(event, RunStatusMessage) or event.run_id == client.watching:
                    client.put(event)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.poll_seconds)
            try:
                self._dispatch(self.watcher.poll())
            except Exception as exc:  # a strange folder must not end the streaming for everyone
                warnings.warn(f"run watcher: {exc!r}", RuntimeWarning, stacklevel=1)
