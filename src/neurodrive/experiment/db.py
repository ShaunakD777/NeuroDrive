"""The SQLite mirror of the run folders, db v1 (ADR-007, ADR-038).

Run folders are the master copy (``store.py``); this database is a copy of them in tables, so a
question across many runs is one SQL query. It lives at ``<runs dir>/neurodrive.db`` and can
always be rebuilt from the folders: when the two disagree, the folders win.

Tables (key columns first; each row is one fact):

- ``runs``: one row per run folder: its meta fields as columns (status, seed, sweep, ...), plus
  the whole ``config.yaml`` and ``meta.json`` as JSON text.
- ``metrics``: one row per value of a metrics line, key (run_id, unit, name). A line with k
  values besides ``unit`` becomes k rows; booleans are stored as 0 / 1, and a list such as
  the toy learner's per-layer ``grad_norm`` gives one row per element (``grad_norm[0]``, ...).
- ``checkpoints``: one row per checkpoint, key (run_id, unit), with a copy of the ``.npz``
  file's bytes (``weights``), so the network can be loaded from the database alone.
- ``exams``: one row per exam card, key (run_id, unit): the training and held-out completion
  rate, crash rate and median progress as columns, the full summaries as JSON.
- ``exam_episodes``: one row per exam episode, key (run_id, unit, track, start_index), with
  every episode metric.

Three ways in:

- :class:`Mirror`: the runner's handle. It copies a run in as it is written (one transaction per
  unit) and, by ADR-038, never stops a training run: if the database cannot be written it warns
  once and stops mirroring that run, and ``nd db check`` / ``nd db rebuild`` find and repair it.
- :func:`rebuild`: empty every table and copy every run folder in again (``nd db rebuild``).
- :func:`check`: build a fresh copy from the folders in memory and list how the database
  differs from it (``nd db check``).

Connections use autocommit mode; every change is wrapped in an explicit transaction, so a
crash leaves either all of a unit's rows or none. WAL mode lets a reader (the server) read while
a training run writes.
"""

import json
import sqlite3
import warnings
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neurodrive.core.nn.io import load_network
from neurodrive.experiment.exam import ExamResult
from neurodrive.experiment.store import RunStore

DB_VERSION = 1
DB_NAME = "neurodrive.db"

_EPISODE_COLUMNS = (
    "status",
    "steps",
    "time_s",
    "progress_m",
    "progress_frac",
    "laps",
    "completed",
    "crashed",
    "lap_times_s",
    "best_lap_s",
    "mean_speed",
    "smoothness",
    "time_to_crash_s",
    "end_s_m",
)

_SCHEMA = (
    """CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        run_version INTEGER NOT NULL,
        method TEXT NOT NULL,
        tag TEXT NOT NULL,
        status TEXT NOT NULL,
        stop_reason TEXT,
        started_at TEXT,
        ended_at TEXT,
        seed INTEGER,
        sweep TEXT,
        git_commit TEXT,
        git_dirty INTEGER,
        budget TEXT,
        exams INTEGER,
        exam_wall_clock_s REAL,
        resumes INTEGER NOT NULL,
        config TEXT NOT NULL,
        meta TEXT NOT NULL
    )""",
    """CREATE TABLE metrics (
        run_id TEXT NOT NULL REFERENCES runs (run_id),
        unit INTEGER NOT NULL,
        name TEXT NOT NULL,
        value REAL,
        PRIMARY KEY (run_id, unit, name)
    ) WITHOUT ROWID""",
    """CREATE TABLE checkpoints (
        run_id TEXT NOT NULL REFERENCES runs (run_id),
        unit INTEGER NOT NULL,
        layer_sizes TEXT NOT NULL,
        n_parameters INTEGER NOT NULL,
        weights BLOB NOT NULL,
        PRIMARY KEY (run_id, unit)
    )""",
    """CREATE TABLE exams (
        run_id TEXT NOT NULL REFERENCES runs (run_id),
        unit INTEGER NOT NULL,
        exam_version INTEGER NOT NULL,
        env_steps INTEGER NOT NULL,
        policy TEXT NOT NULL,
        training_completion_rate REAL,
        training_crash_rate REAL,
        training_progress_frac_median REAL,
        heldout_completion_rate REAL,
        heldout_crash_rate REAL,
        heldout_progress_frac_median REAL,
        training TEXT NOT NULL,
        heldout TEXT NOT NULL,
        PRIMARY KEY (run_id, unit)
    )""",
    """CREATE TABLE exam_episodes (
        run_id TEXT NOT NULL,
        unit INTEGER NOT NULL,
        track TEXT NOT NULL,
        start_index INTEGER NOT NULL,
        split TEXT NOT NULL,
        start_s0_m REAL,
        start_lateral_m REAL,
        status TEXT,
        steps INTEGER,
        time_s REAL,
        progress_m REAL,
        progress_frac REAL,
        laps INTEGER,
        completed INTEGER,
        crashed INTEGER,
        lap_times_s TEXT,
        best_lap_s REAL,
        mean_speed REAL,
        smoothness REAL,
        time_to_crash_s REAL,
        end_s_m REAL,
        PRIMARY KEY (run_id, unit, track, start_index),
        FOREIGN KEY (run_id, unit) REFERENCES exams (run_id, unit)
    )""",
)

# Each table and the number of leading columns that form its primary key; children after
# parents (the order rows are inserted in), so deleting goes the other way round.
TABLES = {"runs": 1, "metrics": 3, "checkpoints": 2, "exams": 2, "exam_episodes": 4}


class DatabaseVersionError(sqlite3.DatabaseError):
    """The database file was made by a different schema version; ``nd db rebuild`` replaces it."""


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """All statements inside happen together or not at all."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _create_schema(conn: sqlite3.Connection) -> None:
    for statement in _SCHEMA:
        conn.execute(statement)
    conn.execute(f"PRAGMA user_version = {DB_VERSION}")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the database at ``path``, with the db v1 tables.

    A new or empty file gets the schema; a file of another schema version raises
    :class:`DatabaseVersionError` (use :func:`rebuild`). ``":memory:"`` gives a private database
    in memory (used by :func:`check`).
    """
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=5.0)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        if str(path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        tables = conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()
        if version == 0 and tables[0] == 0:
            with _transaction(conn):
                _create_schema(conn)
        elif version != DB_VERSION:
            raise DatabaseVersionError(
                f"the database has schema version {version}, this code uses {DB_VERSION}; "
                "rebuild it with: nd db rebuild"
            )
    except BaseException:
        conn.close()
        raise
    return conn


# --- copying one run in ---------------------------------------------------------------------


def _json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, allow_nan=False, default=str)


def _values(name: str, value: Any) -> list[tuple[str, Any]]:
    """One metric as (name, value) rows: a list gives ``name[0]``, ``name[1]``, ...; a dict gives
    ``name.key``; booleans become 0.0 / 1.0, numbers REAL, None NULL, text stays text."""
    if isinstance(value, list | tuple):
        return [row for i, v in enumerate(value) for row in _values(f"{name}[{i}]", v)]
    if isinstance(value, dict):
        return [row for k, v in value.items() for row in _values(f"{name}.{k}", v)]
    if isinstance(value, bool | int | float):
        return [(name, float(value))]
    if value is None or isinstance(value, str):
        return [(name, value)]
    raise ValueError(f"metric {name!r} has a value SQLite cannot hold: {value!r}")


def _optional(kind: type, value: Any) -> Any:
    return None if value is None else kind(value)


_RUN_COLUMNS = (
    "run_id",
    "run_version",
    "method",
    "tag",
    "status",
    "stop_reason",
    "started_at",
    "ended_at",
    "seed",
    "sweep",
    "git_commit",
    "git_dirty",
    "budget",
    "exams",
    "exam_wall_clock_s",
    "resumes",
    "config",
    "meta",
)
# Insert, or update in place (not "INSERT OR REPLACE", which deletes the old row first and so
# would orphan the run's metrics, checkpoints and exams).
_UPSERT_RUN = (
    f"INSERT INTO runs VALUES ({', '.join('?' * len(_RUN_COLUMNS))}) ON CONFLICT (run_id) "
    + "DO UPDATE SET "
    + ", ".join(f"{c} = excluded.{c}" for c in _RUN_COLUMNS[1:])
)


def _upsert_run(conn: sqlite3.Connection, store: RunStore) -> None:
    meta, config = store.read_meta(), store.read_config() or {}
    seed = config.get("seed") if isinstance(config, dict) else None
    exams = meta.get("exams")
    conn.execute(
        _UPSERT_RUN,
        (
            meta["run_id"],
            meta.get("run_version", 0),
            meta["method"],
            meta["tag"],
            meta["status"],
            meta.get("stop_reason"),
            meta.get("started_at"),
            meta.get("ended_at"),
            seed if isinstance(seed, int) else None,
            meta.get("sweep"),
            meta.get("git_commit"),
            _optional(int, meta.get("git_dirty")),
            _json(meta["budget"]) if meta.get("budget") is not None else None,
            _optional(int, exams),
            _optional(float, meta.get("exam_wall_clock_s")),
            len(meta.get("resumed_at") or []),
            _json(config),
            _json(meta),
        ),
    )


def _insert_metrics(conn: sqlite3.Connection, run_id: str, record: dict[str, Any]) -> None:
    unit = int(record["unit"])
    rows = [
        (run_id, unit, key, v)
        for name, value in record.items()
        if name != "unit"
        for key, v in _values(name, value)
    ]
    conn.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?)", rows)


def _insert_checkpoint(conn: sqlite3.Connection, store: RunStore, unit: int) -> None:
    path = store.path / "checkpoints" / f"unit_{unit}.npz"
    spec = load_network(path).spec
    sizes = _json(list(spec.layer_sizes))
    # A checkpoint saved again (unit 0 after a kill in unit 1) replaces its file, so its row too.
    conn.execute("DELETE FROM checkpoints WHERE run_id = ? AND unit = ?", (store.path.name, unit))
    conn.execute(
        "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?)",
        (store.path.name, unit, sizes, spec.n_parameters, path.read_bytes()),
    )


def _insert_exam(conn: sqlite3.Connection, run_id: str, unit: int, card: ExamResult) -> None:
    t, h = card.training, card.heldout
    for table in ("exam_episodes", "exams"):  # an exam taken again replaces the old card
        conn.execute(f"DELETE FROM {table} WHERE run_id = ? AND unit = ?", (run_id, unit))
    conn.execute(
        "INSERT INTO exams VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            unit,
            card.exam_version,
            int(card.env_steps),
            _json(card.policy),
            float(t["completion_rate"]),
            float(t["crash_rate"]),
            float(t["progress_frac_median"]),
            float(h["completion_rate"]),
            float(h["crash_rate"]),
            float(h["progress_frac_median"]),
            _json(t),
            _json(h),
        ),
    )
    rows = []
    for part in card.tracks:
        for e in part.episodes:
            values = [
                _json(list(e[c])) if c == "lap_times_s" else e.get(c) for c in _EPISODE_COLUMNS
            ]
            values = [int(v) if isinstance(v, bool) else v for v in values]
            rows.append(
                (
                    run_id,
                    unit,
                    part.track,
                    int(e["start_index"]),
                    part.split,
                    _optional(float, e.get("start_s0_m")),
                    _optional(float, e.get("start_lateral_m")),
                    *values,
                )
            )
    marks = ", ".join("?" * (7 + len(_EPISODE_COLUMNS)))
    conn.executemany(f"INSERT INTO exam_episodes VALUES ({marks})", rows)


def _delete_run(conn: sqlite3.Connection, run_id: str) -> None:
    for table in reversed(TABLES):
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))


def _copy_run(conn: sqlite3.Connection, store: RunStore) -> None:
    """Insert everything of one run folder: its row, metrics, checkpoints and exam cards."""
    run_id = store.path.name
    _upsert_run(conn, store)
    for record in store.read_metrics():
        _insert_metrics(conn, run_id, record)
    for unit in store.checkpoint_units():
        _insert_checkpoint(conn, store, unit)
    for unit in store.exam_units():
        _insert_exam(conn, run_id, unit, store.load_exam(unit))


# --- the runner's mirror ----------------------------------------------------------------------


class Mirror:
    """Keeps the database in step with one run as the runner writes its folder.

    Every method is safe to call: the first database error (locked for more than 5 s, disk
    full, an old schema, ...) gives one warning and switches the mirror off for the rest of the
    run. The run folder, the master copy, is unaffected (ADR-038).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.conn: sqlite3.Connection | None = None
        self._guard(lambda: setattr(self, "conn", connect(self.path)))

    @property
    def active(self) -> bool:
        return self.conn is not None

    def _guard(self, action: Any) -> None:
        try:
            action()
        except (sqlite3.Error, OSError, ValueError, KeyError) as exc:
            warnings.warn(
                f"the database mirror {self.path.name} stopped for this run ({exc}); the run "
                "folder is complete. Repair the mirror with: nd db rebuild",
                RuntimeWarning,
                stacklevel=3,
            )
            self.close()

    def _write(self, action: Any) -> None:
        if self.conn is None:
            return
        conn = self.conn

        def run() -> None:
            with _transaction(conn):
                action(conn)

        self._guard(run)

    def sync_run(self, store: RunStore) -> None:
        """Replace everything the database holds about this run with the folder's content."""

        def copy(conn: sqlite3.Connection) -> None:
            _delete_run(conn, store.path.name)
            _copy_run(conn, store)

        self._write(copy)

    def add_unit(
        self,
        store: RunStore,
        record: dict[str, Any] | None,
        checkpoints: list[int],
        exams: list[tuple[int, ExamResult]],
    ) -> None:
        """Add one unit's metrics line, and the checkpoints and exam cards just saved."""
        run_id = store.path.name

        def add(conn: sqlite3.Connection) -> None:
            if record is not None:
                _insert_metrics(conn, run_id, record)
            for unit in checkpoints:
                _insert_checkpoint(conn, store, unit)
            for unit, card in exams:
                _insert_exam(conn, run_id, unit, card)

        self._write(add)

    def update_run(self, store: RunStore) -> None:
        """Copy the run's meta.json and config again (status, stop reason, end time, ...)."""
        self._write(lambda conn: _upsert_run(conn, store))

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


# --- rebuild and check ------------------------------------------------------------------------


def run_folders(runs_dir: str | Path) -> list[Path]:
    """Every run folder (a folder with a meta.json) directly inside ``runs_dir``, by name."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted(p for p in runs_dir.iterdir() if (p / "meta.json").is_file())


@dataclass
class RebuildSummary:
    """What a rebuild copied: rows per table, and the folders it could not read (and why)."""

    rows: dict[str, int]
    skipped: list[tuple[str, str]] = field(default_factory=list)


def _copy_all(conn: sqlite3.Connection, runs_dir: str | Path) -> list[tuple[str, str]]:
    """Copy every run folder in; a folder that cannot be read is rolled back and listed."""
    skipped = []
    for path in run_folders(runs_dir):
        conn.execute("SAVEPOINT one_run")
        try:
            _copy_run(conn, RunStore.open(path))
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            conn.execute("ROLLBACK TO one_run")
            skipped.append((path.name, f"{type(exc).__name__}: {exc}"))
        conn.execute("RELEASE one_run")
    return skipped


def _counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES}


def rebuild(runs_dir: str | Path, db_path: str | Path | None = None) -> RebuildSummary:
    """Empty the database (any schema version) and copy every run folder in again.

    One transaction: a reader sees the old content until the new one is complete.
    """
    db_path = Path(db_path) if db_path is not None else Path(runs_dir) / DB_NAME
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=5.0)
    with closing(conn):
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = OFF")  # tables are dropped in any order
        with _transaction(conn):
            names = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            for (name,) in names.fetchall():
                conn.execute(f'DROP TABLE "{name}"')
            _create_schema(conn)
            skipped = _copy_all(conn, runs_dir)
        conn.execute("PRAGMA foreign_keys = ON")
        problems = conn.execute("PRAGMA foreign_key_check").fetchall()
        if problems:
            raise sqlite3.IntegrityError(f"rows without a parent after rebuild: {problems[:3]}")
        return RebuildSummary(rows=_counts(conn), skipped=skipped)


def from_folders(runs_dir: str | Path) -> tuple[sqlite3.Connection, list[tuple[str, str]]]:
    """A fresh database in memory, copied from the folders, and the folders it skipped."""
    conn = connect(":memory:")
    with _transaction(conn):
        skipped = _copy_all(conn, runs_dir)
    return conn, skipped


def dump(conn: sqlite3.Connection) -> dict[str, list[tuple[Any, ...]]]:
    """Every row of every table, ordered by primary key (to compare two databases)."""
    out = {}
    for table, n_key in TABLES.items():
        names = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        order = ", ".join(names[:n_key])
        out[table] = conn.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()
    return out


def _by_run(rows: list[tuple[Any, ...]], n_key: int) -> dict[str, dict[tuple, tuple]]:
    grouped: dict[str, dict[tuple, tuple]] = {}
    for row in rows:
        grouped.setdefault(row[0], {})[row[:n_key]] = row
    return grouped


def check(runs_dir: str | Path, db_path: str | Path | None = None) -> list[str]:
    """How the database differs from the folders, as readable lines; empty when they match.

    The folders are copied into a fresh database in memory (exactly as :func:`rebuild` would),
    and each table is compared with the file run by run, row by row on the primary key.
    """
    db_path = Path(db_path) if db_path is not None else Path(runs_dir) / DB_NAME
    if not db_path.is_file():
        return [f"no database at {db_path.name}; build it with: nd db rebuild"]
    expected, skipped = from_folders(runs_dir)
    lines = [f"cannot read run folder {name}: {reason}" for name, reason in skipped]
    try:
        actual = connect(db_path)
    except sqlite3.Error as exc:
        expected.close()
        return [*lines, str(exc)]
    with closing(expected), closing(actual):
        want, have = dump(expected), dump(actual)
    for table, n_key in TABLES.items():
        w, h = _by_run(want[table], n_key), _by_run(have[table], n_key)
        if table == "runs":
            for run_id in sorted(set(w) | set(h)):
                if run_id not in h:
                    lines.append(f"runs: {run_id} is in the folders but not in the database")
                elif run_id not in w:
                    lines.append(f"runs: {run_id} is in the database but has no folder")
                elif w[run_id] != h[run_id]:
                    lines.append(f"runs: {run_id}: its row differs from meta.json / config.yaml")
            continue
        for run_id in sorted(set(w) | set(h)):
            rw, rh = w.get(run_id, {}), h.get(run_id, {})
            missing = sum(k not in rh for k in rw)
            extra = sum(k not in rw for k in rh)
            changed = sum(k in rh and rh[k] != row for k, row in rw.items())
            if missing or extra or changed:
                lines.append(
                    f"{table}: {run_id}: {missing} rows missing, {extra} extra, {changed} different"
                )
    return lines


# --- an example query -------------------------------------------------------------------------

# The median, over the runs of one sweep, of each run's last exam's held-out median progress.
# Same rule as stats.percentile(values, 50): sort the n values, h = (n - 1) * 0.5,
# lo = floor(h), hi = ceil(h), median = v[lo] + (h - lo) * (v[hi] - v[lo]).
MEDIAN_FINAL_HELDOUT_PROGRESS = """
WITH final AS (
    SELECT e.heldout_progress_frac_median AS v
    FROM exams AS e JOIN runs AS r ON r.run_id = e.run_id
    WHERE r.sweep = :sweep
      AND e.unit = (SELECT max(unit) FROM exams WHERE run_id = e.run_id)
),
ranked AS (
    SELECT v, row_number() OVER (ORDER BY v) - 1 AS i, count(*) OVER () AS n FROM final
),
ends AS (
    SELECT max(CASE WHEN i = (n - 1) / 2 THEN v END) AS a,
           max(CASE WHEN i = n / 2 THEN v END) AS b,
           max((n - 1) * 0.5 - (n - 1) / 2) AS frac
    FROM ranked
)
SELECT a + frac * (b - a) FROM ends
"""


def median_final_heldout_progress(conn: sqlite3.Connection, sweep: str) -> float | None:
    """Median across the sweep's runs (seeds) of the held-out median progress at each run's
    last examined checkpoint; None if no run of the sweep has an exam."""
    return conn.execute(MEDIAN_FINAL_HELDOUT_PROGRESS, {"sweep": sweep}).fetchone()[0]
