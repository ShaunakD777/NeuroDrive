"""Tests for the SQLite mirror (task 5.5b, ADR-007, ADR-038; db v2, task 5.10, ADR-043): the
database written unit by unit equals one rebuilt from the folders, a rebuild after deleting it
loses nothing, check finds differences, a query across seeds gives the same answer as reading the
folders, and the ``studies`` table is derived from the runs' ``study`` field."""

import json
import shutil
import sqlite3
import warnings
from contextlib import closing
from pathlib import Path

import pytest
from walk_learner import KILLS, SEED, UNITS, WalkLearner, run_killed

from neurodrive.core.nn.io import load_network
from neurodrive.experiment import db
from neurodrive.experiment.config import load_config
from neurodrive.experiment.runner import Budget, resume, run, train_toy_run
from neurodrive.experiment.stats import percentile
from neurodrive.experiment.store import RunStore, check_study

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def walk_run(runs_dir, seed=SEED, units=UNITS, exams=True, mirror=True, **meta) -> RunStore:
    store = RunStore.create(runs_dir, "walk", f"s{seed}", {"seed": seed}, extra_meta=meta)
    run(WalkLearner(seed), store, Budget(units=units), exams=exams, mirror=mirror)
    return store


def database(runs_dir) -> sqlite3.Connection:
    return db.connect(Path(runs_dir) / db.DB_NAME)


def dump_file(runs_dir) -> dict:
    with closing(database(runs_dir)) as conn:
        return db.dump(conn)


def dump_folders(runs_dir) -> dict:
    conn, skipped = db.from_folders(runs_dir)
    with closing(conn):
        assert skipped == []
        return db.dump(conn)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    """A runs folder written with the mirror on: a 12-unit walk run with exams and a toy run.

    Any mirror warning is turned into an error: in normal use the mirror never gives up.
    """
    runs_dir = tmp_path_factory.mktemp("runs")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        walk_run(runs_dir)
        train_toy_run(load_config(CONFIGS / "toy_xor.yaml", epochs=20), runs_dir)
    return runs_dir


def test_a_new_database_has_the_v2_tables_in_wal_mode(tmp_path):
    with closing(db.connect(tmp_path / "x.db")) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == set(db.TABLES)
        assert "studies" in tables
        assert conn.execute("PRAGMA user_version").fetchone()[0] == db.DB_VERSION == 2
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_the_incremental_mirror_holds_what_the_folders_hold(runs, tmp_path):
    names = [p.name for p in db.run_folders(runs)]
    (walk,), (toy,) = [n for n in names if "-walk-" in n], [n for n in names if "-toy" in n]
    rows = dump_file(runs)
    walk_metrics = [r for r in rows["metrics"] if r[0] == walk]
    # 12 units × 7 values (env, demo and teacher totals, wall clock, fitness, accepted, diverged)
    names = {r[2] for r in walk_metrics}
    assert names == {
        "env_steps_total",
        "demo_frames_total",
        "teacher_queries_total",
        "wall_clock_s",
        "fitness",
        "accepted",
        "diverged",
    }
    assert len(walk_metrics) == UNITS * len(names)
    assert [r[1] for r in rows["checkpoints"] if r[0] == walk] == [0, 1, 2, 5, 10, 12]
    assert [r[1] for r in rows["exams"] if r[0] == walk] == [0, 1, 2, 5, 10, 12]
    assert len([r for r in rows["exam_episodes"] if r[0] == walk]) == 6 * 50
    assert [r[1] for r in rows["exams"] if r[0] == toy] == []  # toy networks never drive

    # every metrics value is the JSON line's value; lists became one row per element
    store = RunStore.open(runs / toy)
    value = {(r[1], r[2]): r[3] for r in rows["metrics"] if r[0] == toy}
    for line in store.read_metrics():
        assert value[(line["unit"], "loss")] == line["loss"]
        for i, g in enumerate(line["grad_norm"]):
            assert value[(line["unit"], f"grad_norm[{i}]")] == g
        assert value[(line["unit"], "diverged")] == 0.0

    # the unit-5 weights BLOB is the saved network
    blob = next(r[4] for r in rows["checkpoints"] if r[0] == walk and r[1] == 5)
    (tmp_path / "from_db.npz").write_bytes(blob)
    network = load_network(tmp_path / "from_db.npz")
    assert network.equals(RunStore.open(runs / walk).load_checkpoint(5))

    # the run row is meta.json
    run_row = next(r for r in rows["runs"] if r[0] == walk)
    meta = RunStore.open(runs / walk).read_meta()
    assert run_row[4] == "completed" and run_row[5] == "units" and run_row[8] == SEED
    assert json.loads(run_row[-1]) == meta


def test_rebuilt_equals_incremental_and_deleting_loses_nothing(runs, tmp_path):
    incremental = dump_file(runs)
    assert incremental == dump_folders(runs)
    copy = tmp_path / "rebuilt.db"
    summary = db.rebuild(runs, copy)
    assert summary.skipped == []
    assert summary.rows == {t: len(incremental[t]) for t in db.TABLES}
    with closing(db.connect(copy)) as conn:
        assert db.dump(conn) == incremental
    copy.unlink()
    for side in ("-wal", "-shm"):
        Path(f"{copy}{side}").unlink(missing_ok=True)
    db.rebuild(runs, copy)  # deleted, then rebuilt from the folders alone
    with closing(db.connect(copy)) as conn:
        assert db.dump(conn) == incremental


@pytest.mark.parametrize("kill", list(KILLS))
def test_a_killed_and_resumed_run_mirrors_exactly(kill, tmp_path):
    store = run_killed(tmp_path, kill)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resume(WalkLearner(SEED), store) == "completed"
    assert db.check(tmp_path) == []
    assert dump_file(tmp_path) == dump_folders(tmp_path)


def test_check_reports_each_kind_of_difference_and_rebuild_repairs_it(tmp_path):
    a = walk_run(tmp_path, seed=1, units=3, exams=False)
    b = walk_run(tmp_path, seed=2, units=3, exams=False)
    assert db.check(tmp_path) == []
    with closing(database(tmp_path)) as conn:
        conn.execute("DELETE FROM metrics WHERE run_id = ? AND unit = 2", (a.path.name,))
    c = walk_run(tmp_path, seed=3, units=3, exams=False, mirror=False)  # never mirrored
    shutil.rmtree(b.path)  # a run folder deleted by hand
    problems = db.check(tmp_path)
    assert f"metrics: {a.path.name}: 7 rows missing, 0 extra, 0 different" in problems
    assert f"runs: {c.path.name} is in the folders but not in the database" in problems
    assert f"runs: {b.path.name} is in the database but has no folder" in problems
    db.rebuild(tmp_path)
    assert db.check(tmp_path) == []


def test_check_without_a_database_says_so(tmp_path):
    assert db.check(tmp_path) == ["no database at neurodrive.db; build it with: nd db rebuild"]


def test_median_across_seeds_is_the_same_from_sql_and_from_the_folders(tmp_path):
    seeds = [0, 1, 2, 3, 4]
    stores = [walk_run(tmp_path, seed=s, units=1, sweep="sw-walk") for s in seeds]
    walk_run(tmp_path, seed=9, units=1, sweep="another")  # must not count
    finals = []
    for store in stores:  # from the folders: each run's last exam card
        card = store.load_exam(max(store.exam_units()))
        finals.append(card.heldout["progress_frac_median"])
    assert len(set(finals)) > 1  # different seeds, different scores: not a trivial median
    with closing(database(tmp_path)) as conn:
        assert db.median_final_heldout_progress(conn, "sw-walk") == percentile(finals, 50)
        assert db.median_final_heldout_progress(conn, "no-such-sweep") is None


def test_the_sql_median_rule_on_the_worked_example(tmp_path):
    with closing(db.connect(":memory:")) as conn:
        for i, v in enumerate([0.02, 0.05, 0.03, 0.08]):
            meta = {"run_id": f"r{i}", "method": "m", "tag": "t", "status": "completed"}
            conn.execute(
                "INSERT INTO runs (run_id, run_version, method, tag, status, sweep, resumes,"
                " config, meta) VALUES (?, 1, 'm', 't', 'completed', 'w', 0, '{}', ?)",
                (f"r{i}", json.dumps(meta)),
            )
            conn.execute(
                "INSERT INTO exams (run_id, unit, exam_version, env_steps, policy,"
                " heldout_progress_frac_median, training, heldout)"
                " VALUES (?, 3, 1, 0, '{}', ?, '{}', '{}')",
                (f"r{i}", v),
            )
        median = db.median_final_heldout_progress(conn, "w")
    assert median == percentile([0.02, 0.05, 0.03, 0.08], 50)
    assert median == pytest.approx(0.04)


def test_a_mirror_that_cannot_be_written_warns_and_the_run_still_completes(tmp_path):
    (tmp_path / db.DB_NAME).mkdir()  # a folder where the database file should be
    with pytest.warns(RuntimeWarning, match="nd db rebuild"):
        store = walk_run(tmp_path, units=3, exams=False)
    assert store.read_meta()["status"] == "completed" and len(store.read_metrics()) == 3


@pytest.mark.parametrize("version", [1, 99])  # 1: a db v1 file made before task 5.10
def test_an_old_schema_warns_and_rebuild_replaces_it(tmp_path, version):
    path = tmp_path / db.DB_NAME
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE runs (x)")
        conn.execute(f"PRAGMA user_version = {version}")
    with pytest.warns(RuntimeWarning, match=f"schema version {version}"):
        walk_run(tmp_path, units=2, exams=False)
    assert db.check(tmp_path) == [
        f"the database has schema version {version}, this code uses 2; rebuild it with: "
        "nd db rebuild"
    ]
    summary = db.rebuild(tmp_path)
    assert summary.rows["runs"] == 1 and summary.rows["metrics"] == 2 * 7
    assert db.check(tmp_path) == []


def test_rebuild_reads_version_0_folders_and_skips_unreadable_ones(tmp_path):
    old = RunStore.create(tmp_path, "toy", "old", {"seed": 4})
    old.append_metrics({"unit": 1, "loss": 0.5, "grad_norm": [1.0, 2.0]})
    old.finish("completed")
    old.update_meta(run_version=0)
    broken = RunStore.create(tmp_path, "toy", "broken", {})
    (broken.path / "meta.json").write_text("{not json", encoding="utf-8")
    summary = db.rebuild(tmp_path)
    assert summary.rows["runs"] == 1 and summary.rows["metrics"] == 3
    assert [name for name, _ in summary.skipped] == [broken.path.name]
    assert db.check(tmp_path)[0].startswith(f"cannot read run folder {broken.path.name}")


# --- studies (db v2, task 5.10, ADR-043) -------------------------------------------------------


def study_row(runs_dir, study) -> dict | None:
    with closing(database(runs_dir)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM studies WHERE study_id = ?", (study,)).fetchone()
        return dict(row) if row else None


@pytest.fixture
def studies(tmp_path) -> Path:
    """Runs of two studies and runs of none: walk seeds 1, 2 and a toy run in S9 (the walks in
    one sweep), walk seed 3 in S8, walk seed 4 in no study. Any mirror warning is an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        walk_run(tmp_path, seed=1, units=2, sweep="sw-a", study="S9")
        walk_run(tmp_path, seed=2, units=2, sweep="sw-a", study="S9")
        walk_run(tmp_path, seed=3, units=2, study="S8")
        walk_run(tmp_path, seed=4, units=2)
        toy = load_config(CONFIGS / "toy_xor.yaml", epochs=5)
        train_toy_run(toy, tmp_path, extra_meta={"study": "S9"})
    return tmp_path


def test_the_studies_table_is_derived_from_the_runs_study_field(studies):
    metas = [RunStore.open(p).read_meta() for p in db.run_folders(studies)]
    s9 = [m for m in metas if m.get("study") == "S9"]
    row = study_row(studies, "S9")
    assert row == {
        "study_id": "S9",
        "n_runs": 3,
        "methods": json.dumps(sorted({m["method"] for m in s9})),
        "sweeps": json.dumps(["sw-a"]),
        "git_commits": json.dumps(sorted({m["git_commit"] for m in s9} - {None})),
        "started_at": min(m["started_at"] for m in s9),
        "ended_at": max(m["ended_at"] for m in s9),
    }
    assert json.loads(row["methods"]) == ["toy_xor", "walk"]
    assert study_row(studies, "S8")["n_runs"] == 1
    assert study_row(studies, "S8")["sweeps"] == "[]"
    with closing(database(studies)) as conn:
        assert conn.execute("SELECT count(*) FROM studies").fetchone()[0] == 2  # no "None" row
        assert db.runs_of_study(conn, "S9") == sorted(m["run_id"] for m in s9)
        assert db.runs_of_study(conn, "S7") == []
        without = conn.execute("SELECT count(*) FROM runs WHERE study IS NULL").fetchone()[0]
        assert without == 1
    assert dump_file(studies) == dump_folders(studies)  # incremental = rebuilt, studies included


def test_a_running_run_leaves_its_study_open_until_it_ends(studies):
    store = RunStore.create(studies, "walk", "late", {"seed": 5}, extra_meta={"study": "S9"})
    mirror = db.Mirror(studies / db.DB_NAME)
    mirror.sync_run(store)
    assert study_row(studies, "S9")["n_runs"] == 4
    assert study_row(studies, "S9")["ended_at"] is None  # one run of S9 has not ended
    store.finish("completed", "units")
    mirror.update_run(store)  # the status change must refresh the study row
    mirror.close()
    assert study_row(studies, "S9")["ended_at"] == store.read_meta()["ended_at"]
    assert dump_file(studies) == dump_folders(studies)


def test_check_reports_study_rows_that_differ_and_rebuild_repairs_them(studies):
    assert db.check(studies) == []
    with closing(database(studies)) as conn:
        conn.execute("UPDATE studies SET n_runs = 99 WHERE study_id = 'S9'")
        conn.execute("INSERT INTO studies VALUES ('S0', 1, '[]', '[]', '[]', NULL, NULL)")
    (only_s8,) = [p for p in db.run_folders(studies) if p.name.endswith("walk-s3")]
    shutil.rmtree(only_s8)
    problems = db.check(studies)
    assert "studies: S9: its row differs from its runs' meta.json" in problems
    assert "studies: S0 has a row but no run folder of that study" in problems
    assert "studies: S8 has a row but no run folder of that study" in problems
    with closing(database(studies)) as conn:
        conn.execute("DELETE FROM studies WHERE study_id = 'S9'")
    assert "studies: S9 has runs in the folders but no row" in db.check(studies)
    db.rebuild(studies)
    assert db.check(studies) == []
    assert study_row(studies, "S8") is None and study_row(studies, "S9")["n_runs"] == 3


def test_resyncing_a_run_keeps_its_study_row(studies):
    (one,) = [p for p in db.run_folders(studies) if p.name.endswith("walk-s1")]
    before = study_row(studies, "S9")
    mirror = db.Mirror(studies / db.DB_NAME)
    mirror.sync_run(RunStore.open(one))  # delete the run's rows, then copy it again
    mirror.close()
    assert study_row(studies, "S9") == before


@pytest.mark.parametrize("bad", ["", "S 4", "S4/x", "../S4"])
def test_an_invalid_study_id_is_refused_before_a_folder_exists(tmp_path, bad):
    with pytest.raises(ValueError, match="study must be"):
        check_study(bad)
    with pytest.raises(ValueError, match="study must be"):
        RunStore.create(tmp_path, "walk", "s1", {}, extra_meta={"study": bad})
    assert list(tmp_path.iterdir()) == []
