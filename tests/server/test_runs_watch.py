"""Tests for the run watcher (task 5.8, ADR-041): a run folder is written step by step by the
test, the way a training process writes it, and each poll must report exactly what is new."""

import json

import pytest

from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.experiment.store import RunStore
from neurodrive.server.runs_watch import RunsWatcher
from neurodrive.server.schemas import (
    CheckpointMessage,
    ExamDoneMessage,
    RunMetricsMessage,
    RunStatusMessage,
)

LINE = 40  # every test line below is exactly 40 bytes long, newline included


def line(unit: int) -> bytes:
    """A metrics line of exactly 40 bytes for ``unit`` (padded inside the JSON)."""
    text = json.dumps({"unit": unit, "loss": 0.5, "pad": ""})
    text = text.replace('"pad": ""', '"pad": "' + "x" * (LINE - 1 - len(text)) + '"')
    data = (text + "\n").encode()
    assert len(data) == LINE
    return data


def write(store: RunStore, data: bytes) -> None:
    with open(store.path / "metrics.jsonl", "ab") as f:
        f.write(data)


def card(completion: float) -> dict:
    part = {"completion_rate": completion, "crash_rate": 0.25, "progress_frac_median": 0.6}
    return {"exam_version": 1, "training": part, "heldout": {**part, "completion_rate": 0.0}}


def kinds(events) -> list[str]:
    return [e.type for e in events]


def metrics_of(events) -> list[RunMetricsMessage]:
    return [e for e in events if isinstance(e, RunMetricsMessage)]


@pytest.fixture
def running(tmp_path) -> tuple[RunsWatcher, RunStore]:
    """A watcher that took stock of an empty folder, and a run created after that."""
    watcher = RunsWatcher(tmp_path)
    assert watcher.poll() == []  # the first poll only takes stock
    store = RunStore.create(tmp_path, "walk", "t", {"seed": 0})
    store.update_meta(budget={"units": 3, "env_steps": None, "wall_clock_s": None})
    return watcher, store


def test_a_new_run_is_announced_and_its_lines_are_sent_once(running):
    watcher, store = running
    events = watcher.poll()
    assert kinds(events) == ["metrics", "run_status"]  # tracking starts with the (empty) history
    (status,) = [e for e in events if isinstance(e, RunStatusMessage)]
    assert (status.run_id, status.status, status.units_done) == (store.path.name, "running", 0)
    assert status.budget_units == 3 and not status.launched and status.exit_code is None
    assert metrics_of(events)[0].replace and metrics_of(events)[0].records == []

    write(store, line(1) + line(2))
    (sent,) = metrics_of(watcher.poll())
    assert not sent.replace and [r["unit"] for r in sent.records] == [1, 2]
    assert watcher.poll() == []  # nothing new, nothing sent (the status is unchanged too)


def test_a_half_written_line_waits_for_its_newline(running):
    # The Brief's worked example: two 40-byte lines (offset 80), then 25 bytes of a third.
    watcher, store = running
    watcher.poll()
    write(store, line(1) + line(2))
    watcher.poll()
    tracked = watcher._tracked[store.path.name]
    assert tracked.offset == 80
    third = line(3)
    write(store, third[:25])  # n = 105: no newline after byte 80
    assert metrics_of(watcher.poll()) == [] and tracked.offset == 80
    write(store, third[25:])  # n = 120
    (sent,) = metrics_of(watcher.poll())
    assert [r["unit"] for r in sent.records] == [3] and tracked.offset == 120


def test_checkpoints_and_exam_cards_are_reported_after_their_metrics_line(running):
    watcher, store = running
    watcher.poll()
    store.append_metrics({"unit": 1, "loss": 0.4})
    store.save_checkpoint(1, init_network(NetworkSpec(), make_rng(0)))
    (store.path / "exams").mkdir()
    (store.path / "exams" / "unit_1.json").write_text(json.dumps(card(0.5)), encoding="utf-8")
    events = watcher.poll()
    assert kinds(events) == ["metrics", "checkpoint", "exam_done", "run_status"]
    checkpoint = next(e for e in events if isinstance(e, CheckpointMessage))
    assert checkpoint.units == [1]
    exam = next(e for e in events if isinstance(e, ExamDoneMessage))
    assert exam.unit == 1 and exam.training.completion_rate == 0.5
    assert exam.heldout.completion_rate == 0.0 and exam.training.progress_frac_median == 0.6
    assert watcher.poll() == []  # each card once


def test_a_half_written_exam_card_is_read_on_a_later_poll(running):
    watcher, store = running
    watcher.poll()
    (store.path / "exams").mkdir()
    path = store.path / "exams" / "unit_0.json"
    text = json.dumps(card(1.0))
    path.write_text(text[:30], encoding="utf-8")
    assert "exam_done" not in kinds(watcher.poll())
    path.write_text(text, encoding="utf-8")
    assert kinds(watcher.poll()) == ["exam_done"]


def test_a_finished_run_gets_its_last_lines_and_status_and_is_no_longer_tailed(running):
    watcher, store = running
    watcher.poll()
    write(store, line(1) + line(2) + line(3))
    store.finish("completed", "units")
    events = watcher.poll()
    assert kinds(events) == ["metrics", "run_status"]
    assert len(metrics_of(events)[0].records) == 3
    status = events[-1]
    assert (status.status, status.stop_reason, status.units_done) == ("completed", "units", 3)
    assert store.path.name not in watcher._tracked
    write(store, line(4))  # nobody writes after the end; if they did, it is not followed
    assert watcher.poll() == []


def test_a_resumed_run_is_tracked_again_from_its_cut_back_folder(running):
    watcher, store = running
    watcher.poll()
    for unit in (1, 2, 3):
        store.append_metrics({"unit": unit, "loss": 0.1 * unit})
    watcher.poll()
    store.finish("failed", "error")
    watcher.poll()
    # What runner.resume does: cut back to unit 1, status running again, one more resume time.
    store.cut_back(1)
    store.update_meta(status="running", ended_at=None, resumed_at=["2026-11-02T00:00:00+00:00"])
    events = watcher.poll()
    (sent,) = metrics_of(events)
    assert sent.replace and [r["unit"] for r in sent.records] == [1]
    assert events[-1].status == "running" and events[-1].units_done == 1


def test_a_killed_and_resumed_run_still_running_starts_again_too(running):
    # Killed (status stays running), then resumed: meta changes only by its resume list.
    watcher, store = running
    watcher.poll()
    for unit in (1, 2, 3):
        store.append_metrics({"unit": unit, "loss": 0.1 * unit})
    watcher.poll()
    store.cut_back(2)
    store.append_metrics({"unit": 3, "loss": 0.99})  # the recomputed unit 3, same length or not
    store.update_meta(resumed_at=["2026-11-02T00:00:00+00:00"])
    (sent,) = metrics_of(watcher.poll())
    assert sent.replace and [r["loss"] for r in sent.records] == [0.1, 0.2, 0.99]


def test_history_continues_exactly_where_the_next_poll_starts(running):
    watcher, store = running
    watcher.poll()
    write(store, line(1) + line(2))
    watcher.poll()
    history = watcher.history(store.path.name)
    assert kinds(history) == ["metrics", "checkpoint"]
    assert history[0].replace and [r["unit"] for r in history[0].records] == [1, 2]
    write(store, line(3))
    (sent,) = metrics_of(watcher.poll())
    assert [r["unit"] for r in sent.records] == [3]  # not 1, 2 again


def test_history_of_a_running_run_not_yet_tracked_then_every_line_follows(tmp_path):
    watcher = RunsWatcher(tmp_path)
    watcher.poll()
    store = RunStore.create(tmp_path, "walk", "t", {"seed": 0})
    write(store, line(1))
    history = watcher.history(store.path.name)  # asked for before any poll saw the run
    assert history[0].replace and history[0].records == []
    (sent,) = metrics_of(watcher.poll())
    assert not sent.replace and [r["unit"] for r in sent.records] == [1]


def test_history_of_a_finished_run_comes_from_its_folder(tmp_path):
    store = RunStore.create(tmp_path, "walk", "t", {"seed": 0})
    store.append_metrics({"unit": 1, "loss": 0.3})
    store.save_checkpoint(1, init_network(NetworkSpec(), make_rng(0)))
    (store.path / "exams").mkdir()
    (store.path / "exams" / "unit_1.json").write_text(json.dumps(card(0.1)), encoding="utf-8")
    store.finish("completed", "units")
    watcher = RunsWatcher(tmp_path)
    watcher.poll()
    history = watcher.history(store.path.name)
    assert kinds(history) == ["metrics", "checkpoint", "exam_done"]
    assert history[0].records == [{"unit": 1, "loss": 0.3}] and history[1].units == [1]
    for bad in ("nope", "../x", "20261102-000000-walk-missing"):
        with pytest.raises(KeyError):
            watcher.history(bad)


def test_the_first_poll_takes_stock_and_finished_runs_stay_quiet(tmp_path):
    done = RunStore.create(tmp_path, "walk", "done", {"seed": 0})
    done.finish("completed", "units")
    active = RunStore.create(tmp_path, "walk", "active", {"seed": 0})
    active.append_metrics({"unit": 1, "loss": 0.2})
    watcher = RunsWatcher(tmp_path)
    assert watcher.poll() == []
    assert [s.run_id for s in watcher.statuses()] == [active.path.name]  # running runs only
    assert watcher.statuses()[0].units_done == 1
    assert watcher.poll() == []


def test_a_stop_request_shows_in_the_status(running):
    watcher, store = running
    watcher.poll()
    store.request_stop()
    (status,) = watcher.poll()
    assert status.stop_requested
    store.clear_stop_request()
    store.finish("completed", "stopped")
    (status,) = watcher.poll()
    assert (status.stop_requested, status.stop_reason) == (False, "stopped")


def test_a_half_written_meta_is_ignored_until_it_is_complete(running):
    watcher, store = running
    watcher.poll()
    text = (store.path / "meta.json").read_text(encoding="utf-8")
    (store.path / "meta.json").write_text(text[:20], encoding="utf-8")
    assert watcher.poll() == []  # the old meta is kept, nothing breaks
    meta = json.loads(text)
    meta["status"] = "completed"
    (store.path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert watcher.poll()[-1].status == "completed"
