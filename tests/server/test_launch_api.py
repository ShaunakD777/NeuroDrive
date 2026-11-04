"""Tests for launching runs from the app and following them (task 5.8, ADR-041, api v1.13).

These start real training processes (``python -m neurodrive train``), exactly as the server does
for the browser, and compare their run folders with runs trained directly.
"""

import json
import shutil
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neurodrive.experiment.config import load_config
from neurodrive.experiment.db import check
from neurodrive.experiment.runner import train_random_search_run, train_toy_run
from neurodrive.experiment.store import STOP_FILE, RunStore
from neurodrive.server.app import create_app
from neurodrive.server.runs_watch import Launcher

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

# A small random search: 3 candidates per unit, 2 units, exams on (each checkpoint is examined).
SMALL_SEARCH = """\
method: random_search
seed: 0
tag: small
search:
  candidates_per_unit: 3
budget:
  units: 2
"""


@pytest.fixture
def configs(tmp_path) -> Path:
    folder = tmp_path / "configs"
    folder.mkdir()
    shutil.copy(CONFIGS / "toy_xor.yaml", folder / "toy_xor.yaml")
    (folder / "small_search.yaml").write_text(SMALL_SEARCH, encoding="utf-8")
    (folder / "broken.yaml").write_text("method: toy\n", encoding="utf-8")  # not listed
    return folder


@pytest.fixture
def runs(tmp_path) -> Path:
    return tmp_path / "runs"


def within(seconds: float, body) -> None:
    """Run ``body`` in a thread and fail (instead of hanging) if it takes longer than ``seconds``:
    a socket test that waits for a message that never comes would otherwise wait forever."""
    errors: list[BaseException] = []

    def target() -> None:
        try:
            body()
        except BaseException as exc:  # handed to the test below
            errors.append(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), f"did not finish within {seconds} s"
    if errors:
        raise errors[0]


def wait_until(condition, seconds: float = 120.0, what: str = "condition") -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.1)


def finished(store: RunStore) -> bool:
    return store.read_meta()["status"] != "running"


def same_folder(a: RunStore, b: RunStore) -> None:
    """Same config, metrics (apart from wall-clock seconds), checkpoints and exam cards."""

    def metrics(store):
        return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]

    assert a.read_config() == b.read_config()
    assert metrics(a) == metrics(b) and len(metrics(a)) > 0
    assert a.checkpoint_units() == b.checkpoint_units()
    for unit in a.checkpoint_units():
        assert a.load_checkpoint(unit).equals(b.load_checkpoint(unit))
    assert a.exam_units() == b.exam_units()
    for unit in a.exam_units():
        card_a, card_b = a.load_exam(unit).to_dict(), b.load_exam(unit).to_dict()
        assert card_a.pop("policy")["unit"] == card_b.pop("policy")["unit"] == unit
        assert card_a == card_b


def launch(client: TestClient, **body) -> RunStore:
    answer = client.post("/api/runs", json=body)
    assert answer.status_code == 200, answer.text
    return RunStore.open(client.app_runs_dir / answer.json()["run_id"])


@pytest.fixture
def client(runs, configs):
    with TestClient(create_app(runs_dir=runs, configs_dir=configs)) as test_client:
        test_client.app_runs_dir = runs
        yield test_client


# ── what may be launched ───────────────────────────────────────────────────────────────────────


def test_configs_lists_the_training_configs_only(client):
    answer = client.get("/api/configs").json()["configs"]
    assert [c["name"] for c in answer] == ["small_search.yaml", "toy_xor.yaml"]  # not broken
    search, xor = answer
    assert search["method"] == "random_search" and search["length_field"] == "units"
    assert search["length"] == 2 and search["task"] is None
    assert (xor["method"], xor["task"], xor["length_field"], xor["length"]) == (
        "toy",
        "xor",
        "epochs",
        3000,
    )


def test_configs_lists_the_imitation_config(configs, runs):
    # api v1.17 (task 6.3): the listing reads an imitation config instead of failing on it.
    shutil.copy(CONFIGS / "imitation.yaml", configs / "imitation.yaml")
    with TestClient(create_app(runs_dir=runs, configs_dir=configs)) as client:
        answer = client.get("/api/configs").json()["configs"]
    (imitation,) = [c for c in answer if c["name"] == "imitation.yaml"]
    assert (imitation["method"], imitation["task"]) == ("imitation", None)
    assert (imitation["length_field"], imitation["length"]) == ("epochs", 100)


def test_bad_launches_are_refused_and_start_nothing(client, runs):
    cases = [
        ({"config": "missing.yaml"}, "unknown config"),
        ({"config": "broken.yaml"}, "unknown config"),
        ({"config": "../configs/toy_xor.yaml"}, "unknown config"),
        ({"config": "small_search.yaml", "epochs": 5}, "--epochs is for toy and imitation configs"),
        ({"config": "toy_xor.yaml", "units": 5}, "--units is for random_search configs"),
        ({"config": "toy_xor.yaml", "seed": -1}, "seed"),
        ({"config": "toy_xor.yaml", "tag": "a b"}, "tag"),
    ]
    for body, message in cases:
        answer = client.post("/api/runs", json=body)
        assert answer.status_code == 422 and message in answer.text, (body, answer.text)
    wrong_field = client.post("/api/runs", json={"config": "toy_xor.yaml", "lr": 1})
    assert wrong_field.status_code == 422
    assert not runs.exists() or not [p for p in runs.iterdir() if p.name != ".launches"]


# ── a launched run is the same run as one trained from the terminal ────────────────────────────


def test_a_launched_toy_run_equals_a_run_trained_directly(client, configs, tmp_path):
    def body() -> None:
        store = launch(client, config="toy_xor.yaml", seed=3, epochs=200, tag="launch")
        assert store.path.name.endswith("-toy_xor-launch")
        wait_until(lambda: finished(store), what="the launched run")
        assert store.read_meta()["status"] == "completed"
        direct = train_toy_run(
            load_config(configs / "toy_xor.yaml", seed=3, epochs=200, tag="launch"),
            tmp_path / "direct",
        )
        same_folder(store, direct.store)
        log = next((client.app_runs_dir / ".launches").glob("*.log")).read_text()
        assert log.startswith("run folder: runs/")  # no absolute path (and no user name) in it
        assert "status: completed after 200 epochs" in log

    within(180, body)


def test_a_launched_random_search_run_equals_one_trained_directly(client, configs, tmp_path):
    def body() -> None:
        store = launch(client, config="small_search.yaml", seed=4)
        wait_until(lambda: finished(store), what="the launched run")
        assert store.exam_units() == [0, 1, 2]  # every checkpoint examined
        direct = train_random_search_run(
            load_config(configs / "small_search.yaml", seed=4), tmp_path / "direct"
        )
        same_folder(store, direct.store)
        assert check(client.app_runs_dir) == []  # the launched process kept its mirror up to date

    within(300, body)


# ── /ws/runs ───────────────────────────────────────────────────────────────────────────────────


def receive(ws) -> dict:
    return json.loads(ws.receive_text())


def test_ws_runs_streams_every_line_once_in_order(client):
    def body() -> None:
        with client.websocket_connect("/ws/runs") as ws:
            assert receive(ws)["type"] == "hello"
            store = launch(client, config="toy_xor.yaml", epochs=3000, tag="stream")
            run_id = store.path.name
            ws.send_text(json.dumps({"type": "watch", "run_id": run_id}))
            records: list[dict] = []
            replaces = increments = 0
            units: list[int] = []
            final = None
            while final is None:
                message = receive(ws)
                if message.get("run_id") != run_id:
                    continue
                if message["type"] == "metrics":
                    if message["replace"]:
                        replaces += 1
                        records = list(message["records"])
                    else:
                        increments += 1
                        records += message["records"]
                elif message["type"] == "checkpoint":
                    units = message["units"]
                elif message["type"] == "run_status" and message["exit_code"] is not None:
                    final = message
        assert final["status"] == "completed" and final["exit_code"] == 0 and final["launched"]
        assert final["units_done"] == 3000 and final["budget_units"] == 3000
        assert replaces == 1  # the history, once
        assert increments >= 2  # the rest arrived while training, in several pieces
        assert records == store.read_metrics()  # every line once, in order, nothing lost
        assert units == store.checkpoint_units()

    within(180, body)


def test_ws_runs_sends_exam_cards_and_refuses_bad_commands(client, runs):
    def body() -> None:
        store = launch(client, config="small_search.yaml", seed=1)
        wait_until(lambda: finished(store), what="the launched run")
        with client.websocket_connect("/ws/runs") as ws:
            assert receive(ws)["type"] == "hello"
            first = receive(ws)  # a run launched by this server is listed on connect
            assert first["type"] == "run_status" and first["run_id"] == store.path.name
            ws.send_text(json.dumps({"type": "watch", "run_id": store.path.name}))
            history = [receive(ws) for _ in range(5)]
            assert [m["type"] for m in history] == [
                "metrics",
                "checkpoint",
                "exam_done",
                "exam_done",
                "exam_done",
            ]
            assert [m["unit"] for m in history[2:]] == [0, 1, 2]
            card = store.load_exam(2).to_dict()
            assert history[4]["training"]["completion_rate"] == card["training"]["completion_rate"]
            assert (
                history[4]["heldout"]["progress_frac_median"]
                == card["heldout"]["progress_frac_median"]
            )
            for text, problem in (
                ("not json", "Expecting value"),
                (json.dumps({"type": "play"}), "unknown command"),
                (json.dumps({"type": "watch", "run_id": "../x"}), "unknown run"),
                (json.dumps({"type": "watch", "run_id": 3}), "string"),
            ):
                ws.send_text(text)
                answer = receive(ws)
                while answer["type"] != "error":  # status news may come in between
                    answer = receive(ws)
                assert problem in answer["message"], answer

    within(300, body)


# ── stopping ───────────────────────────────────────────────────────────────────────────────────


def test_stop_ends_a_run_cleanly_after_its_current_unit(runs, configs):
    launcher = Launcher(runs, configs, max_active=1)
    with TestClient(create_app(runs_dir=runs, launcher=launcher)) as test_client:
        test_client.app_runs_dir = runs

        def body() -> None:
            store = launch(test_client, config="small_search.yaml", units=500, tag="long")
            busy = test_client.post("/api/runs", json={"config": "toy_xor.yaml", "epochs": 5})
            assert busy.status_code == 409 and "already training" in busy.text
            wait_until(lambda: len(store.read_metrics()) >= 1, what="the first unit")
            stopped = test_client.post(f"/api/runs/{store.path.name}/stop")
            assert stopped.status_code == 200 and stopped.json()["stop_requested"]
            wait_until(lambda: finished(store), what="the stopped run")
            meta = store.read_meta()
            assert (meta["status"], meta["stop_reason"]) == ("completed", "stopped")
            done = store.read_metrics()[-1]["unit"]
            assert done < 500
            assert store.checkpoint_units()[-1] == done and store.exam_units()[-1] == done
            assert not (store.path / STOP_FILE).exists()  # the request was answered and removed
            assert check(runs) == []  # the stopped run is a complete run in the mirror too
            again = test_client.post(f"/api/runs/{store.path.name}/stop")
            assert again.status_code == 409 and "already completed" in again.text
            wait_until(lambda: launcher.exit_codes()[store.path.name] is not None, 30, "exit")
            free = test_client.post("/api/runs", json={"config": "toy_xor.yaml", "epochs": 5})
            assert free.status_code == 200  # the slot is free again

        within(300, body)
        for bad in ("nope", "20261102-000000-walk-missing"):
            assert test_client.post(f"/api/runs/{bad}/stop").status_code == 404


def test_a_process_that_does_not_start_in_time_is_reported(runs, configs):
    launcher = Launcher(runs, configs, launch_timeout_s=0.0)
    with TestClient(create_app(runs_dir=runs, launcher=launcher)) as test_client:
        answer = test_client.post("/api/runs", json={"config": "toy_xor.yaml", "epochs": 5})
    assert answer.status_code == 500 and "did not start within 0 s" in answer.text
