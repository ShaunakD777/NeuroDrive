"""The FastAPI application: health, track data, run folders, live driving over a WebSocket,
launching and following training runs (task 5.8), and demonstration datasets (task 6.2).

REST is request and response (the browser asks once, the server answers once). A WebSocket stays
open, so the server can push messages without being asked each time, which live driving and
training progress need. Training itself runs in separate processes (``server/runs_watch.py``).
"""

import asyncio
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect

from neurodrive import __version__
from neurodrive.core.activations import ACTIVATIONS
from neurodrive.server.datasets import DatasetLibrary
from neurodrive.server.live import LiveSession, TrackLibrary
from neurodrive.server.runs import RunLibrary
from neurodrive.server.runs_watch import Launcher, LaunchError, RunsHub, RunsWatcher
from neurodrive.server.schemas import (
    ActivationCurves,
    CheckpointDiff,
    CheckpointList,
    CheckpointNetwork,
    CompareResponse,
    ConfigListResponse,
    DatasetDetail,
    DatasetListResponse,
    ExamCardResponse,
    HealthResponse,
    HelloMessage,
    LaunchRequest,
    LaunchResponse,
    RunDetail,
    RunListResponse,
    RunMetrics,
    StopResponse,
    ToyCheckpointResponse,
    TrackGeometry,
    TrackListResponse,
    WireModel,
)

DEFAULT_TRACKS_DIR = Path(__file__).resolve().parents[3] / "tracks"  # <repository>/tracks
DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"  # <repository>/runs (git-ignored)
DEFAULT_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"  # what the app may launch
DEFAULT_DATASETS_DIR = Path(__file__).resolve().parents[3] / "data" / "demos"  # git-ignored
MAX_COMPARE_RUNS = 64  # runs per comparison request (8 groups of 8 seeds)
TICK_SECONDS = 1.0 / 120.0  # timer period; Windows rounds it up to about 15 ms, which is still fine


def create_app(
    tracks_dir: Path | None = None,
    runs_dir: Path | None = None,
    configs_dir: Path | None = None,
    launcher: Launcher | None = None,
    datasets_dir: Path | None = None,
) -> FastAPI:
    """Build the app. A function, so each test can get a fresh one (and its own folders).

    ``launcher`` replaces the default one (``configs_dir``, at most 4 runs at once), for tests.
    """
    app = FastAPI(title="NeuroDrive", version=__version__)
    tracks = TrackLibrary(DEFAULT_TRACKS_DIR if tracks_dir is None else tracks_dir)
    runs = RunLibrary(DEFAULT_RUNS_DIR if runs_dir is None else runs_dir)
    datasets = DatasetLibrary(DEFAULT_DATASETS_DIR if datasets_dir is None else datasets_dir)
    if launcher is None:
        configs = DEFAULT_CONFIGS_DIR if configs_dir is None else configs_dir
        launcher = Launcher(runs.runs_dir, configs)
    hub = RunsHub(RunsWatcher(runs.runs_dir, launcher))

    @app.get("/api/health")
    def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @app.get("/api/activations")
    def activations() -> ActivationCurves:
        # The activation functions sampled on z from -5 to 5, for the browser to draw (the browser
        # never computes them itself).
        z = np.linspace(-5.0, 5.0, 101)
        return ActivationCurves(
            z=np.round(z, 4).tolist(),
            curves={name: np.round(f(z), 5).tolist() for name, f in sorted(ACTIVATIONS.items())},
        )

    @app.get("/api/tracks")
    def list_tracks() -> TrackListResponse:
        return TrackListResponse(tracks=tracks.info())

    @app.get("/api/tracks/{name}")
    def track_geometry(name: str) -> TrackGeometry:
        try:
            return tracks.geometry(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown track {name!r}") from None

    # Run folders (api v1.11, ADR-031, ADR-033). Read-only; an unknown run or checkpoint is a 404.
    def missing(what: str) -> HTTPException:
        return HTTPException(status_code=404, detail=f"unknown {what}")

    @app.get("/api/runs")
    def list_runs() -> RunListResponse:
        return runs.list()

    # Declared before /api/runs/{run_id}, so that "compare" is not taken for a run id.
    @app.get("/api/runs/compare")
    def compare_runs(
        ids: str = Query(..., description="Comma-separated run ids."),
        metric: Literal["loss", "val_loss"] = "loss",
    ) -> CompareResponse:
        run_ids = [i for i in ids.split(",") if i]
        if not run_ids:
            raise HTTPException(status_code=422, detail="ids must name at least one run")
        if len(run_ids) > MAX_COMPARE_RUNS:
            raise HTTPException(
                status_code=422, detail=f"compare at most {MAX_COMPARE_RUNS} runs at once"
            )
        try:
            return runs.compare(run_ids, metric)
        except KeyError as error:
            raise missing(f"run {error.args[0]!r}") from None

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> RunDetail:
        try:
            return runs.detail(run_id)
        except KeyError:
            raise missing(f"run {run_id!r}") from None

    @app.get("/api/runs/{run_id}/metrics")
    def run_metrics(run_id: str) -> RunMetrics:
        try:
            return runs.metrics(run_id)
        except KeyError:
            raise missing(f"run {run_id!r}") from None

    @app.get("/api/runs/{run_id}/checkpoints")
    def run_checkpoints(run_id: str) -> CheckpointList:
        try:
            return runs.checkpoints(run_id)
        except KeyError:
            raise missing(f"run {run_id!r}") from None

    @app.get("/api/runs/{run_id}/checkpoints/{unit}")
    def checkpoint_network(run_id: str, unit: int) -> CheckpointNetwork:
        try:
            return runs.network(run_id, unit)
        except KeyError:
            raise missing(f"checkpoint {unit} of run {run_id!r}") from None

    # The run timeline (api v1.14, task 5.9, ADR-042): what changed between two checkpoints, and
    # a checkpoint's exam card (the replay's per-car results).
    @app.get("/api/runs/{run_id}/checkpoints/{unit}/diff")
    def checkpoint_diff(
        run_id: str,
        unit: int,
        from_unit: int = Query(..., alias="from", description="The checkpoint to compare with."),
    ) -> CheckpointDiff:
        try:
            return runs.diff(run_id, unit, from_unit)
        except KeyError:
            raise missing(f"checkpoint {unit} or {from_unit} of run {run_id!r}") from None
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None

    @app.get("/api/runs/{run_id}/exams/{unit}")
    def exam_card(run_id: str, unit: int) -> ExamCardResponse:
        try:
            return runs.exam_card(run_id, unit)
        except KeyError:
            raise missing(f"exam card {unit} of run {run_id!r}") from None

    @app.get("/api/runs/{run_id}/checkpoints/{unit}/toy")
    def toy_checkpoint(run_id: str, unit: int) -> ToyCheckpointResponse:
        try:
            return runs.toy(run_id, unit)
        except KeyError:
            raise missing(f"checkpoint {unit} of run {run_id!r}") from None
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None

    # Demonstration datasets (api v1.15, task 6.2, ADR-045). Read-only, like the run folders.
    @app.get("/api/datasets")
    def list_datasets() -> DatasetListResponse:
        return datasets.list()

    @app.get("/api/datasets/{dataset_id}")
    def dataset_detail(dataset_id: str) -> DatasetDetail:
        try:
            return datasets.detail(dataset_id)
        except KeyError:
            raise missing(f"dataset {dataset_id!r}") from None
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None

    # Launching training runs (api v1.13, task 5.8, ADR-041). The server starts `nd train` as a
    # separate process and never trains itself; /ws/runs reports what the process writes.
    def refused(error: LaunchError) -> HTTPException:
        return HTTPException(status_code=error.status, detail=error.detail)

    @app.get("/api/configs")
    def list_configs() -> ConfigListResponse:
        return ConfigListResponse(configs=launcher.configs())

    @app.post("/api/runs")
    def launch_run(request: LaunchRequest) -> LaunchResponse:
        # A plain (not async) function: FastAPI runs it in a worker thread, so the few seconds
        # spent waiting for the new process to create its folder never block live driving.
        try:
            return LaunchResponse(run_id=launcher.launch(request))
        except LaunchError as error:
            raise refused(error) from None

    @app.post("/api/runs/{run_id}/stop")
    def stop_run(run_id: str) -> StopResponse:
        try:
            launcher.stop(runs.store(run_id))
        except KeyError:
            raise missing(f"run {run_id!r}") from None
        except LaunchError as error:
            raise refused(error) from None
        return StopResponse(run_id=run_id)

    @app.websocket("/ws/runs")
    async def runs_socket(websocket: WebSocket) -> None:
        # Messages are queued per connection (by `watch` and by the poll loop) and one writer
        # sends them in order, so a run's history is never overtaken by later news about it.
        await websocket.accept()
        client = hub.connect()

        async def writer() -> None:
            while True:
                message = await client.queue.get()
                await websocket.send_text(message.model_dump_json())

        sending = asyncio.create_task(writer())
        try:
            while True:
                hub.handle_text(client, await websocket.receive_text())
        except WebSocketDisconnect:
            pass
        finally:
            sending.cancel()
            hub.disconnect(client)

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket) -> None:
        # One independent session per connection. Two things happen at once: commands arrive from
        # the browser, and a timer ticks the simulation. Both only call the (synchronous) session,
        # so they never interleave in the middle of a step.
        await websocket.accept()
        send_lock = asyncio.Lock()

        async def send(message: WireModel) -> None:
            async with send_lock:
                await websocket.send_text(message.model_dump_json())

        async def ticker() -> None:
            while True:
                await asyncio.sleep(TICK_SECONDS)
                for message in session.tick(time.monotonic()):
                    await send(message)

        session = LiveSession(tracks, runs)
        await send(HelloMessage(version=__version__))
        timer = asyncio.create_task(ticker())
        try:
            while True:
                for message in session.handle_text(await websocket.receive_text()):
                    await send(message)
        except WebSocketDisconnect:
            pass
        finally:
            timer.cancel()

    return app


app = create_app()
