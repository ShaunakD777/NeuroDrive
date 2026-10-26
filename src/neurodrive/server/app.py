"""The FastAPI application: health, track data, run folders and live driving over a WebSocket.

REST is request and response (the browser asks once, the server answers once). A WebSocket stays
open, so the server can push messages without being asked each time, which live driving needs.
"""

import asyncio
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from neurodrive import __version__
from neurodrive.core.activations import ACTIVATIONS
from neurodrive.server.live import LiveSession, TrackLibrary
from neurodrive.server.runs import RunLibrary
from neurodrive.server.schemas import (
    ActivationCurves,
    CheckpointList,
    CheckpointNetwork,
    HealthResponse,
    HelloMessage,
    RunDetail,
    RunListResponse,
    RunMetrics,
    ToyCheckpointResponse,
    TrackGeometry,
    TrackListResponse,
    WireModel,
)

DEFAULT_TRACKS_DIR = Path(__file__).resolve().parents[3] / "tracks"  # <repository>/tracks
DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"  # <repository>/runs (git-ignored)
TICK_SECONDS = 1.0 / 120.0  # timer period; Windows rounds it up to about 15 ms, which is still fine


def create_app(tracks_dir: Path | None = None, runs_dir: Path | None = None) -> FastAPI:
    """Build the app. A function, so each test can get a fresh one (and its own folders)."""
    app = FastAPI(title="NeuroDrive", version=__version__)
    tracks = TrackLibrary(DEFAULT_TRACKS_DIR if tracks_dir is None else tracks_dir)
    runs = RunLibrary(DEFAULT_RUNS_DIR if runs_dir is None else runs_dir)

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

    # Run folders (api v1.10, ADR-031). Read-only; an unknown run or checkpoint is a 404.
    def missing(what: str) -> HTTPException:
        return HTTPException(status_code=404, detail=f"unknown {what}")

    @app.get("/api/runs")
    def list_runs() -> RunListResponse:
        return runs.list()

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

    @app.get("/api/runs/{run_id}/checkpoints/{unit}/toy")
    def toy_checkpoint(run_id: str, unit: int) -> ToyCheckpointResponse:
        try:
            return runs.toy(run_id, unit)
        except KeyError:
            raise missing(f"checkpoint {unit} of run {run_id!r}") from None
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None

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

        session = LiveSession(tracks)
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
