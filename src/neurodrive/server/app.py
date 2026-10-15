"""The FastAPI application: health, track data, and live driving over a WebSocket.

REST is request and response (the browser asks once, the server answers once). A WebSocket stays
open, so the server can push messages without being asked each time, which live driving needs.
"""

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from neurodrive import __version__
from neurodrive.server.live import LiveSession, TrackLibrary
from neurodrive.server.schemas import (
    HealthResponse,
    HelloMessage,
    TrackGeometry,
    TrackListResponse,
    WireModel,
)

DEFAULT_TRACKS_DIR = Path(__file__).resolve().parents[3] / "tracks"  # <repository>/tracks
TICK_SECONDS = 1.0 / 120.0  # timer period; Windows rounds it up to about 15 ms, which is still fine


def create_app(tracks_dir: Path | None = None) -> FastAPI:
    """Build the app. A function, so each test can get a fresh one (and its own tracks folder)."""
    app = FastAPI(title="NeuroDrive", version=__version__)
    tracks = TrackLibrary(DEFAULT_TRACKS_DIR if tracks_dir is None else tracks_dir)

    @app.get("/api/health")
    def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @app.get("/api/tracks")
    def list_tracks() -> TrackListResponse:
        return TrackListResponse(tracks=tracks.info())

    @app.get("/api/tracks/{name}")
    def track_geometry(name: str) -> TrackGeometry:
        try:
            return tracks.geometry(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown track {name!r}") from None

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
