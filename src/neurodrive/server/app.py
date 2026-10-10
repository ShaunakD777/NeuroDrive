"""The FastAPI application: a health check and the live WebSocket.

REST is request and response (the browser asks once, the server answers once). A WebSocket stays
open, so the server can push messages without being asked each time, which live driving needs.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from neurodrive import __version__
from neurodrive.server.schemas import HealthResponse, HelloMessage


def create_app() -> FastAPI:
    """Build the app. A function, so each test can get a fresh one."""
    app = FastAPI(title="NeuroDrive", version=__version__)

    @app.get("/api/health")
    def health() -> HealthResponse:
        return HealthResponse(version=__version__)

    @app.websocket("/ws/live")
    async def live(websocket: WebSocket) -> None:
        # Phase 0 placeholder: greet the client, then echo every text message back unchanged.
        # Task 1.7 replaces the echo with live driving sessions.
        await websocket.accept()
        await websocket.send_text(HelloMessage(version=__version__).model_dump_json())
        try:
            while True:
                await websocket.send_text(await websocket.receive_text())
        except WebSocketDisconnect:
            pass

    return app


app = create_app()
