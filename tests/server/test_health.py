"""Tests for the server skeleton: health check and the WebSocket hello."""

from fastapi.testclient import TestClient

from neurodrive import __version__
from neurodrive.server.app import create_app
from neurodrive.server.schemas import API_VERSION, HelloMessage


def test_health_is_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_live_socket_sends_hello_first() -> None:
    client = TestClient(create_app())
    with client.websocket_connect("/ws/live") as ws:
        hello = HelloMessage.model_validate_json(ws.receive_text())
    assert hello.type == "hello"
    assert hello.version == __version__
    assert hello.api == API_VERSION
