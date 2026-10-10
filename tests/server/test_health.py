"""Tests for the server skeleton: health check, WebSocket hello, WebSocket echo."""

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


def test_live_socket_echoes_text_unchanged() -> None:
    client = TestClient(create_app())
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_text()  # skip hello
        for text in ["ping", '{"type": "play"}', ""]:
            ws.send_text(text)
            assert ws.receive_text() == text
