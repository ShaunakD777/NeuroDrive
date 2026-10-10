"""Message shapes sent between the server and the browser (api v1).

Every shape is a Pydantic model, and the TypeScript types for the web app are generated from
these (task 0.4), so the two sides cannot drift apart.
"""

from typing import Literal

from pydantic import BaseModel

API_VERSION = "v1"


class HealthResponse(BaseModel):
    """Answer to ``GET /api/health``: the server is up, and which NeuroDrive it is running."""

    status: Literal["ok"] = "ok"
    version: str


class HelloMessage(BaseModel):
    """First message on ``/ws/live``, sent as soon as a client connects.

    ``version`` is the NeuroDrive package version and ``api`` the message-contract version.
    """

    type: Literal["hello"] = "hello"
    version: str
    api: str = API_VERSION
