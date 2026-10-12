"""Message shapes sent between the server and the browser (api v1).

Every shape is a Pydantic model, and the TypeScript types for the web app are generated from
these (task 0.4), so the two sides cannot drift apart.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

API_VERSION = "v1"


class WireModel(BaseModel):
    """Base class for everything the server sends.

    The server always fills in every field, so the generated schema marks fields that have a
    default value as required too. Without this, TypeScript would treat them as maybe-missing.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class HealthResponse(WireModel):
    """Answer to ``GET /api/health``: the server is up, and which NeuroDrive it is running."""

    status: Literal["ok"] = "ok"
    version: str


class HelloMessage(WireModel):
    """First message on ``/ws/live``, sent as soon as a client connects.

    ``version`` is the NeuroDrive package version and ``api`` the message-contract version.
    """

    type: Literal["hello"] = "hello"
    version: str
    api: str = API_VERSION


# Every model the browser sends or receives. `npm run gen:types` turns each one into a TypeScript
# type, so a new message shape is added here and nowhere else.
WIRE_MODELS: tuple[type[WireModel], ...] = (HealthResponse, HelloMessage)
