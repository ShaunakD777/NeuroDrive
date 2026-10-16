"""Print the JSON Schema of every server message model as one JSON object on standard output.

``npm run gen:types`` (in ``web/``) runs this and turns each schema into a TypeScript type, so the
browser's types always come from the Python models in ``neurodrive/server/schemas.py``.
"""

import json

from neurodrive.server.schemas import WIRE_MODELS, WireModel


def schema_for(model: type[WireModel]) -> dict:
    """The JSON Schema of one model, as the server would send it."""
    schema = model.model_json_schema(mode="serialization")
    # Pydantic titles every field ("Version", "Type", ...). The TypeScript converter would turn
    # each title into a named alias, and two models with a "version" field would then clash. The
    # models nested inside this one (under "$defs") have fields too, so clean those as well.
    for part in [schema, *schema.get("$defs", {}).values()]:
        for field_schema in part.get("properties", {}).values():
            field_schema.pop("title", None)
    return schema


def main() -> None:
    schemas = {m.__name__: schema_for(m) for m in WIRE_MODELS}
    print(json.dumps(schemas, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
