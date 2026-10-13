"""Write the track JSON files in tracks/ from the corner points defined here.

Run from the repository root:  ``uv run python scripts/make_tracks.py``

The corner points are chosen by hand (no randomness), so the output is identical on every run.
The loader (``neurodrive.env.track``) draws a smooth curve through them. Corner points go
counter-clockwise, so the left edge of the road is the inside edge of the loop.
"""

import json
from pathlib import Path

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"

# name -> track v1 fields. Metres; x right, y up.
TRACKS: dict[str, dict] = {
    "train_loop_01": {
        "name": "train_loop_01",
        "version": 1,
        "split": "train",
        "width": 12,
        "start": {"s0": 0.0, "direction": 1},
        "centerline": [
            [0, 0],
            [60, -5],
            [120, 0],
            [170, 30],
            [190, 80],
            [165, 125],
            [110, 145],
            [65, 125],
            [20, 130],
            [-25, 150],
            [-80, 120],
            [-90, 60],
            [-50, 20],
        ],
    },
}


def render(track: dict) -> str:
    """Return the JSON text of a track: one corner point per line, so diffs stay readable."""
    points = ",\n".join(f"    [{x}, {y}]" for x, y in track["centerline"])
    start = json.dumps(track["start"])
    return (
        "{\n"
        f'  "name": {json.dumps(track["name"])},\n'
        f'  "version": {track["version"]},\n'
        f'  "split": {json.dumps(track["split"])},\n'
        f'  "width": {track["width"]},\n'
        f'  "start": {start},\n'
        '  "centerline": [\n'
        f"{points}\n"
        "  ]\n"
        "}\n"
    )


def build_tracks() -> dict[str, str]:
    """Return {file name: JSON text} for every track defined above."""
    return {f"{name}.json": render(track) for name, track in TRACKS.items()}


def main() -> None:
    TRACKS_DIR.mkdir(exist_ok=True)
    for file_name, text in build_tracks().items():
        (TRACKS_DIR / file_name).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote tracks/{file_name}")


if __name__ == "__main__":
    main()
