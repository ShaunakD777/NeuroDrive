"""Write the track JSON files in tracks/ from the corner points defined here.

Run from the repository root:  ``uv run python scripts/make_tracks.py``

The corner points are chosen by hand (no randomness), so the output is identical on every run.
The loader (``neurodrive.env.track``) draws a smooth curve through them. Corner points go
counter-clockwise, so the left edge of the road is the inside edge of the loop.
"""

import json
from pathlib import Path

import numpy as np

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"

# Monza, traced by hand from a circuit map (pixels, x right and y DOWN), in its racing direction
# (clockwise), starting at the start/finish line on the main straight. Corner by corner: main
# straight, T1-T2 chicane (Rettifilo), Curva Grande, T4-T5 chicane (Roggia), Lesmo 1 and 2, the
# long diagonal, the T8-T10 chicane (Ascari), the back straight and the Parabolica (T11). The
# chicanes are drawn slightly wider than the real ones, so that no bend is tighter than about
# 11 m: a 12 m road cannot follow a bend tighter than 6 m without its inside edge folding over.
MONZA_PIXELS = [
    (670, 374.5),
    (600, 374.8),
    (530, 375),
    (460, 375),
    (400, 374.7),
    (364, 374.5),
    (356, 373.8),
    (350, 370.5),
    (345, 365),
    (339, 362.5),
    (330, 364),  # T1-T2
    (320, 366),
    (300, 372),
    (285, 375),
    (255, 375.5),
    (235, 373),
    (215, 368),
    (198, 360),
    (185, 350),
    (175, 339),
    (166, 323),
    (160, 305),  # Curva Grande
    (156, 282),
    (152, 255),
    (149, 225),
    (147, 205),
    (145, 190),
    (142, 176),
    (137, 167),
    (129, 163),
    (123, 157),
    (119, 146),  # T4-T5 (Roggia)
    (114, 130),
    (106, 107),
    (98, 82),
    (93, 66),
    (95, 53),
    (102, 46),
    (114, 43),
    (140, 39),
    (165, 34),  # Lesmo 1
    (180, 31.5),
    (189, 34),
    (196, 42),
    (205, 56),
    (222, 78),  # Lesmo 2
    (260, 121),
    (293, 157),
    (330, 198),
    (360, 232),
    (380, 253),  # the diagonal
    (391, 266),
    (399, 273),
    (407, 275),
    (418, 272.5),
    (429, 272),
    (438, 276),
    (446, 282),
    (454, 288),
    (464, 291),  # T8-T10 (Ascari)
    (490, 292),
    (560, 293.5),
    (640, 295.5),
    (700, 297),
    (735, 297.5),  # back straight
    (752, 299),
    (764, 304),
    (771, 313),
    (774, 324),
    (772, 336),
    (766, 347),
    (756, 357),
    (742, 366),
    (725, 371.5),
    (705, 374),
    (685, 374.5),  # Parabolica
]
MONZA_LENGTH = 5793.0  # metres, the real circuit
MONZA_SCALE = 0.5  # our track is half the real length


def monza_centerline(direction: int) -> list[list[float]]:
    """Corner points of the half-size Monza in metres (x right, y up), start line at (0, 0).

    The traced pixels are scaled so that the corner-to-corner outline is half the real length,
    and y is flipped (pixels count downwards). ``direction`` +1 keeps the racing direction
    (clockwise); -1 drives the same road the other way round (counter-clockwise), still starting
    on the start line.
    """
    px = np.array(MONZA_PIXELS, dtype=float)
    perimeter = np.linalg.norm(np.diff(np.vstack([px, px[:1]]), axis=0), axis=1).sum()
    scale = MONZA_LENGTH * MONZA_SCALE / perimeter  # metres per pixel
    xy = np.column_stack([(px[:, 0] - px[0, 0]) * scale, -(px[:, 1] - px[0, 1]) * scale])
    if direction == -1:
        xy = np.vstack([xy[:1], xy[:0:-1]])  # same start point, the rest in reverse order
    # "+ 0.0" turns a rounded -0.0 into 0.0, so the file never shows "-0.0".
    return [[round(float(x), 1) + 0.0, round(float(y), 1) + 0.0] for x, y in xy]


# name -> track v1 fields. Metres; x right, y up.
TRACKS: dict[str, dict] = {
    # Monza at half size, in its racing direction (clockwise): a training track.
    "monza_cw": {
        "name": "monza_cw",
        "version": 1,
        "split": "train",
        "width": 12,
        "start": {"s0": 0.0, "direction": 1},
        "centerline": monza_centerline(+1),
    },
    # The same road driven the other way (counter-clockwise): held out, never trained or tuned on.
    "monza_ccw": {
        "name": "monza_ccw",
        "version": 1,
        "split": "heldout",
        "width": 12,
        "start": {"s0": 0.0, "direction": 1},
        "centerline": monza_centerline(-1),
    },
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
