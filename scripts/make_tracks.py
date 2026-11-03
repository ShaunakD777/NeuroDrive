"""Write the track JSON files in tracks/ from the corner points defined here.

Run from the repository root:  ``uv run python scripts/make_tracks.py``

The corner points are chosen by hand, either directly or as straight and arc pieces (no
randomness), so the output is identical on every run. The loader (``neurodrive.env.track``) draws
a smooth curve through them. Most loops go counter-clockwise, so the left edge of the road is the
inside edge; ``monza_cw`` and ``train_loop_01_mirror`` go clockwise.

Track set (task 5.1, ADR-034): ``train_loop_01`` is the training track; the held-out tracks are
``hairpins_01``, ``sweeper_01``, ``train_loop_01_mirror`` and ``monza_ccw``. ``monza_cw`` is a
second training track for studies on a long road.
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


# The training loop: a smooth loop with gentle and medium curves (task 1.2).
TRAIN_LOOP_01 = [
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
]

# The held-out tracks of task 5.1 are drawn as pieces: ("straight", length in metres) or
# ("arc", radius in metres, turn in degrees, positive = left). A length of None is left open and
# solved by close_loop. Each list turns +360 degrees in total (counter-clockwise).
HAIRPINS_01 = [
    ("straight", 300),  # start straight
    ("arc", 13, 150),  # left hairpin
    ("straight", 160),  # diagonal
    ("arc", 13, -150),  # right hairpin
    ("straight", 160),
    ("arc", 30, 90),
    ("straight", 100),
    ("arc", 30, 90),
    ("straight", None),  # top straight, solved
    ("arc", 30, 90),
    ("straight", None),  # left-hand side, solved
    ("arc", 30, 90),
]
SWEEPER_01 = [
    ("straight", 220),  # start straight
    ("arc", 70, 210),  # the sweeper: about 250 m at a steady 70 m radius
    ("straight", None),  # solved
    ("arc", 35, -60),  # right bend
    ("straight", 80),
    ("arc", 30, 120),
    ("straight", None),  # left-hand side, solved
    ("arc", 30, 90),
]
ARC_STEP_DEG = 15.0  # an arc gets a corner point at least every 15 degrees
STRAIGHT_STEP = 40.0  # a straight gets a corner point at least every 40 m


def trace_pieces(pieces: list[tuple]) -> list[tuple[float, float]]:
    """Walk ``pieces`` with a pen and return the points it drops, first (0, 0) to last.

    The pen starts at (0, 0) heading east (heading h = 0, counter-clockwise positive). A
    straight of length L moves it L metres along h, in equal steps of at most 40 m. An arc of
    radius r turning by angle a moves it round a circle whose centre is r metres to the pen's
    left (a > 0) or right (a < 0): the centre is c = pen + sign(a) r (-sin h, cos h), and after
    turning to heading h' the pen is at c + sign(a) r (sin h', -cos h'). A point is dropped at
    least every 15 degrees. Every length must be a number here (see close_loop).
    """
    x = y = h = 0.0
    points = [(x, y)]
    for piece in pieces:
        if piece[0] == "straight":
            length = piece[1]
            steps = max(1, int(np.ceil(length / STRAIGHT_STEP)))
            for k in range(1, steps + 1):
                d = length * k / steps
                points.append((x + np.cos(h) * d, y + np.sin(h) * d))
        else:
            _, radius, degrees = piece
            turn, side = np.radians(degrees), np.sign(degrees)
            cx, cy = x - side * radius * np.sin(h), y + side * radius * np.cos(h)
            steps = max(2, int(np.ceil(abs(degrees) / ARC_STEP_DEG)))
            for k in range(1, steps + 1):
                hk = h + turn * k / steps
                points.append((cx + side * radius * np.sin(hk), cy - side * radius * np.cos(hk)))
            h += turn
        x, y = points[-1]
    return points


def close_loop(pieces: list[tuple]) -> list[tuple]:
    """Return ``pieces`` with its two open straights (length None) sized to close the loop.

    The pen's end point depends linearly on the straight lengths: end = e0 + L1 d1 + L2 d2,
    where e0 is the end point with both open straights at length 0 and d1, d2 are the unit
    directions the pen faces on them (known, because the arcs before them fix the heading).
    Setting end = (0, 0) gives two linear equations, one for x and one for y, in L1 and L2.
    """
    open_idx = [i for i, piece in enumerate(pieces) if piece[0] == "straight" and piece[1] is None]
    if len(open_idx) != 2:
        raise ValueError("a loop needs exactly two open straights to close")
    if sum(piece[2] for piece in pieces if piece[0] == "arc") != 360:
        raise ValueError("the arcs must turn +360 degrees in total")
    zeroed = [("straight", 0.0) if i in open_idx else piece for i, piece in enumerate(pieces)]
    e0 = np.array(trace_pieces(zeroed)[-1])
    directions = []
    for i in open_idx:
        h = np.radians(sum(piece[2] for piece in pieces[:i] if piece[0] == "arc"))
        directions.append([np.cos(h), np.sin(h)])
    lengths = np.linalg.solve(np.array(directions).T, -e0)
    if np.any(lengths <= 0):
        raise ValueError(f"the open straights would need lengths {lengths}; change the pieces")
    closed = list(pieces)
    for i, length in zip(open_idx, lengths, strict=True):
        closed[i] = ("straight", float(length))
    return closed


def pieces_centerline(pieces: list[tuple]) -> list[list[float]]:
    """Corner points (metres, rounded to 0.1 m) of the closed loop drawn by ``pieces``.

    The pen's last point is the start again, so it is dropped (track v1 never repeats it).
    """
    points = trace_pieces(close_loop(pieces))[:-1]
    return [[round(float(x), 1) + 0.0, round(float(y), 1) + 0.0] for x, y in points]


def mirrored(centerline: list[list[float]]) -> list[list[float]]:
    """Reflect corner points in the y-axis: (x, y) becomes (-x, y).

    Lengths and radii stay the same, every left bend becomes a right bend (in the same order),
    and a road heading h becomes pi - h. ``0 - x`` keeps 0 as 0 rather than -0.
    """
    return [[0 - x, y] for x, y in centerline]


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
        "centerline": TRAIN_LOOP_01,
    },
    # Held out (task 5.1): two hairpins tighter than any bend of the training loop.
    "hairpins_01": {
        "name": "hairpins_01",
        "version": 1,
        "split": "heldout",
        "width": 12,
        "start": {"s0": 0.0, "direction": 1},
        "centerline": pieces_centerline(HAIRPINS_01),
    },
    # Held out (task 5.1): one long bend of steady radius.
    "sweeper_01": {
        "name": "sweeper_01",
        "version": 1,
        "split": "heldout",
        "width": 12,
        "start": {"s0": 0.0, "direction": 1},
        "centerline": pieces_centerline(SWEEPER_01),
    },
    # Held out (task 5.1): the training loop reflected, so every bend turns the other way.
    "train_loop_01_mirror": {
        "name": "train_loop_01_mirror",
        "version": 1,
        "split": "heldout",
        "width": 12,
        "start": {"s0": 0.0, "direction": 1},
        "centerline": mirrored(TRAIN_LOOP_01),
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
