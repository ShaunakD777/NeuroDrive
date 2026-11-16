// Drawing geometry for the Environment view: world metres to screen pixels, a camera, and a few
// helpers. Pure functions with no browser objects, so they can be unit-tested. All numbers about
// the simulation come from the server; this only decides where to put them on the screen.
//
// World coordinates: metres, x to the right, y up, heading 0 along +x and counter-clockwise
// positive. Screen coordinates: pixels, x to the right, y DOWN, so y is flipped.

export interface Size {
  w: number;
  h: number;
}

/** The camera: the world point at the centre of the canvas and how many pixels one metre is. */
export interface Camera {
  cx: number;
  cy: number;
  scale: number;
}

export const MIN_SCALE = 0.5;
export const MAX_SCALE = 40;

export function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale));
}

/** World (x, y) in metres to screen (px, py) in pixels. */
export function worldToScreen(cam: Camera, size: Size, x: number, y: number): [number, number] {
  return [size.w / 2 + (x - cam.cx) * cam.scale, size.h / 2 - (y - cam.cy) * cam.scale];
}

/** Screen (px, py) back to world (x, y). */
export function screenToWorld(cam: Camera, size: Size, px: number, py: number): [number, number] {
  return [cam.cx + (px - size.w / 2) / cam.scale, cam.cy - (py - size.h / 2) / cam.scale];
}

/** A camera centred on one world point at a chosen scale (used to follow the car). */
export function followCamera(x: number, y: number, scale: number): Camera {
  return { cx: x, cy: y, scale: clampScale(scale) };
}

/** Pixels to keep clear on each side of the canvas (floating panels, a frame). */
export interface Margins {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/**
 * A camera that shows every point of `points` (each [x, y]) inside the canvas minus `margin`
 * pixels: one number for every side, or one per side (task E3). With a box x ∈ [x0, x1],
 * y ∈ [y0, y1] and margins t, r, b, l the scale is s = min((w − l − r)/(x1 − x0),
 * (h − t − b)/(y1 − y0)); the free area's centre lies (l − r)/2 px right of and (t − b)/2 px
 * below the canvas centre, so the camera centre moves the other way:
 * cx = (x0 + x1)/2 − (l − r)/(2s), cy = (y0 + y1)/2 + (t − b)/(2s). Example: a 1000 × 600 canvas,
 * margins 70, 28, 164, 28 and a box from (0, 0) to (400, 200) give s = 1.83 and cy = 74.3, so the
 * box's top lands 70 px from the top and its bottom 164 px from the bottom.
 */
export function fitCamera(points: number[][], size: Size, margin: number | Margins = 24): Camera {
  if (points.length === 0 || size.w <= 0 || size.h <= 0) return { cx: 0, cy: 0, scale: 1 };
  const m = asMargins(margin);
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const scale = clampScale(
    Math.min((size.w - m.left - m.right) / spanX, (size.h - m.top - m.bottom) / spanY),
  );
  const [dx, dy] = marginShift(m, scale);
  return { cx: (minX + maxX) / 2 + dx, cy: (minY + maxY) / 2 + dy, scale };
}

function asMargins(margin: number | Margins): Margins {
  return typeof margin === "number" ? { top: margin, right: margin, bottom: margin, left: margin } : margin;
}

/**
 * How far (metres) to move a camera centre so that a world point meant to be in the middle shows
 * in the middle of the area the margins leave free: (−(l − r)/(2s), (t − b)/(2s)) at s px per
 * metre. Used by `fitCamera` and by the following camera, so a followed car is not hidden behind
 * the panels at the bottom. Example: t = 70, b = 164, l = r, s = 6 → (0, −7.83): the camera looks
 * 7.83 m below the car, so the car shows 47 px above the canvas centre.
 */
export function marginShift(margin: number | Margins, scale: number): [number, number] {
  const m = asMargins(margin);
  return [-(m.left - m.right) / (2 * scale), (m.top - m.bottom) / (2 * scale)];
}

/**
 * Margins shrunk so the panels never take more than half of the canvas (E3 verification): when
 * t + b > h/2, both are multiplied by (h/2)/(t + b); likewise l and r against w/2. Each pair keeps
 * its proportion, so the free area's middle moves the same way, only less. Half is enough for a
 * followed car: the shift is at most h/4 and the look-ahead at most 0.22·h, so the car stays at
 * least 0.03·h inside the canvas. Example: a 390 × 260 stage with t = 70 and b = 230 (a tall
 * telemetry): t + b = 300 > 130, factor 130/300 = 0.433, so t = 30.3 and b = 99.7; l = r = 28
 * (56 ≤ 195) stay.
 */
export function clampMargins(margin: number | Margins, size: Size): Margins {
  const m = asMargins(margin);
  const v = Math.min(1, size.h / 2 / Math.max(m.top + m.bottom, 1e-9));
  const u = Math.min(1, size.w / 2 / Math.max(m.left + m.right, 1e-9));
  return { top: m.top * v, right: m.right * u, bottom: m.bottom * v, left: m.left * u };
}

// ---- Camera motion (task E3) ----

/**
 * The camera on its way from `from` to `to`, a fraction p (0..1, already eased) of the way: the
 * centre moves in a straight line, cx = cx0 + (cx1 − cx0)·p, and the scale geometrically,
 * s = s0·(s1/s0)^p, so every moment of a zoom multiplies the size by the same factor. Example:
 * from (0, 0, 2) to (100, 0, 8) at p = 0.5 → (50, 0, 4), the geometric middle of 2 and 8.
 */
export function easeCamera(from: Camera, to: Camera, p: number): Camera {
  return {
    cx: from.cx + (to.cx - from.cx) * p,
    cy: from.cy + (to.cy - from.cy) * p,
    scale: from.scale * (to.scale / from.scale) ** p,
  };
}

/** Ease-out: easeOut(u) = 1 − (1 − u)³ for u clamped to 0..1; fast at first, slow at the end. */
export function easeOut(u: number): number {
  return 1 - (1 - Math.min(1, Math.max(0, u))) ** 3;
}

/**
 * Where a following camera looks: d = min(0.28·v + 2, 0.22·min(w, h)/s) metres ahead of the car
 * at (x, y) with heading ψ and speed v (m/s), at s px per metre on a w × h canvas, so the road it
 * is about to drive is on screen; the second term keeps the car within 22 % of the shorter side
 * from the centre. Example: v = 20, s = 6, 1000 × 600 → d = 7.6 m; paused, d = 2 m at any zoom.
 */
export function lookAhead(x: number, y: number, psi: number, v: number, scale: number, size: Size): [number, number] {
  const metres = Math.min(0.28 * Math.max(0, v) + 2, (0.22 * Math.min(size.w, size.h)) / scale);
  return [x + Math.cos(psi) * metres, y + Math.sin(psi) * metres];
}

// ---- Mouse pan and zoom (task 4.10) ----

/** Smallest and largest scale (pixels per metre) the mouse may zoom to. */
export interface ScaleLimits {
  min: number;
  max: number;
}

/**
 * The zoom limits: MIN_SCALE to MAX_SCALE, but the lower limit drops to half the "whole track"
 * scale `fitScale` when that is smaller, so a small canvas can always zoom out to the whole track.
 */
export function scaleLimits(fitScale: number): ScaleLimits {
  return { min: Math.min(MIN_SCALE, fitScale / 2), max: MAX_SCALE };
}

/** One wheel event's zoom factor f = e^(-0.0015 Δ): a mouse notch (Δ = ±100) gives 0.86 or 1.16. */
export function wheelFactor(deltaY: number): number {
  return Math.exp(-0.0015 * deltaY);
}

/**
 * Zoom by `factor` around the screen point (px, py): the world point w under it stays under it.
 * w = screenToWorld(p); s' = s · factor (within `limits`); cx' = wx − (px − W/2)/s',
 * cy' = wy + (py − H/2)/s'. Example: 200 × 200 canvas, centre (0, 0), s = 2, cursor (150, 100),
 * factor 2 → s' = 4, centre (12.5, 0).
 */
export function zoomAt(
  cam: Camera,
  size: Size,
  px: number,
  py: number,
  factor: number,
  limits: ScaleLimits,
): Camera {
  const [wx, wy] = screenToWorld(cam, size, px, py);
  const scale = Math.min(limits.max, Math.max(limits.min, cam.scale * factor));
  return { cx: wx - (px - size.w / 2) / scale, cy: wy + (py - size.h / 2) / scale, scale };
}

/**
 * Move the view with a drag of (dx, dy) screen pixels: the track follows the mouse, so the centre
 * moves the other way, cx' = cx − dx/s and cy' = cy + dy/s (screen y points down).
 */
export function panBy(cam: Camera, dx: number, dy: number): Camera {
  return { cx: cam.cx - dx / cam.scale, cy: cam.cy + dy / cam.scale, scale: cam.scale };
}

/** The end point (world metres) of a ray of length `distance` at angle `rel` from the heading. */
export function rayEnd(
  x: number,
  y: number,
  psi: number,
  rel: number,
  distance: number,
): [number, number] {
  const angle = psi + rel;
  return [x + distance * Math.cos(angle), y + distance * Math.sin(angle)];
}

/**
 * The four corners (world metres) of a car drawn as a rectangle `length` long and `width` wide,
 * centred on (x, y) and pointing along `psi`: front-left, front-right, rear-right, rear-left.
 */
export function carCorners(
  x: number,
  y: number,
  psi: number,
  length: number,
  width: number,
): [number, number][] {
  const c = Math.cos(psi);
  const s = Math.sin(psi);
  const hl = length / 2;
  const hw = width / 2;
  // Local axes: forward (c, s) and left (-s, c).
  const corner = (forward: number, left: number): [number, number] => [
    x + forward * c - left * s,
    y + forward * s + left * c,
  ];
  return [corner(hl, hw), corner(hl, -hw), corner(-hl, -hw), corner(-hl, hw)];
}

/**
 * The index of the car closest to a click at screen (px, py), or null if none is within `radius`
 * pixels. `xs` and `ys` are the cars' world positions.
 */
export function nearestCar(
  cam: Camera,
  size: Size,
  xs: number[],
  ys: number[],
  px: number,
  py: number,
  radius: number,
): number | null {
  let best: number | null = null;
  let bestDistance = radius;
  for (let i = 0; i < xs.length; i++) {
    const [sx, sy] = worldToScreen(cam, size, xs[i], ys[i]);
    const d = Math.hypot(sx - px, sy - py);
    if (d <= bestDistance) {
      bestDistance = d;
      best = i;
    }
  }
  return best;
}
