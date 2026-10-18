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

/**
 * A camera that shows every point of `points` (each [x, y]) with `margin` pixels to spare: centred
 * on the middle of their bounding box, at the largest scale that still fits both directions.
 */
export function fitCamera(points: number[][], size: Size, margin = 24): Camera {
  if (points.length === 0 || size.w <= 0 || size.h <= 0) return { cx: 0, cy: 0, scale: 1 };
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
  const scale = Math.min((size.w - 2 * margin) / spanX, (size.h - 2 * margin) / spanY);
  return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, scale: clampScale(scale) };
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
