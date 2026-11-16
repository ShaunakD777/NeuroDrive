// Camera maths. The first half is the production module (web/src/views/environment/geometry.ts)
// unchanged in behaviour; `easeCamera` and `lookAhead` are the two additions the redesign proposes.

export interface Size {
  w: number;
  h: number;
}

/** The world point at the centre of the canvas and how many pixels one metre is. */
export interface Camera {
  cx: number;
  cy: number;
  scale: number;
}

export const MIN_SCALE = 0.5;
export const MAX_SCALE = 40;

export function worldToScreen(cam: Camera, size: Size, x: number, y: number): [number, number] {
  return [size.w / 2 + (x - cam.cx) * cam.scale, size.h / 2 - (y - cam.cy) * cam.scale];
}

export function screenToWorld(cam: Camera, size: Size, px: number, py: number): [number, number] {
  return [cam.cx + (px - size.w / 2) / cam.scale, cam.cy - (py - size.h / 2) / cam.scale];
}

/** A camera that shows every point with `margin` pixels to spare on each named side. */
export function fitCamera(
  points: number[][],
  size: Size,
  margin: { top: number; right: number; bottom: number; left: number },
): Camera {
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
  const w = Math.max(1, size.w - margin.left - margin.right);
  const h = Math.max(1, size.h - margin.top - margin.bottom);
  const scale = Math.max(0.2, Math.min(w / Math.max(maxX - minX, 1e-6), h / Math.max(maxY - minY, 1e-6)));
  // The free area's centre is not the canvas centre when the margins differ: shift the world centre.
  const offX = (margin.left - margin.right) / 2 / scale;
  const offY = (margin.top - margin.bottom) / 2 / scale;
  return { cx: (minX + maxX) / 2 - offX, cy: (minY + maxY) / 2 + offY, scale };
}

export function zoomAt(cam: Camera, size: Size, px: number, py: number, factor: number, min: number): Camera {
  const [wx, wy] = screenToWorld(cam, size, px, py);
  const scale = Math.min(MAX_SCALE, Math.max(min, cam.scale * factor));
  return { cx: wx - (px - size.w / 2) / scale, cy: wy + (py - size.h / 2) / scale, scale };
}

export function panBy(cam: Camera, dx: number, dy: number): Camera {
  return { cx: cam.cx - dx / cam.scale, cy: cam.cy + dy / cam.scale, scale: cam.scale };
}

/** One wheel event's zoom factor: a mouse notch (Δ = ±100) gives 0.86 or 1.16 (as production). */
export function wheelFactor(deltaY: number): number {
  return Math.exp(-0.0015 * deltaY);
}

// ── Proposed additions ──────────────────────────────────────────────────────────────────

/**
 * The camera on its way from `from` to `to`, `p` of the way there (0..1, already eased). Position
 * moves in a straight line; the scale moves geometrically, so a zoom feels even all the way.
 */
export function easeCamera(from: Camera, to: Camera, p: number): Camera {
  return {
    cx: from.cx + (to.cx - from.cx) * p,
    cy: from.cy + (to.cy - from.cy) * p,
    scale: from.scale * (to.scale / from.scale) ** p,
  };
}

export const easeOut = (p: number) => 1 - (1 - Math.min(1, Math.max(0, p))) ** 3;

/**
 * Where a following camera looks: a little ahead of the car, further the faster it goes, so the
 * road the car is about to drive (and its rays) are on screen, not the road behind it.
 */
export function lookAhead(x: number, y: number, psi: number, v: number, scale: number, size: Size): [number, number] {
  const metres = Math.min(0.28 * v + 2, (Math.min(size.w, size.h) * 0.22) / scale);
  return [x + Math.cos(psi) * metres, y + Math.sin(psi) * metres];
}
