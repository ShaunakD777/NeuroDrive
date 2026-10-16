// Small pure helpers for the Neuron view: where to put a point on the activation-curve chart, and
// words for what the output means. The curve itself and every number on it come from the Python
// server; nothing here computes an activation.

export interface Box {
  /** Pixel position of the chart's top-left corner and its size. */
  left: number;
  top: number;
  width: number;
  height: number;
}

/** The z range the server samples the curves on (docs/ARCHITECTURE.md §7, GET /api/activations). */
export const Z_MIN = -5;
export const Z_MAX = 5;

export interface Bounds {
  min: number;
  max: number;
}

/** The value range to show on the chart's vertical axis: the curve's range plus a little space. */
export function curveBounds(values: number[]): Bounds {
  if (values.length === 0) return { min: -1, max: 1 };
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (max - min < 1e-9) {
    min -= 0.5;
    max += 0.5;
  }
  const pad = (max - min) * 0.12;
  return { min: min - pad, max: max + pad };
}

/** Horizontal pixel of z (clamped to the sampled range). */
export function plotX(z: number, box: Box): number {
  const t = (Math.min(Z_MAX, Math.max(Z_MIN, z)) - Z_MIN) / (Z_MAX - Z_MIN);
  return box.left + t * box.width;
}

/** Vertical pixel of an output value a (larger values are higher up the screen). */
export function plotY(a: number, bounds: Bounds, box: Box): number {
  const t = (a - bounds.min) / (bounds.max - bounds.min);
  return box.top + (1 - Math.min(1, Math.max(0, t))) * box.height;
}

/** The points of a polyline through the curve, as an SVG `points` string. */
export function curvePoints(z: number[], values: number[], bounds: Bounds, box: Box): string {
  return z
    .map((zi, i) => `${plotX(zi, box).toFixed(1)},${plotY(values[i], bounds, box).toFixed(1)}`)
    .join(" ");
}

/** True when z lies outside the sampled range, so the live dot is parked at the chart's edge. */
export function offChart(z: number): boolean {
  return z < Z_MIN || z > Z_MAX;
}

/** What the range of each activation is, in words (documentation, not a calculation). */
export const RANGE_TEXT: Record<string, string> = {
  tanh: "−1 to +1",
  sigmoid: "0 to 1 (never negative)",
  relu: "0 and up (never negative)",
  linear: "any value",
};

/** The steering command in words: "straight", "slightly left", "hard right", ... */
export function describeSteer(output: number): string {
  const size = Math.abs(output);
  if (size < 0.05) return "straight";
  const side = output < 0 ? "left" : "right";
  if (size < 0.3) return `slightly ${side}`;
  if (size < 0.7) return side;
  return `hard ${side}`;
}

/** The angle, in degrees, to turn the steering arrow: right (positive) is clockwise. */
export function dialAngle(output: number): number {
  return Math.min(1, Math.max(-1, output)) * 70;
}
