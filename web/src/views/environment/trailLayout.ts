// Pure helpers for the decision trail (task 2.11). They only pick and place numbers the server
// sent (where the car was, what it decided); nothing here simulates or computes the network.

import type { Trail } from "../../state/live";

/**
 * The value of output `m` (0 steer, 1 throttle) at each trail point, for colouring the trail, or
 * null when no output is selected (the trail is then drawn plain).
 */
export function trailValues(trail: Trail | null, m: number | null): number[] | null {
  if (trail === null || m === null) return null;
  return trail.action.map((a) => a[m] ?? 0);
}

/**
 * Colour and width for one trail segment of value `v` (−1..1): the sign picks the colour (the same
 * as the ray colours: positive = accent, negative = warn), the size picks the strength.
 */
export function trailStyle(v: number): { positive: boolean; width: number; alpha: number } {
  const size = Math.min(1, Math.abs(v));
  return { positive: v >= 0, width: 1.5 + 4 * size, alpha: 0.35 + 0.65 * size };
}

/**
 * The trail point nearest to (px, py) on screen, if one is within `radius` pixels: returns its step,
 * or null. `screen[i]` is where trail point i is drawn.
 */
export function nearestTrailStep(
  screen: [number, number][],
  steps: number[],
  px: number,
  py: number,
  radius: number,
): number | null {
  let best: number | null = null;
  let bestDistance = radius;
  screen.forEach(([sx, sy], i) => {
    const d = Math.hypot(sx - px, sy - py);
    if (d <= bestDistance) {
      bestDistance = d;
      best = steps[i];
    }
  });
  return best;
}

/** "4.0 s ago" for a step `back` steps before now, with physics step `dt`. */
export function agoText(back: number, dt: number): string {
  return `${(back * dt).toFixed(1)} s ago`;
}
