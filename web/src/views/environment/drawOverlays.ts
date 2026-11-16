// The selected car's data on the track, in CSS pixels (task E2, DESIGN_SPEC §7.3–7.4): its decision
// trail, its sensor fan and five rays, and the what-if paths. Every line has a dark casing under
// it, so it reads on asphalt, kerb and grass alike.

import { maxAbs, rayWidth } from "../network/inspect";
import { placeRayLabels, PILL_HEIGHT } from "./cars";
import { rayEnd, worldToScreen } from "./geometry";
import type { Scene, Theme } from "./scene";
import { trailStyle } from "./trailLayout";

const CASING = "rgba(8, 10, 14, 0.42)";
const NUMBER_FONT = "500 11px 'IBM Plex Mono', ui-monospace, monospace";
/** Pixels over which a ray fades in from the car, so the car itself stays readable. */
const RAY_FADE = 26;

/**
 * A rounded label pill at (x, y): x is its left edge (or its middle with `align: "center"`), y its
 * vertical centre. Used for the ray labels and the cars' tags.
 */
export function pill(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  options: { bg: string; fg: string; align?: "left" | "center"; font?: string; border?: string; height?: number },
): void {
  ctx.font = options.font ?? NUMBER_FONT;
  const h = options.height ?? PILL_HEIGHT;
  const w = Math.max(h, Math.ceil(ctx.measureText(text).width) + (h < PILL_HEIGHT ? 8 : 10));
  const left = options.align === "center" ? x - w / 2 : x;
  ctx.beginPath();
  ctx.roundRect(left, y - h / 2, w, h, 4);
  ctx.fillStyle = options.bg;
  ctx.fill();
  if (options.border) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = options.border;
    ctx.stroke();
  }
  ctx.fillStyle = options.fg;
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillText(text, left + 5, y + 0.5);
}

/**
 * The selected car's five rays: a faint fan through their ends (the field of view), then each ray
 * cased in dark and fading in from the car, a ringed dot where it hits, and a label pill placed so
 * neighbours do not stack (`placeRayLabels`). Colour by distance d of range R: d/R < 0.2 danger,
 * < 0.5 near, otherwise clear; no hit is dashed and labelled "≥50 m". With an output selected the
 * colour is the sign of its sensitivity and the width its size.
 */
export function drawRays(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const { frame, scenario } = scene;
  if (frame === null || scenario === null || frame.selected === null) return;
  const i = frame.selected.index;
  // Looking at an earlier step: the rays the car had then, from where it was then.
  const m = scene.moment;
  const x = m === null ? frame.cars.x[i] : m.x;
  const y = m === null ? frame.cars.y[i] : m.y;
  const psi = m === null ? frame.cars.psi[i] : m.psi;
  if (x === undefined || y === undefined || psi === undefined) return;
  const distances = m === null ? frame.selected.rays : m.rays;
  const [sx, sy] = worldToScreen(scene.camera, scene.size, x, y);
  const ends = distances.map((distance, k) => {
    const [ex, ey] = rayEnd(x, y, psi, scenario.ray_angles[k], distance);
    return worldToScreen(scene.camera, scene.size, ex, ey);
  });
  const sensitivity = scene.raySensitivity;
  const strongest = sensitivity === null ? 1 : maxAbs(sensitivity);

  // The field of view.
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ends.forEach(([ex, ey]) => ctx.lineTo(ex, ey));
  ctx.closePath();
  ctx.fillStyle = "rgba(255, 255, 255, 0.055)";
  ctx.fill();

  ctx.lineCap = "round";
  const hits = distances.map((d) => d < scenario.ray_range - 1e-6);
  distances.forEach((distance, k) => {
    const [ex, ey] = ends[k];
    const hit = hits[k];
    const emphasised = scene.highlight.includes(k);
    const s = sensitivity?.[k];
    const fraction = distance / scenario.ray_range;
    let colour: string;
    let width: number;
    if (s !== undefined) {
      colour = s < 0 ? theme.neg : theme.pos;
      width = rayWidth(s, strongest);
    } else {
      colour = !hit ? theme.rayNone : fraction < 0.2 ? theme.rayDanger : fraction < 0.5 ? theme.rayNear : theme.rayClear;
      width = emphasised ? 4.5 : hit ? 2 : 1.2;
    }
    ctx.globalAlpha = scene.highlight.length > 0 && !emphasised ? 0.28 : 1;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = "rgba(8, 10, 14, 0.35)";
    ctx.lineWidth = width + 2.5;
    ctx.stroke();
    // Transparent at the car, full colour after RAY_FADE px (or 35 % of a short ray).
    const length = Math.hypot(ex - sx, ey - sy);
    const gradient = ctx.createLinearGradient(sx, sy, ex, ey);
    gradient.addColorStop(0, "rgba(255, 255, 255, 0)");
    gradient.addColorStop(Math.min(0.35, RAY_FADE / Math.max(RAY_FADE, length)), colour);
    gradient.addColorStop(1, colour);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = width;
    if (!hit && s === undefined) ctx.setLineDash([5, 6]);
    ctx.stroke();
    ctx.setLineDash([]);
    if (hit) {
      ctx.beginPath();
      ctx.arc(ex, ey, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(8, 10, 14, 0.75)";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(ex, ey, 3, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  });

  // The labels: the distance, or the slope "s +0.57" while an output is selected.
  const labels = distances.map((distance, k) => {
    const s = sensitivity?.[k];
    if (s !== undefined) return `s ${s >= 0 ? "+" : "−"}${Math.abs(s).toFixed(2)}`;
    return hits[k] ? `${distance.toFixed(1)} m` : `≥${scenario.ray_range.toFixed(0)} m`;
  });
  ctx.font = NUMBER_FONT;
  const widths = labels.map((label) => Math.ceil(ctx.measureText(label).width) + 10);
  const spots = placeRayLabels([sx, sy], ends, widths, scene.highlight, scene.compact);
  spots.forEach((spot, k) => {
    if (spot === null) return;
    ctx.globalAlpha = scene.highlight.length > 0 && !scene.highlight.includes(k) ? 0.28 : 1;
    pill(ctx, labels[k], spot.x, spot.y, {
      bg: theme.hudBg,
      fg: hits[k] || sensitivity !== null ? theme.hudText : theme.hudMuted,
    });
  });
  ctx.globalAlpha = 1;
}

/** A world polyline as a screen path (not stroked). */
function polyline(ctx: CanvasRenderingContext2D, scene: Scene, points: number[][]): void {
  ctx.beginPath();
  points.forEach(([x, y], k) => {
    const [px, py] = worldToScreen(scene.camera, scene.size, x, y);
    if (k === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
}

/**
 * The two held-action paths (under the cars): the real action dashed white, the what-if action
 * solid in the selection colour, both cased in dark; a dot where the real one ends.
 */
export function drawGhostPaths(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const ghost = scene.ghost;
  if (ghost === null) return;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  polyline(ctx, scene, ghost.real);
  ctx.strokeStyle = "rgba(8, 10, 14, 0.4)";
  ctx.lineWidth = 5;
  ctx.stroke();
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.setLineDash([]);
  polyline(ctx, scene, ghost.what_if);
  ctx.strokeStyle = "rgba(8, 10, 14, 0.4)";
  ctx.lineWidth = 6;
  ctx.stroke();
  ctx.strokeStyle = theme.select;
  ctx.lineWidth = 3;
  ctx.stroke();
  const [rx, ry] = worldToScreen(scene.camera, scene.size, ghost.real_end[0], ghost.real_end[1]);
  ctx.beginPath();
  ctx.arc(rx, ry, 4.5, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(8, 10, 14, 0.6)";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(rx, ry, 3, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
  ctx.fill();
}

/**
 * The decision trail: a dark casing along the remembered path, then either one line in the
 * selection colour (no output selected) or one segment per step coloured by the output's sign
 * and sized by its value (`trailStyle`). A gap in the remembered steps leaves a gap in the line.
 */
export function drawTrail(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const trail = scene.trail;
  if (trail === null || trail.points.length < 2) return;
  const screen = trail.points.map(([x, y]) => worldToScreen(scene.camera, scene.size, x, y));
  const joined = (k: number) => trail.steps[k + 1] === trail.steps[k] + 1; // nothing missing between
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // One path for the whole trail, broken at gaps.
  ctx.beginPath();
  screen.forEach(([px, py], k) => {
    if (k === 0 || !joined(k - 1)) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = CASING;
  ctx.lineWidth = trail.values === null ? 4.5 : 7.5;
  ctx.stroke();
  if (trail.values === null) {
    ctx.strokeStyle = theme.select;
    ctx.lineWidth = 2.25;
    ctx.stroke();
    return;
  }
  for (let k = 0; k + 1 < screen.length; k++) {
    if (!joined(k)) continue;
    const style = trailStyle(trail.values[k] ?? 0);
    ctx.globalAlpha = style.alpha;
    ctx.lineWidth = style.width;
    ctx.strokeStyle = style.positive ? theme.pos : theme.neg;
    ctx.beginPath();
    ctx.moveTo(screen[k][0], screen[k][1]);
    ctx.lineTo(screen[k + 1][0], screen[k + 1][1]);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}
