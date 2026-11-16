// The selected car's data on the track, in CSS pixels: its decision trail, its five rays and the
// what-if paths. Moved unchanged from the pre-redesign `draw.ts` in task E1 (only their colours now
// come from light-on-dark tokens, so they read on the dark asphalt); task E2 restyles them.

import { maxAbs, rayWidth } from "../network/inspect";
import { rayEnd, worldToScreen } from "./geometry";
import type { Scene, Theme } from "./scene";
import { trailStyle } from "./trailLayout";

export function drawRays(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const { frame, scenario } = scene;
  if (frame === null || scenario === null || frame.selected === null) return;
  const i = frame.selected.index;
  // Looking at an earlier step: the rays the car had then, from where it was then.
  const m = scene.moment;
  const x = m === null ? frame.cars.x[i] : m.x;
  const y = m === null ? frame.cars.y[i] : m.y;
  const psi = m === null ? frame.cars.psi[i] : m.psi;
  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "middle";
  (m === null ? frame.selected.rays : m.rays).forEach((distance, k) => {
    const hit = distance < scenario.ray_range - 1e-6;
    const emphasised = scene.highlight.includes(k);
    ctx.globalAlpha = scene.highlight.length > 0 && !emphasised ? 0.3 : 1;
    const [ex, ey] = rayEnd(x, y, psi, scenario.ray_angles[k], distance);
    const [sx, sy] = worldToScreen(scene.camera, scene.size, x, y);
    const [tx, ty] = worldToScreen(scene.camera, scene.size, ex, ey);
    const fraction = distance / scenario.ray_range;
    const s = scene.raySensitivity?.[k];
    if (s !== undefined && scene.raySensitivity !== null) {
      ctx.lineWidth = rayWidth(s, maxAbs(scene.raySensitivity));
      ctx.strokeStyle = s < 0 ? theme.warn : theme.accent;
    } else {
      ctx.lineWidth = emphasised ? 5 : hit ? 2 : 1;
      ctx.strokeStyle = !hit ? theme.muted : fraction < 0.2 ? theme.bad : fraction < 0.5 ? theme.warn : theme.ok;
    }
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(tx, ty);
    ctx.stroke();
    if (hit) {
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath();
      ctx.arc(tx, ty, 3, 0, Math.PI * 2);
      ctx.fill();
      if (!scene.compact || emphasised) {
        ctx.fillStyle = theme.text;
        // In sensitivity mode the slope replaces the distance (the HUD still lists the distances).
        const label = s === undefined ? `${distance.toFixed(1)} m` : `s ${s >= 0 ? "+" : "−"}${Math.abs(s).toFixed(2)}`;
        ctx.fillText(label, tx + 6, ty);
      }
    }
    ctx.globalAlpha = 1;
  });
}

function polyline(ctx: CanvasRenderingContext2D, scene: Scene, points: number[][]): void {
  ctx.beginPath();
  points.forEach(([x, y], k) => {
    const [px, py] = worldToScreen(scene.camera, scene.size, x, y);
    if (k === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
}

/** The two held-action paths (under the cars). */
export function drawGhostPaths(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const ghost = scene.ghost;
  if (ghost === null) return;
  ctx.lineCap = "round";
  ctx.lineWidth = 2;
  ctx.strokeStyle = theme.edge;
  ctx.setLineDash([6, 5]);
  polyline(ctx, scene, ghost.real);
  ctx.setLineDash([]);
  ctx.lineWidth = 3;
  ctx.strokeStyle = theme.accent;
  polyline(ctx, scene, ghost.what_if);
  // A small mark where holding the real action would end.
  const [rx, ry] = worldToScreen(scene.camera, scene.size, ghost.real_end[0], ghost.real_end[1]);
  ctx.fillStyle = theme.edge;
  ctx.beginPath();
  ctx.arc(rx, ry, 3, 0, Math.PI * 2);
  ctx.fill();
}

/** The decision trail: one segment per pair of consecutive steps, coloured by the output. */
export function drawTrail(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const trail = scene.trail;
  if (trail === null || trail.points.length < 2) return;
  ctx.lineCap = "round";
  const screen = trail.points.map(([x, y]) => worldToScreen(scene.camera, scene.size, x, y));
  for (let k = 0; k + 1 < screen.length; k++) {
    if (trail.steps[k + 1] !== trail.steps[k] + 1) continue; // a gap: nothing remembered between
    const v = trail.values?.[k];
    if (v === undefined) {
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 2;
      ctx.strokeStyle = theme.edge;
    } else {
      const style = trailStyle(v);
      ctx.globalAlpha = style.alpha;
      ctx.lineWidth = style.width;
      ctx.strokeStyle = style.positive ? theme.accent : theme.warn;
    }
    ctx.beginPath();
    ctx.moveTo(screen[k][0], screen[k][1]);
    ctx.lineTo(screen[k + 1][0], screen[k + 1][1]);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}
