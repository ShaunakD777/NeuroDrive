// Canvas 2D drawing of the world: road, cars and rays. It only turns what the server sent into
// pixels; the colours come from the page's CSS variables, so light and dark themes both work.

import type { FrameMessage, ScenarioMessage, TrackGeometry } from "../../api/generated";
import { maxAbs, rayWidth } from "../network/inspect";
import {
  type Camera,
  carCorners,
  rayEnd,
  type Size,
  worldToScreen,
} from "./geometry";

export interface Theme {
  bg: string;
  road: string;
  edge: string;
  text: string;
  muted: string;
  accent: string;
  ok: string;
  warn: string;
  bad: string;
}

/** Read the colours from the CSS variables in effect on `el`. */
export function readTheme(el: Element): Theme {
  const style = getComputedStyle(el);
  const get = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    bg: get("--bg", "#f6f7f9"),
    road: get("--surface", "#ffffff"),
    edge: get("--text-muted", "#5b6472"),
    text: get("--text", "#1b1f27"),
    muted: get("--border", "#d5d9e0"),
    accent: get("--accent", "#2563eb"),
    ok: get("--ok", "#15803d"),
    warn: get("--warn", "#b45309"),
    bad: get("--bad", "#b91c1c"),
  };
}

export interface Scene {
  size: Size;
  camera: Camera;
  geometry: TrackGeometry | null;
  scenario: ScenarioMessage | null;
  frame: FrameMessage | null;
  /** Index of the selected car. */
  selected: number;
  /** The small drawer version: no distance labels. */
  compact: boolean;
  /** Indices of rays to emphasise (the ones the selected neuron reads); empty for none. */
  highlight: number[];
  /**
   * The selected output's sensitivity to each of the five rays (task 2.10), or null. When set,
   * each ray is coloured by its sign (accent = the output rises as the ray gets longer, warn = it
   * falls) and drawn thicker the stronger it is, instead of the usual distance colours.
   */
  raySensitivity: number[] | null;
}

const CAR_LENGTH_EXTRA = 2.0; // car length = wheelbase + this, metres (display only)

/** Trace a closed loop of world points as a canvas sub-path. */
function loopPath(ctx: CanvasRenderingContext2D, scene: Scene, points: number[][]): void {
  points.forEach(([x, y], i) => {
    const [px, py] = worldToScreen(scene.camera, scene.size, x, y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
}

function drawRoad(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const geometry = scene.geometry;
  if (geometry === null) return;

  // The road is the ring between the two edges: fill both loops with the even-odd rule.
  ctx.beginPath();
  loopPath(ctx, scene, geometry.left);
  loopPath(ctx, scene, geometry.right);
  ctx.fillStyle = theme.road;
  ctx.fill("evenodd");

  ctx.lineWidth = 2;
  ctx.strokeStyle = theme.edge;
  ctx.beginPath();
  loopPath(ctx, scene, geometry.left);
  ctx.stroke();
  ctx.beginPath();
  loopPath(ctx, scene, geometry.right);
  ctx.stroke();

  // The centre line, dashed.
  ctx.save();
  ctx.setLineDash([8, 10]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = theme.muted;
  ctx.beginPath();
  loopPath(ctx, scene, geometry.centerline);
  ctx.stroke();
  ctx.restore();

  // The start line: across the road at the sample nearest the start arc length.
  const n = geometry.centerline.length;
  const i = Math.round((geometry.start_s0 / geometry.length) * n) % n;
  const [lx, ly] = worldToScreen(scene.camera, scene.size, geometry.left[i][0], geometry.left[i][1]);
  const [rx, ry] = worldToScreen(scene.camera, scene.size, geometry.right[i][0], geometry.right[i][1]);
  ctx.lineWidth = 3;
  ctx.strokeStyle = theme.text;
  ctx.beginPath();
  ctx.moveTo(lx, ly);
  ctx.lineTo(rx, ry);
  ctx.stroke();
}

function drawRays(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const { frame, scenario } = scene;
  if (frame === null || scenario === null || frame.selected === null) return;
  const i = frame.selected.index;
  const { x, y, psi } = frame.cars;
  ctx.font = "11px system-ui, sans-serif";
  ctx.textBaseline = "middle";
  frame.selected.rays.forEach((distance, k) => {
    const hit = distance < scenario.ray_range - 1e-6;
    const emphasised = scene.highlight.includes(k);
    ctx.globalAlpha = scene.highlight.length > 0 && !emphasised ? 0.3 : 1;
    const [ex, ey] = rayEnd(x[i], y[i], psi[i], scenario.ray_angles[k], distance);
    const [sx, sy] = worldToScreen(scene.camera, scene.size, x[i], y[i]);
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

function drawCars(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const { frame, scenario } = scene;
  if (frame === null || scenario === null) return;
  const length = scenario.wheelbase + CAR_LENGTH_EXTRA;
  const width = 2 * scenario.car_half_width;
  const { x, y, psi, alive } = frame.cars;
  for (let i = 0; i < x.length; i++) {
    const corners = carCorners(x[i], y[i], psi[i], length, width).map(([wx, wy]) =>
      worldToScreen(scene.camera, scene.size, wx, wy),
    );
    ctx.beginPath();
    corners.forEach(([px, py], k) => (k === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
    ctx.closePath();
    ctx.fillStyle = !alive[i] ? theme.bad : i === scene.selected ? theme.accent : theme.edge;
    ctx.fill();
    ctx.lineWidth = i === scene.selected ? 2 : 1;
    ctx.strokeStyle = theme.text;
    ctx.stroke();
    // The front edge (first two corners) is drawn heavier so the heading is visible.
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(corners[0][0], corners[0][1]);
    ctx.lineTo(corners[1][0], corners[1][1]);
    ctx.stroke();
  }
}

/** Draw the whole scene: background, road, rays of the selected car, then the cars. */
export function drawScene(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, scene.size.w, scene.size.h);
  drawRoad(ctx, theme, scene);
  drawRays(ctx, theme, scene);
  drawCars(ctx, theme, scene);
}
