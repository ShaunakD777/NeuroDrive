// The world under the cars (task E1, lifted from the redesign prototype's `scene.ts`): drawn in
// metres under the world transform that `drawScene` sets (y up), so a kerb 0.95 m wide is "0.95".
// Lines that must stay visible at any zoom are given at least one or two screen pixels, using
// px = 1 / scale, the size of one screen pixel in metres. The distance boards are labels and are
// drawn afterwards in pixels. Nothing here is data: the road the cars feel is the two edge loops,
// and the white edge lines are drawn exactly on them.

import type { TrackGeometry } from "../../api/generated";
import { type Camera, type Size, worldToScreen } from "./geometry";
import type { Theme } from "./scene";
import type { TrackArt } from "./trackArt";

// ---- Textures: small noise tiles from a seeded generator, made once per theme ----
//
// Measured in task E1: a pattern filled under the world transform (scaled, and for the stripes
// rotated) is filtered pixel by pixel and cost 2–15 ms per fill. So the grain is filled in device
// pixels at a whole-pixel offset that moves with the world (Chromium copies such a pattern without
// filtering; the grain keeps one size at every zoom, which suits a grain), and the mowing stripes are
// geometry: plain bands filled with one colour.

interface Textures {
  asphalt: CanvasPattern | null;
  grass: CanvasPattern | null;
}

const TILE = 160; // device pixels per noise tile

let textures: { key: string; value: Textures } | null = null;

/**
 * A square tile of `dots` light and dark specks. The positions come from a linear congruential
 * generator with a fixed seed (state ← (1664525·state + 1013904223) mod 2³²), so the grain is the
 * same on every load and in every screenshot.
 */
function noiseTile(size: number, dots: number, seed: number, light: number, dark: number): HTMLCanvasElement {
  const tile = document.createElement("canvas");
  tile.width = size;
  tile.height = size;
  const g = tile.getContext("2d");
  if (g === null) return tile;
  let state = seed;
  const random = () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
  for (let i = 0; i < dots; i++) {
    const bright = random() < 0.5;
    const alpha = (bright ? light : dark) * (0.4 + 0.6 * random());
    g.fillStyle = bright ? `rgba(255,255,255,${alpha})` : `rgba(0,0,0,${alpha})`;
    const r = random() < 0.85 ? 1 : 2;
    g.fillRect(Math.floor(random() * size), Math.floor(random() * size), r, r);
  }
  return tile;
}

function texturesFor(ctx: CanvasRenderingContext2D, theme: Theme): Textures {
  if (textures !== null && textures.key === theme.key) return textures.value;
  const asphalt = ctx.createPattern(noiseTile(TILE, 5200, 7, theme.night ? 0.05 : 0.075, 0.16), "repeat");
  const grass = ctx.createPattern(noiseTile(TILE, 2600, 23, theme.night ? 0.03 : 0.06, 0.1), "repeat");
  textures = { key: theme.key, value: { asphalt, grass } };
  return textures.value;
}

/**
 * Fill the current path (already built under the world transform: a canvas path keeps the points
 * as they were transformed when added) with a grain pattern in device pixels. The pattern is
 * shifted by the world origin's device position, rounded to whole pixels and taken modulo the
 * tile, so the grain moves with the track when the camera pans.
 */
function fillGrain(ctx: CanvasRenderingContext2D, pattern: CanvasPattern, rule: CanvasFillRule = "nonzero"): void {
  const t = ctx.getTransform();
  pattern.setTransform(new DOMMatrix([1, 0, 0, 1, Math.round(t.e) % TILE, Math.round(t.f) % TILE]));
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = pattern;
  ctx.fill(rule);
  ctx.restore();
}

// Mowing stripes: bands STRIPE metres wide every 2·STRIPE metres, across the direction STRIPE_ANGLE.
const STRIPE = 9;
const STRIPE_ANGLE = (28 * Math.PI) / 180;

/**
 * The stripes over the visible rectangle as one path of parallelograms. With u = x·cos θ + y·sin θ
 * (distance along the mowing direction θ) and v = −x·sin θ + y·cos θ (across it), stripe m covers
 * 2·STRIPE·m ≤ u < 2·STRIPE·m + STRIPE; each is cut to the rectangle's range of v and turned back
 * into x = u·cos θ − v·sin θ, y = u·sin θ + v·cos θ.
 */
function stripesPath(ctx: CanvasRenderingContext2D, x0: number, y0: number, w: number, h: number): void {
  const cos = Math.cos(STRIPE_ANGLE);
  const sin = Math.sin(STRIPE_ANGLE);
  const corners = [
    [x0, y0],
    [x0 + w, y0],
    [x0, y0 + h],
    [x0 + w, y0 + h],
  ];
  const us = corners.map(([x, y]) => x * cos + y * sin);
  const vs = corners.map(([x, y]) => -x * sin + y * cos);
  const [v0, v1] = [Math.min(...vs), Math.max(...vs)];
  const point = (u: number, v: number): [number, number] => [u * cos - v * sin, u * sin + v * cos];
  ctx.beginPath();
  for (let m = Math.floor(Math.min(...us) / (2 * STRIPE)); 2 * STRIPE * m < Math.max(...us); m++) {
    const u0 = 2 * STRIPE * m;
    const quad = [point(u0, v0), point(u0 + STRIPE, v0), point(u0 + STRIPE, v1), point(u0, v1)];
    loop(ctx, quad);
  }
}

/** Trace a closed loop of points as a sub-path (in whatever units the transform says). */
function loop(ctx: CanvasRenderingContext2D, points: number[][]): void {
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.closePath();
}

// Zoom thresholds, pixels per metre: below them a layer would be noise or invisible.
const GRAIN_SCALE = 1.2; // grass and sand grain
const ASPHALT_GRAIN_SCALE = 1.6;
const DASH_SCALE = 1.4; // centre dashes
const ARROW_SCALE = 3; // direction arrows
const BOARD_SCALE = 1.1; // distance boards

/** The world in ten layers. Expects the world transform (metres, y up) to be set. */
export function drawWorld(
  ctx: CanvasRenderingContext2D,
  theme: Theme,
  track: TrackGeometry,
  art: TrackArt,
  camera: Camera,
  size: Size,
): void {
  const tex = texturesFor(ctx, theme);
  const px = 1 / camera.scale; // one screen pixel in metres

  // 1. Grass: base, mowing stripes, fine grain, over exactly the visible rectangle.
  const x0 = camera.cx - size.w / 2 / camera.scale;
  const y0 = camera.cy - size.h / 2 / camera.scale;
  const w = size.w / camera.scale;
  const h = size.h / camera.scale;
  ctx.fillStyle = theme.grass;
  ctx.fillRect(x0, y0, w, h);
  stripesPath(ctx, x0, y0, w, h);
  ctx.fillStyle = theme.grass2;
  ctx.fill();
  if (tex.grass && camera.scale > GRAIN_SCALE) {
    ctx.beginPath();
    ctx.rect(x0, y0, w, h);
    fillGrain(ctx, tex.grass);
  }

  // 2. The mown verge: a lighter band 5.5 m wide either side of the road.
  ctx.lineJoin = "round";
  ctx.beginPath();
  loop(ctx, track.centerline);
  ctx.strokeStyle = theme.verge;
  ctx.lineWidth = track.width + 11;
  ctx.stroke();

  // 3. Sand traps outside the tight bends.
  ctx.fillStyle = theme.sand;
  for (const polygon of art.sand) {
    ctx.beginPath();
    loop(ctx, polygon);
    ctx.fill();
  }
  if (tex.grass && camera.scale > GRAIN_SCALE && art.sand.length > 0) {
    ctx.beginPath();
    for (const polygon of art.sand) loop(ctx, polygon);
    fillGrain(ctx, tex.grass);
  }

  // 4. A dark lip under the road edge, then the asphalt itself: the ring between the two edge
  //    loops, filled with the even-odd rule (inside one loop and not the other).
  ctx.beginPath();
  loop(ctx, track.centerline);
  ctx.strokeStyle = theme.asphalt2;
  ctx.lineWidth = track.width + 1.1;
  ctx.stroke();
  ctx.beginPath();
  loop(ctx, track.left);
  loop(ctx, track.right);
  ctx.fillStyle = theme.asphalt;
  ctx.fill("evenodd");
  if (tex.asphalt && camera.scale > ASPHALT_GRAIN_SCALE) fillGrain(ctx, tex.asphalt, "evenodd");

  // 5. The rubbered-in line down the middle: three soft dark strokes.
  ctx.beginPath();
  loop(ctx, track.centerline);
  for (const [width, alpha] of [
    [5.6, 0.05],
    [3.6, 0.06],
    [1.9, 0.06],
  ]) {
    ctx.strokeStyle = `rgba(0,0,0,${alpha})`;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  // 6. Kerbs: alternating blocks outside the edge, one path per colour.
  for (const tone of [0, 1]) {
    ctx.beginPath();
    for (const quad of art.kerbs) if (quad.tone === tone) loop(ctx, quad.points);
    ctx.fillStyle = tone === 0 ? theme.kerbA : theme.kerbB;
    ctx.fill();
  }

  // 7. The white edge lines, exactly on the edges the rays and the crash test use.
  ctx.strokeStyle = theme.line;
  ctx.lineWidth = Math.max(0.26, 1.4 * px);
  for (const edge of [track.left, track.right]) {
    ctx.beginPath();
    loop(ctx, edge);
    ctx.stroke();
  }

  // 8. The centre line (heading error and lateral offset are measured from it).
  if (camera.scale > DASH_SCALE) {
    ctx.beginPath();
    loop(ctx, track.centerline);
    ctx.setLineDash([2.6, 5.4]);
    ctx.strokeStyle = theme.marking;
    ctx.lineWidth = Math.max(0.16, 1 * px);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 9. Direction arrows: two chevrons pointing the way the cars drive.
  if (camera.scale > ARROW_SCALE) {
    ctx.strokeStyle = theme.marking;
    ctx.lineWidth = 0.24;
    ctx.lineCap = "round";
    for (const a of art.arrows) {
      ctx.save();
      ctx.translate(a.x, a.y);
      ctx.rotate(a.heading);
      ctx.beginPath();
      for (const dx of [0, 1.3]) {
        ctx.moveTo(-0.9 + dx, 1.1);
        ctx.lineTo(0.1 + dx, 0);
        ctx.lineTo(-0.9 + dx, -1.1);
      }
      ctx.stroke();
      ctx.restore();
    }
    ctx.lineCap = "butt";
  }

  // 10. The start/finish line: two rows of 12 checks across the road, one either side of it.
  const { left, right, tangent } = art.start;
  const columns = 12;
  const ax = (left[0] - right[0]) / columns;
  const ay = (left[1] - right[1]) / columns;
  const cell = Math.hypot(ax, ay);
  for (let row = 0; row < 2; row++) {
    for (let col = 0; col < columns; col++) {
      const ox = right[0] + ax * col + tangent[0] * cell * (row - 1);
      const oy = right[1] + ay * col + tangent[1] * cell * (row - 1);
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + ax, oy + ay);
      ctx.lineTo(ox + ax + tangent[0] * cell, oy + ay + tangent[1] * cell);
      ctx.lineTo(ox + tangent[0] * cell, oy + tangent[1] * cell);
      ctx.closePath();
      ctx.fillStyle = (row + col) % 2 === 0 ? theme.line : theme.chequer;
      ctx.fill();
    }
  }
}

/**
 * Distance boards (where 100 m, 200 m, … of a lap are; progress in the HUD is in metres), in CSS
 * pixels, so they keep their size at every zoom. Boards off the canvas are skipped.
 */
export function drawBoards(ctx: CanvasRenderingContext2D, theme: Theme, art: TrackArt, camera: Camera, size: Size): void {
  if (camera.scale <= BOARD_SCALE) return;
  ctx.font = "600 10px 'IBM Plex Sans Condensed', system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const board of art.boards) {
    const [px, py] = worldToScreen(camera, size, board.x, board.y);
    if (px < -40 || py < -20 || px > size.w + 40 || py > size.h + 20) continue;
    const text = `${board.metres}`;
    const w = ctx.measureText(text).width + 8;
    ctx.beginPath();
    ctx.roundRect(px - w / 2, py - 7, w, 14, 2);
    ctx.fillStyle = theme.board;
    ctx.fill();
    ctx.fillStyle = theme.boardInk;
    ctx.fillText(text, px, py + 0.5);
  }
  ctx.textAlign = "left";
}

// ---- The plain map: the pre-redesign road, kept for the Datasets view until task X4 ----

export interface MapTheme {
  bg: string;
  road: string;
  edge: string;
  text: string;
  muted: string;
}

export function readMapTheme(el: Element): MapTheme {
  const style = getComputedStyle(el);
  const get = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    bg: get("--map-bg", "#f6f7f9"),
    road: get("--map-road", "#ffffff"),
    edge: get("--map-edge", "#5b6472"),
    text: get("--map-text", "#1b1f27"),
    muted: get("--map-muted", "#d5d9e0"),
  };
}

/** Trace a closed loop of world points as a screen-space sub-path. */
function screenLoop(ctx: CanvasRenderingContext2D, camera: Camera, size: Size, points: number[][]): void {
  points.forEach(([x, y], i) => {
    const [px, py] = worldToScreen(camera, size, x, y);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
}

/**
 * The track as a plain diagram, exactly as the Environment drew it before task E1: background,
 * white ring between the edges, the two edges, the dashed centre line and a line at the start.
 */
export function drawMap(
  ctx: CanvasRenderingContext2D,
  theme: MapTheme,
  size: Size,
  camera: Camera,
  geometry: TrackGeometry | null,
): void {
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, size.w, size.h);
  if (geometry === null) return;

  // The road is the ring between the two edges: fill both loops with the even-odd rule.
  ctx.beginPath();
  screenLoop(ctx, camera, size, geometry.left);
  screenLoop(ctx, camera, size, geometry.right);
  ctx.fillStyle = theme.road;
  ctx.fill("evenodd");

  ctx.lineWidth = 2;
  ctx.strokeStyle = theme.edge;
  ctx.beginPath();
  screenLoop(ctx, camera, size, geometry.left);
  ctx.stroke();
  ctx.beginPath();
  screenLoop(ctx, camera, size, geometry.right);
  ctx.stroke();

  // The centre line, dashed.
  ctx.save();
  ctx.setLineDash([8, 10]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = theme.muted;
  ctx.beginPath();
  screenLoop(ctx, camera, size, geometry.centerline);
  ctx.stroke();
  ctx.restore();

  // The start line: across the road at the sample nearest the start arc length.
  const n = geometry.centerline.length;
  const i = Math.round((geometry.start_s0 / geometry.length) * n) % n;
  const [lx, ly] = worldToScreen(camera, size, geometry.left[i][0], geometry.left[i][1]);
  const [rx, ry] = worldToScreen(camera, size, geometry.right[i][0], geometry.right[i][1]);
  ctx.lineWidth = 3;
  ctx.strokeStyle = theme.text;
  ctx.beginPath();
  ctx.moveTo(lx, ly);
  ctx.lineTo(rx, ry);
  ctx.stroke();
}
