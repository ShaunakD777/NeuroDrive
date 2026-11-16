// Canvas 2D drawing of the Environment: it only turns what the server sent into pixels. Every
// colour comes from CSS variables (`styles/tokens.css`), so the light and dark themes both work;
// the world has its own day and night palette (task E1, ADR-064).
//
// The layers, bottom to top (task E1; the order of DESIGN_SPEC §7.1):
//
//   WORLD, drawn in metres    grass → verge → sand → asphalt → rubber line → kerbs → edge lines →
//                             centre dashes → arrows → start line            (drawWorld.ts)
//   BOARDS, in pixels         distance boards every 100 m                     (drawWorld.ts)
//   DATA, in pixels           decision trail → rays → what-if paths           (drawOverlays.ts)
//   CARS, in pixels           cars → race ghost → look-back car → what-if car (drawCars.ts)
//
// Rule of the scene: decoration never sits above data, and everything a car or a ray can touch is
// inside the two white edge lines. Cars and data layers keep their pre-redesign shapes until E2.

import type { FrameMessage, GhostPaths, ScenarioMessage, TrackGeometry } from "../../api/generated";
import { drawCars, drawGhostCar, drawMoment, drawRaceGhost } from "./drawCars";
import { drawGhostPaths, drawRays, drawTrail } from "./drawOverlays";
import { drawBoards, drawWorld } from "./drawWorld";
import type { Camera, Size } from "./geometry";
import { trackArt } from "./trackArt";

export interface Theme {
  /** True for the night world (dark page theme). */
  night: boolean;
  /** Changes whenever a world colour changes, so textures are rebuilt only then. */
  key: string;
  // The world (task E1).
  grass: string;
  grass2: string;
  verge: string;
  sand: string;
  asphalt: string;
  asphalt2: string;
  line: string;
  chequer: string;
  kerbA: string;
  kerbB: string;
  marking: string;
  board: string;
  boardInk: string;
  // Cars and data layers, the pre-redesign roles in light-on-dark colours until E2.
  edge: string;
  text: string;
  muted: string;
  accent: string;
  ok: string;
  warn: string;
  bad: string;
  /** The GA rank bands, best fifth first (task 7.6). */
  rank: string[];
}

const RANK_FALLBACK = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf"];

/** Read the colours from the CSS variables in effect on `el` (read again on every draw). */
export function readTheme(el: Element): Theme {
  const style = getComputedStyle(el);
  const get = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  const world = {
    grass: get("--w-grass", "#5c8650"),
    grass2: get("--w-grass-2", "#557c4a"),
    verge: get("--w-verge", "#729c62"),
    sand: get("--w-sand", "#c9ba8c"),
    asphalt: get("--w-asphalt", "#383c43"),
    asphalt2: get("--w-asphalt-2", "#30343a"),
    line: get("--w-line", "#f3f4f1"),
    chequer: get("--w-chequer", "#16181c"),
    kerbA: get("--w-kerb-a", "#d8443b"),
    kerbB: get("--w-kerb-b", "#f3f4f1"),
    marking: get("--w-marking", "rgba(255, 255, 255, 0.3)"),
    board: get("--w-board", "#f3f4f1"),
    boardInk: get("--w-board-ink", "#1b1f27"),
  };
  const night = get("--w-night", "0") === "1";
  return {
    night,
    key: `${night}|${Object.values(world).join("|")}`,
    ...world,
    edge: get("--w-edge", "#9aa3b2"),
    text: get("--w-text", "#e8ebf0"),
    muted: get("--w-muted", "rgba(255, 255, 255, 0.34)"),
    accent: get("--w-accent", "#6aa0ff"),
    ok: get("--w-ok", "#4ade80"),
    warn: get("--w-warn", "#fbbf24"),
    bad: get("--w-bad", "#f87171"),
    rank: RANK_FALLBACK.map((fallback, k) => get(`--car-rank-${k + 1}`, fallback)),
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
  /** The small drawer version: no distance labels or boards. */
  compact: boolean;
  /** Indices of rays to emphasise (the ones the selected neuron reads); empty for none. */
  highlight: number[];
  /**
   * The selected output's sensitivity to each of the five rays (task 2.10), or null. When set,
   * each ray is coloured by its sign (accent = the output rises as the ray gets longer, warn = it
   * falls) and drawn thicker the stronger it is, instead of the usual distance colours.
   */
  raySensitivity: number[] | null;
  /**
   * The what-if ghost (ADR-026), or null: the selected car's next second if it held its real
   * action (dashed path) or the what-if action (solid path, ending in a see-through ghost car).
   */
  ghost: GhostPaths | null;
  /**
   * The selected car's decision trail (task 2.11), or null: where it was at each remembered step
   * (`steps[i]` at `points[i]`), and, when an output is selected, that output's value there
   * (`values`), which colours the trail; null values draw it plain.
   */
  trail: { steps: number[]; points: number[][]; values: number[] | null } | null;
  /**
   * The earlier moment being looked at (task 2.11), or null: the car's pose and rays then. Its rays
   * replace the live ones and an outlined car marks where it was.
   */
  moment: { step: number; x: number; y: number; psi: number; rays: number[] } | null;
}

/**
 * The canvas transform (a, b, c, d, e, f) that draws world metres where `worldToScreen` puts them,
 * in CSS pixels: X = a·x + c·y + e, Y = b·x + d·y + f with a = s, d = −s (y is flipped),
 * e = w/2 − cx·s and f = h/2 + cy·s, where (cx, cy) is the camera centre, s its pixels per metre
 * and w × h the canvas in CSS pixels. `drawScene` multiplies it onto the caller's transform (the
 * device-pixel ratio), so world and pixel layers agree exactly.
 */
export function worldTransform(camera: Camera, size: Size): [number, number, number, number, number, number] {
  const s = camera.scale;
  return [s, 0, 0, -s, size.w / 2 - camera.cx * s, size.h / 2 + camera.cy * s];
}

/**
 * The still world of one canvas, kept as a picture (task E1). Measured: the textured world costs up
 * to ~28 ms a frame on a ratio-2 screen with software drawing, the cars about 1–4 ms. While the
 * camera, size, theme and track stay the same (Whole track, a free view, a paused follow) the
 * picture is copied instead of drawn. It is made on the second frame in a row with the same view,
 * so a moving camera, which never repeats, keeps drawing directly and pays nothing extra.
 */
interface WorldCache {
  /** The view the picture shows ("" for none yet). */
  key: string;
  geometry: TrackGeometry | null;
  picture: HTMLCanvasElement | null;
  /** The view of the previous frame. */
  lastKey: string;
  lastGeometry: TrackGeometry | null;
}

const worldCaches = new WeakMap<HTMLCanvasElement | OffscreenCanvas, WorldCache>();

function viewKey(theme: Theme, scene: Scene, base: DOMMatrix, width: number, height: number): string {
  const { cx, cy, scale } = scene.camera;
  return [theme.key, cx, cy, scale, scene.size.w, scene.size.h, base.a, base.b, base.c, base.d, base.e, base.f, width, height].join("|");
}

/** The world drawn into `target` (the canvas's own context or the cache's) under `base`. */
function paintWorld(target: CanvasRenderingContext2D, base: DOMMatrix, theme: Theme, scene: Scene, geometry: TrackGeometry): void {
  target.save();
  target.setTransform(base.multiply(new DOMMatrix(worldTransform(scene.camera, scene.size))));
  drawWorld(target, theme, geometry, trackArt(geometry), scene.camera, scene.size);
  target.restore();
}

function drawWorldCached(ctx: CanvasRenderingContext2D, base: DOMMatrix, theme: Theme, scene: Scene, geometry: TrackGeometry): void {
  const { width, height } = ctx.canvas;
  let cache = worldCaches.get(ctx.canvas);
  if (cache === undefined) {
    cache = { key: "", geometry: null, picture: null, lastKey: "", lastGeometry: null };
    worldCaches.set(ctx.canvas, cache);
  }
  const key = viewKey(theme, scene, base, width, height);
  const repeated = key === cache.lastKey && geometry === cache.lastGeometry;
  cache.lastKey = key;
  cache.lastGeometry = geometry;
  if (!(cache.key === key && cache.geometry === geometry && cache.picture !== null)) {
    if (!repeated || typeof document === "undefined") {
      paintWorld(ctx, base, theme, scene, geometry);
      return;
    }
    const picture = cache.picture ?? document.createElement("canvas");
    picture.width = width;
    picture.height = height;
    const pctx = picture.getContext("2d");
    if (pctx === null) {
      paintWorld(ctx, base, theme, scene, geometry);
      return;
    }
    paintWorld(pctx, base, theme, scene, geometry);
    cache.picture = picture;
    cache.key = key;
    cache.geometry = geometry;
  }
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(cache.picture, 0, 0);
  ctx.restore();
}

/** Draw the whole scene: the world, the boards, then the selected car's data and every car. */
export function drawScene(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const base = ctx.getTransform(); // CSS pixels → device pixels, set by the caller
  ctx.fillStyle = theme.grass;
  ctx.fillRect(0, 0, scene.size.w, scene.size.h);
  if (scene.geometry !== null) {
    drawWorldCached(ctx, base, theme, scene, scene.geometry);
    if (!scene.compact) drawBoards(ctx, theme, trackArt(scene.geometry), scene.camera, scene.size);
  }
  drawTrail(ctx, theme, scene);
  drawRays(ctx, theme, scene);
  drawGhostPaths(ctx, theme, scene);
  drawCars(ctx, theme, scene);
  drawRaceGhost(ctx, theme, scene);
  drawMoment(ctx, theme, scene);
  drawGhostCar(ctx, theme, scene);
}
