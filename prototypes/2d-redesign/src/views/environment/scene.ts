// Canvas 2D drawing of the world. Like production's draw.ts it only turns numbers into pixels,
// and reads every colour from CSS variables. What is new is the layering:
//
//   WORLD (drawn in metres, y up)      grass → verge → sand → asphalt → rubber line → kerbs →
//                                      edge lines → centre dashes → arrows → start line
//   DATA  (drawn in pixels)            decision trail → sensor fan and rays → what-if paths
//   CARS  (drawn in metres)            shadows → cars, worst rank first → ghost → look-back car
//   LABELS (drawn in pixels)           distance boards, rings, status badges, name tags
//
// Rule of the scene: decoration never sits above data. Everything a car or a ray can touch is
// inside the two white edge lines.

import type { CarPose } from "../../mock/mockServer";
import type { Track } from "../../mock/types";
import { type Camera, type Size, worldToScreen } from "./camera";
import type { TrackArt } from "./trackArt";

export interface WorldTheme {
  key: string;
  grass: string;
  grass2: string;
  verge: string;
  sand: string;
  asphalt: string;
  asphalt2: string;
  line: string;
  kerbA: string;
  kerbB: string;
  shadow: string;
  marking: string;
  board: string;
  boardInk: string;
  rank: string[];
  elite: string;
  rayClear: string;
  rayNear: string;
  rayDanger: string;
  rayNone: string;
  pos: string;
  neg: string;
  select: string;
  hudBg: string;
  hudText: string;
  ok: string;
  warn: string;
  bad: string;
  idle: string;
  night: boolean;
}

export function readWorldTheme(el: Element): WorldTheme {
  const style = getComputedStyle(el);
  const get = (name: string) => style.getPropertyValue(name).trim();
  const night = document.documentElement.dataset.theme === "dark";
  return {
    key: `${night}`,
    grass: get("--w-grass"),
    grass2: get("--w-grass-2"),
    verge: get("--w-verge"),
    sand: get("--w-sand"),
    asphalt: get("--w-asphalt"),
    asphalt2: get("--w-asphalt-2"),
    line: get("--w-line"),
    kerbA: get("--w-kerb-a"),
    kerbB: get("--w-kerb-b"),
    shadow: get("--w-shadow"),
    marking: get("--w-marking"),
    board: get("--w-board"),
    boardInk: get("--w-board-ink"),
    rank: [1, 2, 3, 4, 5].map((k) => get(`--car-rank-${k}`)),
    elite: get("--car-elite"),
    rayClear: get("--ray-clear"),
    rayNear: get("--ray-near"),
    rayDanger: get("--ray-danger"),
    rayNone: get("--ray-none"),
    pos: get("--track-pos"),
    neg: get("--track-neg"),
    select: get("--track-select"),
    hudBg: get("--hud-bg"),
    hudText: get("--hud-text"),
    ok: "#4ad17f",
    warn: "#f5b83d",
    bad: "#ff6b5e",
    idle: "#a3acba",
    night,
  };
}

export interface SceneRays {
  x: number;
  y: number;
  psi: number;
  distances: number[];
  angles: number[];
  range: number;
  /** Indices to emphasise (the selected input neuron's ray); empty for none. */
  highlight: number[];
  /** The selected output's sensitivity to each ray, or null for the distance colours. */
  sensitivity: number[] | null;
}

export interface Scene {
  size: Size;
  camera: Camera;
  track: Track;
  art: TrackArt;
  cars: CarPose[];
  /** Body colour of each car. */
  colors: string[];
  /** Draw order: worst first, so the best end up on top. */
  order: number[];
  elite: number[];
  ghostCar: number | null;
  selected: number;
  rays: SceneRays | null;
  trail: { points: number[][]; values: number[] | null } | null;
  moment: { x: number; y: number; psi: number; label: string } | null;
  ghost: { real: number[][]; whatIf: number[][]; end: [number, number, number] } | null;
  selectedLabel: string;
  ghostLabel: string;
  compact: boolean;
}

const CAR_LENGTH = 4.5; // wheelbase 2.5 m + 2 m, as production draws it
const DETAIL_SCALE = 4.2; // pixels per metre above which the full car sprite is drawn

// ── Textures: small seeded noise tiles, made once per theme ─────────────────────────────

interface Textures {
  asphalt: CanvasPattern | null;
  grass: CanvasPattern | null;
  stripes: CanvasPattern | null;
}

let textures: { key: string; value: Textures } | null = null;

function noiseTile(size: number, dots: number, seed: number, light: number, dark: number): HTMLCanvasElement {
  const tile = document.createElement("canvas");
  tile.width = size;
  tile.height = size;
  const g = tile.getContext("2d")!;
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

function texturesFor(ctx: CanvasRenderingContext2D, theme: WorldTheme): Textures {
  if (textures !== null && textures.key === theme.key) return textures.value;
  const asphalt = ctx.createPattern(noiseTile(160, 2600, 7, theme.night ? 0.05 : 0.075, 0.16), "repeat");
  asphalt?.setTransform(new DOMMatrix().scale(9 / 160)); // one tile = 9 m
  const grass = ctx.createPattern(noiseTile(160, 2200, 23, theme.night ? 0.03 : 0.06, 0.1), "repeat");
  grass?.setTransform(new DOMMatrix().scale(14 / 160));
  // Mowing stripes: bands 9 m wide at 28°.
  const band = document.createElement("canvas");
  band.width = 64;
  band.height = 4;
  const g = band.getContext("2d")!;
  g.fillStyle = theme.grass2;
  g.fillRect(0, 0, 32, 4);
  const stripes = ctx.createPattern(band, "repeat");
  stripes?.setTransform(new DOMMatrix().rotate(28).scale(18 / 64));
  textures = { key: theme.key, value: { asphalt, grass, stripes } };
  return textures.value;
}

// ── Helpers ─────────────────────────────────────────────────────────────────────────────

function ratioOf(ctx: CanvasRenderingContext2D, size: Size): number {
  return ctx.canvas.width / Math.max(1, size.w);
}

/** Draw in world metres (y up) from here on. */
function worldSpace(ctx: CanvasRenderingContext2D, scene: Scene): void {
  const ratio = ratioOf(ctx, scene.size);
  const k = scene.camera.scale * ratio;
  ctx.setTransform(k, 0, 0, -k, ratio * (scene.size.w / 2) - scene.camera.cx * k, ratio * (scene.size.h / 2) + scene.camera.cy * k);
}

/** Draw in CSS pixels from here on. */
function screenSpace(ctx: CanvasRenderingContext2D, scene: Scene): void {
  const ratio = ratioOf(ctx, scene.size);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function loop(ctx: CanvasRenderingContext2D, points: number[][]): void {
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.closePath();
}

function pill(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  options: { bg: string; fg: string; align?: "left" | "center"; font?: string; border?: string },
): void {
  ctx.font = options.font ?? "500 11px 'IBM Plex Mono', ui-monospace, monospace";
  const w = Math.ceil(ctx.measureText(text).width) + 10;
  const h = 17;
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

// ── The world ───────────────────────────────────────────────────────────────────────────

function drawWorld(ctx: CanvasRenderingContext2D, theme: WorldTheme, scene: Scene): void {
  const { track, art, camera, size } = scene;
  const tex = texturesFor(ctx, theme);
  const px = 1 / camera.scale; // one screen pixel in metres

  // 1. Grass: base, mowing stripes, fine noise.
  const x0 = camera.cx - size.w / 2 / camera.scale;
  const y0 = camera.cy - size.h / 2 / camera.scale;
  const w = size.w / camera.scale;
  const h = size.h / camera.scale;
  ctx.fillStyle = theme.grass;
  ctx.fillRect(x0, y0, w, h);
  if (tex.stripes) {
    ctx.fillStyle = tex.stripes;
    ctx.fillRect(x0, y0, w, h);
  }
  if (tex.grass && camera.scale > 1.2) {
    ctx.fillStyle = tex.grass;
    ctx.fillRect(x0, y0, w, h);
  }

  // 2. The mown verge: a lighter band following the road.
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
  if (tex.grass && camera.scale > 1.2) {
    ctx.fillStyle = tex.grass;
    for (const polygon of art.sand) {
      ctx.beginPath();
      loop(ctx, polygon);
      ctx.fill();
    }
  }

  // 4. A dark lip under the road edge, then the asphalt itself (the ring between the edges).
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
  if (tex.asphalt && camera.scale > 1.6) {
    ctx.fillStyle = tex.asphalt;
    ctx.fill("evenodd");
  }

  // 5. The rubbered-in line down the middle: three soft strokes.
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

  // 6. Kerbs: alternating blocks outside the edge.
  for (const tone of [0, 1]) {
    ctx.beginPath();
    for (const quad of art.kerbs) {
      if (quad.tone !== tone) continue;
      loop(ctx, quad.points);
    }
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
  if (camera.scale > 1.4) {
    ctx.beginPath();
    loop(ctx, track.centerline);
    ctx.setLineDash([2.6, 5.4]);
    ctx.strokeStyle = theme.marking;
    ctx.lineWidth = Math.max(0.16, 1 * px);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // 9. Direction arrows.
  if (camera.scale > 3) {
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

  // 10. The start/finish line: two rows of checks across the road.
  const { left, right, tangent } = art.start;
  const across = [left[0] - right[0], left[1] - right[1]];
  const columns = 12;
  const cell = Math.hypot(across[0], across[1]) / columns;
  const ax = across[0] / columns;
  const ay = across[1] / columns;
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
      ctx.fillStyle = (row + col) % 2 === 0 ? theme.line : "#16181c";
      ctx.fill();
    }
  }
}

// ── Cars ────────────────────────────────────────────────────────────────────────────────

interface CarLook {
  body: string;
  alpha: number;
  detail: boolean;
  /** Pixels per metre, for hairlines. */
  scale: number;
  braking: boolean;
  helmet: string;
  outline?: string;
  dashed?: boolean;
  lights?: boolean;
}

function bodyPath(ctx: CanvasRenderingContext2D): void {
  ctx.beginPath();
  ctx.moveTo(2.2, 0);
  ctx.bezierCurveTo(1.7, 0.2, 1.1, 0.26, 0.7, 0.34);
  ctx.lineTo(0.45, 0.72);
  ctx.lineTo(-0.9, 0.72);
  ctx.bezierCurveTo(-1.4, 0.62, -1.7, 0.46, -1.95, 0.36);
  ctx.lineTo(-1.95, -0.36);
  ctx.bezierCurveTo(-1.7, -0.46, -1.4, -0.62, -0.9, -0.72);
  ctx.lineTo(0.45, -0.72);
  ctx.lineTo(0.7, -0.34);
  ctx.bezierCurveTo(1.1, -0.26, 1.7, -0.2, 2.2, 0);
  ctx.closePath();
}

/** The small-scale car: a dart whose point is the front. */
function dartPath(ctx: CanvasRenderingContext2D): void {
  ctx.beginPath();
  ctx.moveTo(2.25, 0);
  ctx.lineTo(0.9, 0.95);
  ctx.lineTo(-2.25, 0.95);
  ctx.lineTo(-2.25, -0.95);
  ctx.lineTo(0.9, -0.95);
  ctx.closePath();
}

function drawCar(ctx: CanvasRenderingContext2D, pose: { x: number; y: number; psi: number }, look: CarLook, theme: WorldTheme): void {
  // At overview scales a true-size car is a few pixels: draw it larger (never below ~10 px long).
  const grow = Math.max(1, 10 / (CAR_LENGTH * look.scale));
  const px = 1 / (look.scale * grow);

  ctx.save();
  ctx.globalAlpha = look.alpha;

  if (look.detail && !look.dashed) {
    // Shadow: the same shape, pushed towards the lower right (the sun is upper left).
    ctx.save();
    ctx.translate(pose.x + 0.3, pose.y - 0.36);
    ctx.rotate(pose.psi);
    ctx.beginPath();
    ctx.roundRect(-2.2, -0.98, 4.4, 1.96, 0.5);
    ctx.fillStyle = theme.shadow;
    ctx.fill();
    ctx.restore();
  }

  ctx.translate(pose.x, pose.y);
  ctx.rotate(pose.psi);
  ctx.scale(grow, grow);

  if (!look.detail) {
    dartPath(ctx);
    ctx.fillStyle = look.body;
    ctx.fill();
    ctx.lineWidth = 1 * px;
    ctx.strokeStyle = look.outline ?? "rgba(10,12,16,0.85)";
    if (look.dashed) ctx.setLineDash([3 * px, 2.5 * px]);
    ctx.stroke();
    ctx.restore();
    return;
  }

  if (look.lights) {
    const beam = ctx.createLinearGradient(2, 0, 12, 0);
    beam.addColorStop(0, "rgba(255,244,214,0.22)");
    beam.addColorStop(1, "rgba(255,244,214,0)");
    ctx.beginPath();
    ctx.moveTo(2, 0.5);
    ctx.lineTo(12, 3.4);
    ctx.lineTo(12, -3.4);
    ctx.lineTo(2, -0.5);
    ctx.closePath();
    ctx.fillStyle = beam;
    ctx.fill();
  }

  // Tyres.
  ctx.fillStyle = "#14161a";
  for (const wx of [1.32, -1.38]) {
    for (const wy of [-0.79, 0.79]) {
      ctx.beginPath();
      ctx.roundRect(wx - 0.44, wy - 0.21, 0.88, 0.42, 0.1);
      ctx.fill();
    }
  }
  // Wings: body colour, darkened.
  for (const [x, y, w, h] of [
    [-2.24, -0.86, 0.36, 1.72],
    [1.8, -0.96, 0.36, 1.92],
  ]) {
    ctx.fillStyle = look.body;
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = "rgba(0,0,0,0.34)";
    ctx.fillRect(x, y, w, h);
  }
  // Body.
  bodyPath(ctx);
  ctx.fillStyle = look.body;
  ctx.fill();
  ctx.lineWidth = Math.max(0.05, 1 * px);
  ctx.strokeStyle = look.outline ?? "rgba(10,12,16,0.7)";
  if (look.dashed) ctx.setLineDash([0.35, 0.3]);
  ctx.stroke();
  ctx.setLineDash([]);
  // Engine cover shading and the nose stripe.
  ctx.fillStyle = "rgba(0,0,0,0.2)";
  ctx.beginPath();
  ctx.ellipse(-1.15, 0, 0.62, 0.24, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 0.11;
  ctx.beginPath();
  ctx.moveTo(2.05, 0);
  ctx.lineTo(0.55, 0);
  ctx.stroke();
  // Cockpit and helmet.
  ctx.fillStyle = "#14161a";
  ctx.beginPath();
  ctx.ellipse(-0.1, 0, 0.62, 0.3, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = look.helmet;
  ctx.beginPath();
  ctx.arc(-0.22, 0, 0.2, 0, Math.PI * 2);
  ctx.fill();
  // Brake lights.
  if (look.braking) {
    ctx.fillStyle = "#ff4338";
    ctx.fillRect(-2.3, -0.62, 0.1, 0.34);
    ctx.fillRect(-2.3, 0.28, 0.1, 0.34);
  }
  ctx.restore();
}

function drawCars(ctx: CanvasRenderingContext2D, theme: WorldTheme, scene: Scene): void {
  const detail = scene.camera.scale >= DETAIL_SCALE;
  for (const i of scene.order) {
    if (i === scene.ghostCar) continue;
    const car = scene.cars[i];
    const running = car.status === 0;
    drawCar(
      ctx,
      car,
      {
        body: scene.colors[i],
        alpha: running ? 1 : 0.42,
        detail,
        scale: scene.camera.scale,
        braking: running && car.throttle < -0.08,
        helmet: i === scene.selected ? theme.select : "#f3f4f1",
        lights: theme.night && running && detail,
      },
      theme,
    );
  }
  // The ghost: see-through and dashed, over the cars (it often sits on last generation's elite copy).
  const g = scene.ghostCar;
  if (g !== null && g < scene.cars.length) {
    drawCar(
      ctx,
      scene.cars[g],
      { body: "rgba(240,242,246,0.5)", alpha: 0.75, detail, scale: scene.camera.scale, braking: false, helmet: "#f3f4f1", outline: "#f3f4f1", dashed: true },
      theme,
    );
  }
  // The selected car again, on top of everything, so it is never hidden in a pack.
  const s = scene.selected;
  if (s !== g && s < scene.cars.length) {
    const car = scene.cars[s];
    drawCar(
      ctx,
      car,
      {
        body: scene.colors[s],
        alpha: 1,
        detail,
        scale: scene.camera.scale,
        braking: car.status === 0 && car.throttle < -0.08,
        helmet: theme.select,
        lights: false,
      },
      theme,
    );
  }
  if (scene.moment !== null) {
    drawCar(
      ctx,
      scene.moment,
      { body: theme.select, alpha: 0.5, detail, scale: scene.camera.scale, braking: false, helmet: "#ffffff", outline: "#ffffff" },
      theme,
    );
  }
  if (scene.ghost !== null) {
    const [x, y, psi] = scene.ghost.end;
    drawCar(
      ctx,
      { x, y, psi },
      { body: theme.select, alpha: 0.55, detail, scale: scene.camera.scale, braking: false, helmet: "#ffffff", outline: "#ffffff", dashed: true },
      theme,
    );
  }
}

// ── Data layers (pixels) ────────────────────────────────────────────────────────────────

function drawTrail(ctx: CanvasRenderingContext2D, theme: WorldTheme, scene: Scene): void {
  const trail = scene.trail;
  if (trail === null || trail.points.length < 2) return;
  const screen = trail.points.map(([x, y]) => worldToScreen(scene.camera, scene.size, x, y));
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // A dark casing first, so the colours read on asphalt, kerb and grass alike.
  ctx.beginPath();
  screen.forEach(([px, py], k) => (k === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
  ctx.strokeStyle = "rgba(8,10,14,0.42)";
  ctx.lineWidth = trail.values === null ? 4.5 : 7.5;
  ctx.stroke();
  if (trail.values === null) {
    // No output chosen: the plain trail wears the selection colour (it is the selected car's).
    ctx.strokeStyle = theme.select;
    ctx.lineWidth = 2.25;
    ctx.stroke();
    return;
  }
  for (let k = 0; k + 1 < screen.length; k++) {
    const v = trail.values[k] ?? 0;
    const size = Math.min(1, Math.abs(v));
    ctx.globalAlpha = 0.45 + 0.55 * size;
    ctx.lineWidth = 1.5 + 4 * size;
    ctx.strokeStyle = v >= 0 ? theme.pos : theme.neg;
    ctx.beginPath();
    ctx.moveTo(screen[k][0], screen[k][1]);
    ctx.lineTo(screen[k + 1][0], screen[k + 1][1]);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawRays(ctx: CanvasRenderingContext2D, theme: WorldTheme, scene: Scene): void {
  const rays = scene.rays;
  if (rays === null) return;
  const [sx, sy] = worldToScreen(scene.camera, scene.size, rays.x, rays.y);
  const ends = rays.distances.map((d, k) => {
    const angle = rays.psi + rays.angles[k];
    return worldToScreen(scene.camera, scene.size, rays.x + Math.cos(angle) * d, rays.y + Math.sin(angle) * d);
  });

  // The field of view: a faint fan through the ray ends.
  ctx.beginPath();
  ctx.moveTo(sx, sy);
  ends.forEach(([ex, ey]) => ctx.lineTo(ex, ey));
  ctx.closePath();
  ctx.fillStyle = "rgba(255,255,255,0.055)";
  ctx.fill();

  const strongest = rays.sensitivity === null ? 1 : Math.max(1e-6, ...rays.sensitivity.map(Math.abs));
  ctx.lineCap = "round";
  rays.distances.forEach((distance, k) => {
    const [ex, ey] = ends[k];
    const hit = distance < rays.range - 1e-6;
    const emphasised = rays.highlight.includes(k);
    const faded = rays.highlight.length > 0 && !emphasised;
    const s = rays.sensitivity?.[k];
    const fraction = distance / rays.range;
    let colour: string;
    let width: number;
    if (s !== undefined) {
      colour = s < 0 ? theme.neg : theme.pos;
      width = 1.2 + 5 * (Math.abs(s) / strongest);
    } else {
      colour = !hit ? theme.rayNone : fraction < 0.2 ? theme.rayDanger : fraction < 0.5 ? theme.rayNear : theme.rayClear;
      width = emphasised ? 4.5 : hit ? 2 : 1.2;
    }
    ctx.globalAlpha = faded ? 0.28 : 1;
    // Dark casing, then the ray, fading in from the car so the car itself stays readable.
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = "rgba(8,10,14,0.35)";
    ctx.lineWidth = width + 2.5;
    ctx.stroke();
    const gradient = ctx.createLinearGradient(sx, sy, ex, ey);
    gradient.addColorStop(0, "rgba(255,255,255,0)");
    gradient.addColorStop(Math.min(0.35, 26 / Math.max(26, Math.hypot(ex - sx, ey - sy))), colour);
    gradient.addColorStop(1, colour);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = width;
    if (!hit && s === undefined) ctx.setLineDash([5, 6]);
    ctx.stroke();
    ctx.setLineDash([]);
    if (hit) {
      ctx.beginPath();
      ctx.arc(ex, ey, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(8,10,14,0.75)";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(ex, ey, 3, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
    }
    if (!scene.compact || emphasised) {
      const label = s !== undefined ? `s ${s >= 0 ? "+" : "−"}${Math.abs(s).toFixed(2)}` : hit ? `${distance.toFixed(1)} m` : `≥${rays.range.toFixed(0)} m`;
      // Push the label outward along the ray, so labels of neighbouring rays do not stack.
      const len = Math.hypot(ex - sx, ey - sy) || 1;
      const ux = (ex - sx) / len;
      const uy = (ey - sy) / len;
      ctx.font = "500 11px 'IBM Plex Mono', ui-monospace, monospace";
      const w = ctx.measureText(label).width + 10;
      const lx = ex + ux * 12 + (ux < 0 ? -w : 0) * (Math.abs(ux) > 0.35 ? 1 : 0.5);
      const ly = ey + uy * 13;
      pill(ctx, label, lx, ly, { bg: theme.hudBg, fg: hit || s !== undefined ? theme.hudText : "#a3acba" });
    }
    ctx.globalAlpha = 1;
  });
}

function drawGhostPaths(ctx: CanvasRenderingContext2D, theme: WorldTheme, scene: Scene): void {
  const ghost = scene.ghost;
  if (ghost === null) return;
  const line = (points: number[][]) => {
    ctx.beginPath();
    points.forEach(([x, y], k) => {
      const [px, py] = worldToScreen(scene.camera, scene.size, x, y);
      if (k === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
  };
  ctx.lineCap = "round";
  line(ghost.real);
  ctx.strokeStyle = "rgba(8,10,14,0.4)";
  ctx.lineWidth = 5;
  ctx.stroke();
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.setLineDash([]);
  line(ghost.whatIf);
  ctx.strokeStyle = "rgba(8,10,14,0.4)";
  ctx.lineWidth = 6;
  ctx.stroke();
  ctx.strokeStyle = theme.select;
  ctx.lineWidth = 3;
  ctx.stroke();
}

// ── Labels (pixels) ─────────────────────────────────────────────────────────────────────

const BADGE: Record<number, { glyph: string; colour: keyof WorldTheme }> = {
  1: { glyph: "×", colour: "bad" },
  2: { glyph: "✓", colour: "ok" },
  3: { glyph: "Ⅱ", colour: "warn" },
  4: { glyph: "◷", colour: "idle" },
};

function drawLabels(ctx: CanvasRenderingContext2D, theme: WorldTheme, scene: Scene): void {
  const { camera, size } = scene;

  // Distance boards: where 100 m, 200 m, … of a lap are (Progress in the telemetry is in metres).
  if (!scene.compact && camera.scale > 1.1) {
    ctx.font = "600 10px 'IBM Plex Sans Condensed', system-ui, sans-serif";
    for (const board of scene.art.boards) {
      const [px, py] = worldToScreen(camera, size, board.x, board.y);
      if (px < -40 || py < -20 || px > size.w + 40 || py > size.h + 20) continue;
      const text = `${board.metres}`;
      const w = ctx.measureText(text).width + 8;
      ctx.beginPath();
      ctx.roundRect(px - w / 2, py - 7, w, 14, 2);
      ctx.fillStyle = theme.board;
      ctx.fill();
      ctx.fillStyle = theme.boardInk;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, px, py + 0.5);
    }
    ctx.textAlign = "left";
  }

  // Elite rings (gold) and status badges of stopped cars.
  const showBadges = camera.scale >= 3.2;
  scene.cars.forEach((car, i) => {
    const [px, py] = worldToScreen(camera, size, car.x, car.y);
    if (px < -30 || py < -30 || px > size.w + 30 || py > size.h + 30) return;
    if (scene.elite.includes(i)) {
      ctx.beginPath();
      ctx.arc(px, py, Math.max(9, 2.9 * camera.scale), 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(8,10,14,0.5)";
      ctx.lineWidth = 3.5;
      ctx.stroke();
      ctx.strokeStyle = theme.elite;
      ctx.lineWidth = 1.75;
      ctx.stroke();
    }
    const badge = BADGE[car.status];
    if (badge === undefined || i === scene.ghostCar) return;
    const colour = theme[badge.colour] as string;
    if (!showBadges && i !== scene.selected) {
      ctx.beginPath();
      ctx.arc(px, py, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      return;
    }
    const by = py - Math.max(13, 2.2 * camera.scale);
    ctx.beginPath();
    ctx.arc(px, by, 7.5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(15,18,24,0.92)";
    ctx.fill();
    ctx.strokeStyle = colour;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = colour;
    ctx.font = "700 10px 'IBM Plex Sans', system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(badge.glyph, px, by + 0.5);
    ctx.textAlign = "left";
  });

  // The selection ring: a reticle in the selection colour, cased in dark.
  const s = scene.selected;
  if (s < scene.cars.length) {
    const car = scene.cars[s];
    const [px, py] = worldToScreen(camera, size, car.x, car.y);
    const r = Math.max(16, 3.5 * camera.scale);
    const quarter = (Math.PI * r) / 2;
    for (const [colour, width] of [
      ["rgba(8,10,14,0.55)", 5],
      [theme.select, 2.25],
    ] as const) {
      ctx.beginPath();
      ctx.arc(px, py, r, Math.PI / 8, Math.PI / 8 + Math.PI * 2);
      ctx.setLineDash([quarter * 0.5, quarter * 0.5]);
      ctx.strokeStyle = colour;
      ctx.lineWidth = width;
      ctx.lineCap = "butt";
      ctx.stroke();
    }
    ctx.setLineDash([]);
    if (!scene.compact) {
      pill(ctx, scene.selectedLabel, px, py - r - 14, {
        bg: theme.hudBg,
        fg: "#ffffff",
        align: "center",
        font: "600 11px 'IBM Plex Sans Condensed', system-ui, sans-serif",
        border: theme.select,
      });
    }
  }

  const g = scene.ghostCar;
  if (g !== null && g < scene.cars.length && !scene.compact && g !== s) {
    const [px, py] = worldToScreen(camera, size, scene.cars[g].x, scene.cars[g].y);
    pill(ctx, scene.ghostLabel, px + 12, py + 16, {
      bg: "rgba(15,18,24,0.78)",
      fg: "#d9dde4",
      font: "600 10.5px 'IBM Plex Sans Condensed', system-ui, sans-serif",
    });
  }
  if (scene.moment !== null && !scene.compact) {
    const [px, py] = worldToScreen(camera, size, scene.moment.x, scene.moment.y);
    pill(ctx, scene.moment.label, px + 14, py - 16, {
      bg: theme.hudBg,
      fg: "#ffffff",
      font: "600 11px 'IBM Plex Sans Condensed', system-ui, sans-serif",
      border: theme.select,
    });
  }
  if (scene.ghost !== null && !scene.compact) {
    const [px, py] = worldToScreen(camera, size, scene.ghost.end[0], scene.ghost.end[1]);
    pill(ctx, "WHAT-IF · 2 s", px + 14, py - 16, {
      bg: theme.hudBg,
      fg: "#ffffff",
      font: "600 11px 'IBM Plex Sans Condensed', system-ui, sans-serif",
      border: theme.select,
    });
  }
}

export function drawScene(ctx: CanvasRenderingContext2D, theme: WorldTheme, scene: Scene): void {
  worldSpace(ctx, scene);
  drawWorld(ctx, theme, scene);
  screenSpace(ctx, scene);
  drawTrail(ctx, theme, scene);
  drawRays(ctx, theme, scene);
  drawGhostPaths(ctx, theme, scene);
  worldSpace(ctx, scene);
  drawCars(ctx, theme, scene);
  screenSpace(ctx, scene);
  drawLabels(ctx, theme, scene);
}
