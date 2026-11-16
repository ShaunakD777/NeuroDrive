// The cars on the track (task E2, DESIGN_SPEC §7.2), lifted from the prototype's car and label
// layers. Two passes:
//
//   drawCars       in world metres, under the world transform `drawScene` set: every car as a
//                  small race car (a dart at overview zooms), worst rank first; then a race's
//                  ghost; then the selected car on top; then the look-back and what-if cars.
//   drawCarLabels  in CSS pixels, so they keep their size at any zoom: elite rings, the badges
//                  of stopped cars, the selection reticle and the name tags.
//
// Each car is drawn once in its own frame (u forward, v to the left, centred on (0, 0), for a
// 4.5 × 2.0 m car) and placed by `carTransform`: a translation to its position, a turn by its
// heading and a stretch to the scenario's car size.

import type { ScenarioMessage } from "../../api/generated";
import {
  BADGES,
  bodyColor,
  bodyOf,
  carTransform,
  DETAIL_SCALE,
  drawOrder,
  ghostTag,
  growFactor,
  isBraking,
  LABEL_SCALE,
  nameTag,
  showCarNumbers,
} from "./cars";
import { pill } from "./drawOverlays";
import { worldToScreen } from "./geometry";
import type { Scene, Theme } from "./scene";

const CAR_LENGTH_EXTRA = 2.0; // car length = wheelbase + this, metres (display only)
const TYRE = "#14161a";
const LIGHT = "#f3f4f1"; // helmets, the ghost's outline
const OUTLINE = "rgba(10, 12, 16, 0.7)";
const CASING = "rgba(8, 10, 14, 0.55)";
const BRAKE = "#ff4338";
const TAG_FONT = "600 11px 'IBM Plex Sans Condensed', system-ui, sans-serif";

/** The car's length and width in metres, from the scenario (4.5 × 2.0 with today's car). */
function carSize(scenario: ScenarioMessage): { length: number; width: number } {
  return { length: scenario.wheelbase + CAR_LENGTH_EXTRA, width: 2 * scenario.car_half_width };
}

interface CarLook {
  body: string;
  alpha: number;
  /** The full race car (true) or the dart. */
  detail: boolean;
  braking: boolean;
  helmet: string;
  outline?: string;
  dashed?: boolean;
  /** Headlights (running cars at night). */
  lights?: boolean;
}

/** The body outline, in sprite metres: a tapered nose, side pods, a narrow tail. */
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

/** The overview car: a dart filling the car's box, its point the front. */
function dartPath(ctx: CanvasRenderingContext2D): void {
  ctx.beginPath();
  ctx.moveTo(2.25, 0);
  ctx.lineTo(0.9, 0.95);
  ctx.lineTo(-2.25, 0.95);
  ctx.lineTo(-2.25, -0.95);
  ctx.lineTo(0.9, -0.95);
  ctx.closePath();
}

/** One car at a world pose, under the world transform. */
function drawCar(
  ctx: CanvasRenderingContext2D,
  theme: Theme,
  scene: Scene,
  pose: { x: number; y: number; psi: number },
  look: CarLook,
): void {
  if (scene.scenario === null) return;
  const { length, width } = carSize(scene.scenario);
  const scale = scene.camera.scale;
  const grow = growFactor(length, scale);
  const px = 1 / (scale * grow); // one CSS pixel, in sprite metres
  ctx.save();
  ctx.globalAlpha = look.alpha;

  if (look.detail && !look.dashed) {
    // The shadow: the car's box pushed towards the lower right (the light comes from the upper left).
    ctx.save();
    ctx.transform(...carTransform(pose.x + 0.3, pose.y - 0.36, pose.psi, length, width));
    ctx.beginPath();
    ctx.roundRect(-2.2, -0.98, 4.4, 1.96, 0.5);
    ctx.fillStyle = theme.shadow;
    ctx.fill();
    ctx.restore();
  }

  ctx.transform(...carTransform(pose.x, pose.y, pose.psi, length, width, grow));

  if (!look.detail) {
    dartPath(ctx);
    ctx.fillStyle = look.body;
    ctx.fill();
    ctx.lineWidth = px;
    ctx.strokeStyle = look.outline ?? "rgba(10, 12, 16, 0.85)";
    if (look.dashed) ctx.setLineDash([3 * px, 2.5 * px]);
    ctx.stroke();
    ctx.restore();
    return;
  }

  if (look.lights) {
    const beam = ctx.createLinearGradient(2, 0, 12, 0);
    beam.addColorStop(0, "rgba(255, 244, 214, 0.22)");
    beam.addColorStop(1, "rgba(255, 244, 214, 0)");
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
  ctx.fillStyle = TYRE;
  for (const wx of [1.32, -1.38]) {
    for (const wy of [-0.79, 0.79]) {
      ctx.beginPath();
      ctx.roundRect(wx - 0.44, wy - 0.21, 0.88, 0.42, 0.1);
      ctx.fill();
    }
  }
  // Rear and front wings: the body colour, darkened.
  for (const [x, y, w, h] of [
    [-2.24, -0.86, 0.36, 1.72],
    [1.8, -0.96, 0.36, 1.92],
  ]) {
    ctx.fillStyle = look.body;
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = "rgba(0, 0, 0, 0.34)";
    ctx.fillRect(x, y, w, h);
  }
  // The body and its outline.
  bodyPath(ctx);
  ctx.fillStyle = look.body;
  ctx.fill();
  ctx.lineWidth = Math.max(0.05, px);
  ctx.strokeStyle = look.outline ?? OUTLINE;
  if (look.dashed) ctx.setLineDash([0.35, 0.3]);
  ctx.stroke();
  ctx.setLineDash([]);
  // Engine cover shading and the light stripe on the nose.
  ctx.fillStyle = "rgba(0, 0, 0, 0.2)";
  ctx.beginPath();
  ctx.ellipse(-1.15, 0, 0.62, 0.24, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.55)";
  ctx.lineWidth = 0.11;
  ctx.beginPath();
  ctx.moveTo(2.05, 0);
  ctx.lineTo(0.55, 0);
  ctx.stroke();
  // Cockpit and helmet.
  ctx.fillStyle = TYRE;
  ctx.beginPath();
  ctx.ellipse(-0.1, 0, 0.62, 0.3, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = look.helmet;
  ctx.beginPath();
  ctx.arc(-0.22, 0, 0.2, 0, Math.PI * 2);
  ctx.fill();
  // Brake lights.
  if (look.braking) {
    ctx.fillStyle = BRAKE;
    ctx.fillRect(-2.3, -0.62, 0.1, 0.34);
    ctx.fillRect(-2.3, 0.28, 0.1, 0.34);
  }
  ctx.restore();
}

/** True when a world point is near enough the view for something `margin` pixels big to show. */
function onScreen(scene: Scene, x: number, y: number, margin: number): boolean {
  const [px, py] = worldToScreen(scene.camera, scene.size, x, y);
  return px > -margin && py > -margin && px < scene.size.w + margin && py < scene.size.h + margin;
}

/** Every car, in world metres (the caller has set the world transform). */
export function drawCars(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const { frame, scenario } = scene;
  if (frame === null || scenario === null) return;
  const info = scenario.population;
  const { x, y, psi, status, action } = frame.cars;
  const count = x.length;
  const detail = scene.camera.scale >= DETAIL_SCALE;
  const { length } = carSize(scenario);
  // A car's half length on screen plus its headlights, so a car at the edge is not cut off early.
  const margin = 3 * length * scene.camera.scale * growFactor(length, scene.camera.scale) + 20;
  const s = scene.selected;
  const ghost = info?.ghost_car ?? null;
  const ghostLook = (helmet: string): CarLook => ({
    body: theme.ghostBody,
    alpha: 0.75,
    detail,
    braking: false,
    helmet,
    outline: LIGHT,
    dashed: true,
  });
  const carLook = (i: number, selected: boolean): CarLook => {
    const running = status[i] === 0;
    return {
      body: bodyColor(theme, bodyOf(info, count, i, s)),
      alpha: running || selected ? 1 : 0.42,
      detail,
      braking: isBraking(status[i], action[i]),
      helmet: selected ? theme.select : LIGHT,
      lights: theme.night && running && detail,
    };
  };

  for (const i of drawOrder(info, count, s)) {
    if (!onScreen(scene, x[i], y[i], margin)) continue;
    drawCar(ctx, theme, scene, { x: x[i], y: y[i], psi: psi[i] }, carLook(i, false));
  }
  // The ghost: see-through and dashed, over the generation (it often sits on the elite copy).
  if (ghost !== null && ghost < count && ghost !== s) {
    drawCar(ctx, theme, scene, { x: x[ghost], y: y[ghost], psi: psi[ghost] }, ghostLook(LIGHT));
  }
  // The selected car last, so it is never hidden in a pack.
  if (s < count) {
    const pose = { x: x[s], y: y[s], psi: psi[s] };
    drawCar(ctx, theme, scene, pose, s === ghost ? ghostLook(theme.select) : carLook(s, true));
  }
  // Where the selected car was at the step being looked at.
  const m = scene.moment;
  if (m !== null) {
    drawCar(ctx, theme, scene, m, { body: theme.select, alpha: 0.5, detail, braking: false, helmet: "#ffffff", outline: "#ffffff" });
  }
  // Where the what-if action would take it.
  if (scene.ghost !== null) {
    const [gx, gy, gpsi] = scene.ghost.what_if_end;
    drawCar(
      ctx,
      theme,
      scene,
      { x: gx, y: gy, psi: gpsi },
      { body: theme.select, alpha: 0.55, detail, braking: false, helmet: "#ffffff", outline: "#ffffff", dashed: true },
    );
  }
}

/** Elite rings, status badges, the selection reticle and the name tags, in CSS pixels. */
export function drawCarLabels(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const { frame, scenario, camera, size } = scene;
  if (frame === null || scenario === null) return;
  const info = scenario.population;
  const { x, y, status } = frame.cars;
  const s = scene.selected;
  const ghost = info?.ghost_car ?? null;
  const showBadges = camera.scale >= LABEL_SCALE;
  const numbers = showCarNumbers(info, x.length, camera.scale, scene.compact);

  for (let i = 0; i < x.length; i++) {
    if (!onScreen(scene, x[i], y[i], 30)) continue;
    const [px, py] = worldToScreen(camera, size, x[i], y[i]);
    if (numbers && i !== s) {
      // The car's number below it, outlined in its livery (ADR-065): colour alone cannot tell
      // every pair of liveries apart.
      pill(ctx, `${i + 1}`, px, py + Math.max(13, 2.4 * camera.scale), {
        bg: theme.hudBg,
        fg: theme.hudText,
        align: "center",
        font: "600 10px 'IBM Plex Sans Condensed', system-ui, sans-serif",
        border: bodyColor(theme, bodyOf(info, x.length, i, s)),
        height: 14,
      });
    }
    if (info !== null && info.elite.includes(i)) {
      ctx.beginPath();
      ctx.arc(px, py, Math.max(9, 2.9 * camera.scale), 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(8, 10, 14, 0.5)";
      ctx.lineWidth = 3.5;
      ctx.stroke();
      ctx.strokeStyle = theme.elite;
      ctx.lineWidth = 1.75;
      ctx.stroke();
    }
    const badge = BADGES[status[i]];
    if (badge === undefined || i === ghost) continue;
    const colour = theme[badge.colour];
    if (!showBadges && i !== s) {
      // Zoomed out: a small dot in the badge's colour.
      ctx.beginPath();
      ctx.arc(px, py, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      continue;
    }
    const by = py - Math.max(13, 2.2 * camera.scale);
    ctx.beginPath();
    ctx.arc(px, by, 7.5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(15, 18, 24, 0.92)";
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
  }

  // The selection reticle: a ring broken into four arcs, in the selection colour over a dark casing.
  if (s < x.length) {
    const [px, py] = worldToScreen(camera, size, x[s], y[s]);
    const r = Math.max(16, 3.5 * camera.scale);
    const quarter = (Math.PI * r) / 2;
    ctx.lineCap = "butt";
    for (const [colour, lineWidth] of [
      [CASING, 5],
      [theme.select, 2.25],
    ] as const) {
      ctx.beginPath();
      ctx.arc(px, py, r, Math.PI / 8, Math.PI / 8 + Math.PI * 2);
      ctx.setLineDash([quarter * 0.5, quarter * 0.5]);
      ctx.strokeStyle = colour;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
    ctx.setLineDash([]);
    if (!scene.compact) {
      pill(ctx, nameTag(scenario.policy, info, s), px, py - r - 14, {
        bg: theme.hudBg,
        fg: "#ffffff",
        align: "center",
        font: TAG_FONT,
        border: theme.select,
      });
    }
  }
  if (scene.compact) return;
  if (info !== null && ghost !== null && ghost < x.length && ghost !== s) {
    const [px, py] = worldToScreen(camera, size, x[ghost], y[ghost]);
    pill(ctx, ghostTag(info), px + 12, py + 16, {
      bg: "rgba(15, 18, 24, 0.78)",
      fg: "#d9dde4",
      font: "600 10.5px 'IBM Plex Sans Condensed', system-ui, sans-serif",
    });
  }
  const m = scene.moment;
  if (m !== null) {
    const [px, py] = worldToScreen(camera, size, m.x, m.y);
    pill(ctx, `STEP ${m.step}`, px + 14, py - 16, { bg: theme.hudBg, fg: "#ffffff", font: TAG_FONT, border: theme.select });
  }
  if (scene.ghost !== null) {
    const [px, py] = worldToScreen(camera, size, scene.ghost.what_if_end[0], scene.ghost.what_if_end[1]);
    pill(ctx, `WHAT-IF · ${scene.ghost.seconds} s`, px + 14, py - 16, {
      bg: theme.hudBg,
      fg: "#ffffff",
      font: TAG_FONT,
      border: theme.select,
    });
  }
}
