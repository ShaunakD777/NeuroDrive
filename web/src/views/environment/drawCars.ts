// The cars on the track, in CSS pixels: every car, a raced GA generation, its ghost, the look-back
// car and the what-if car. Moved unchanged from the pre-redesign `draw.ts` in task E1 (only their
// colours now come from light-on-dark tokens, so they read on the dark asphalt); task E2 restyles
// them.

import { carCorners, worldToScreen } from "./geometry";
import { carRole } from "./population";
import type { Scene, Theme } from "./scene";

const CAR_LENGTH_EXTRA = 2.0; // car length = wheelbase + this, metres (display only)

export function drawCars(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const { frame, scenario } = scene;
  if (frame === null || scenario === null) return;
  if (scenario.population) return drawRace(ctx, theme, scene);
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

/**
 * A raced GA generation (task 7.6): every network's car filled with its rank band (the GA's own
 * ranking, best fifth strongest), elites ringed, stopped cars faded, the selected car outlined
 * heavily (its fill carries the rank). Worse cars are drawn first so the best end up on top. The
 * ghost is drawn later, over the cars (`drawRaceGhost`).
 */
function drawRace(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const { frame, scenario } = scene;
  const info = scenario?.population;
  if (frame === null || !info) return;
  const { x, y, psi, alive } = frame.cars;
  const order = Array.from({ length: info.size }, (_, i) => i).sort((a, b) => info.ranks[b] - info.ranks[a]);
  for (const i of order) {
    const role = carRole(info, i);
    if (role === null || i >= x.length) continue;
    carPath(ctx, scene, x[i], y[i], psi[i]);
    ctx.globalAlpha = alive[i] ? 1 : 0.35;
    ctx.fillStyle = theme.rank[role.band ?? 0];
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.text;
    ctx.stroke();
    if (role.elite) {
      const [px, py] = worldToScreen(scene.camera, scene.size, x[i], y[i]);
      const r = Math.max(6, ((scenario.wheelbase + CAR_LENGTH_EXTRA) / 2) * scene.camera.scale + 3);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  const s = scene.selected;
  if (s < info.size && s < x.length) {
    carPath(ctx, scene, x[s], y[s], psi[s]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = theme.accent;
    ctx.stroke();
  }
}

/** The ghost of a race: see-through, dashed, labelled, over the cars (it may sit on car 1). */
export function drawRaceGhost(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const { frame, scenario } = scene;
  const info = scenario?.population;
  const g = info?.ghost_car;
  if (frame === null || !info || g === null || g === undefined || g >= frame.cars.x.length) return;
  const { x, y, psi } = frame.cars;
  carPath(ctx, scene, x[g], y[g], psi[g]);
  ctx.globalAlpha = 0.3;
  ctx.fillStyle = theme.edge;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = scene.selected === g ? 3 : 1.5;
  ctx.strokeStyle = scene.selected === g ? theme.accent : theme.text;
  ctx.stroke();
  ctx.setLineDash([]);
  if (!scene.compact) {
    const [px, py] = worldToScreen(scene.camera, scene.size, x[g], y[g]);
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.textBaseline = "top";
    ctx.fillStyle = theme.text;
    ctx.fillText(`gen ${info.ghost_unit} best`, px + 8, py + 8);
  }
}

/** One car outline at a world pose, as a closed screen-space path. */
function carPath(ctx: CanvasRenderingContext2D, scene: Scene, x: number, y: number, psi: number): void {
  const { scenario } = scene;
  if (scenario === null) return;
  const length = scenario.wheelbase + CAR_LENGTH_EXTRA;
  const width = 2 * scenario.car_half_width;
  const corners = carCorners(x, y, psi, length, width).map(([wx, wy]) =>
    worldToScreen(scene.camera, scene.size, wx, wy),
  );
  ctx.beginPath();
  corners.forEach(([px, py], k) => (k === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
  ctx.closePath();
}

/** The see-through ghost car at the end of the what-if path (over the cars). */
export function drawGhostCar(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const ghost = scene.ghost;
  if (ghost === null) return;
  const [x, y, psi] = ghost.what_if_end;
  ctx.globalAlpha = 0.45;
  carPath(ctx, scene, x, y, psi);
  ctx.fillStyle = theme.accent;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = theme.text;
  ctx.stroke();
  ctx.setLineDash([]);
  if (!scene.compact) {
    const [px, py] = worldToScreen(scene.camera, scene.size, x, y);
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.textBaseline = "bottom";
    ctx.fillStyle = theme.accent;
    ctx.fillText("what-if", px + 8, py - 8);
  }
}

/** The outlined car where the selected car was at the step being looked at. */
export function drawMoment(ctx: CanvasRenderingContext2D, theme: Theme, scene: Scene): void {
  const m = scene.moment;
  if (m === null) return;
  carPath(ctx, scene, m.x, m.y, m.psi);
  ctx.globalAlpha = 0.3;
  ctx.fillStyle = theme.accent;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = theme.accent;
  ctx.stroke();
  if (!scene.compact) {
    const [px, py] = worldToScreen(scene.camera, scene.size, m.x, m.y);
    ctx.font = "600 11px system-ui, sans-serif";
    ctx.textBaseline = "bottom";
    ctx.fillStyle = theme.accent;
    ctx.fillText(`step ${m.step}`, px + 8, py - 8);
  }
}
