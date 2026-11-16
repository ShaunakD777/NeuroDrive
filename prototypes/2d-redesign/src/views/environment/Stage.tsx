import { useEffect, useRef, useState } from "react";

import { castRay, detailAt, poseAt, rollout, whatIf } from "../../mock/mockServer";
import { carName, carRole, useApp } from "../../state/store";
import { type Camera, easeCamera, easeOut, fitCamera, lookAhead, panBy, type Size, wheelFactor, worldToScreen, zoomAt } from "./camera";
import { drawScene, readWorldTheme, type Scene } from "./scene";
import { trackArt } from "./trackArt";

const FOLLOW_ZOOM = 7.5; // pixels per metre
const COMPACT_ZOOM = 4.6;
const DRAG_THRESHOLD = 4;
const TRANSITION_MS = 450;

/** Rank band of a car, 0 = best fifth (production: views/environment/population.ts). */
export function rankBand(rank: number, size: number): number {
  return Math.min(4, Math.floor((5 * rank) / Math.max(1, size - 1)));
}

/**
 * The canvas: the track, the cars and everything drawn on them. It redraws every display frame
 * from the store, so React never re-renders it while the cars move. The mouse works as in
 * production (task 4.10): wheel zooms, drag pans, click selects a car or a trail point,
 * double-click follows the car again.
 */
export function Stage({ compact }: { compact: boolean }) {
  const boxRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<Size>({ w: 0, h: 0 });
  const [dragging, setDragging] = useState(false);

  const cameraRef = useRef<Camera>({ cx: 0, cy: 0, scale: 1 });
  const freeRef = useRef<Camera>({ cx: 0, cy: 0, scale: 1 });
  const zoomRef = useRef(compact ? COMPACT_ZOOM : FOLLOW_ZOOM);
  const transitionRef = useRef<{ from: Camera; start: number } | null>(null);
  const followKeyRef = useRef("");
  const fitScaleRef = useRef(1);
  const pressRef = useRef<{ x: number; y: number; start: Camera; dragging: boolean } | null>(null);
  const trailScreenRef = useRef<{ keys: number[]; screen: [number, number][] }>({ keys: [], screen: [] });
  // The compact stage in a drawer always follows; the full stage uses the shared camera mode.
  const localMode = useRef<"follow" | "fit" | "free">("follow");

  useEffect(() => {
    const box = boxRef.current;
    if (box === null) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ w: Math.floor(entry.contentRect.width), h: Math.floor(entry.contentRect.height) });
    });
    observer.observe(box);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(size.w * ratio));
    canvas.height = Math.max(1, Math.round(size.h * ratio));
  }, [size]);

  const mode = () => (compact ? localMode.current : useApp.getState().cameraMode);
  const setMode = (next: "follow" | "fit" | "free") => {
    if (compact) localMode.current = next;
    else useApp.getState().set({ cameraMode: next });
  };

  useEffect(() => {
    if (size.w === 0 || size.h === 0) return;
    let raf = 0;
    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      const state = useApp.getState();
      const data = state.data;
      if (!canvas || !ctx || data === null) return;
      const { race, track } = data;
      const info = race.scenario.population;
      const n = race.scenario.n_cars;
      const t = state.t;
      const cars = Array.from({ length: n }, (_, i) => poseAt(race, t, i));
      const selected = state.car;
      const record = data.cars.get(selected) ?? null;
      const looking = state.viewKey !== null;
      const shownT = state.viewKey ?? t;
      const shownPose = looking ? poseAt(race, shownT, selected) : cars[selected];
      const detail = record === null ? null : detailAt(record, shownT);

      // ── Camera ──
      const margins = compact
        ? { top: 14, right: 14, bottom: 14, left: 14 }
        : { top: 70, right: 28, bottom: 138, left: 28 };
      const fit = fitCamera([...track.left, ...track.right], size, margins);
      fitScaleRef.current = fit.scale;
      const current = mode();
      const [fx, fy] = lookAhead(shownPose.x, shownPose.y, shownPose.psi, looking ? 0 : shownPose.v, zoomRef.current, size);
      const target: Camera =
        current === "free" ? freeRef.current : current === "fit" ? fit : { cx: fx, cy: fy, scale: zoomRef.current };
      // A change of mode, car or moment starts a short eased move instead of a jump.
      const key = `${current}:${current === "follow" ? `${selected}:${state.viewKey}` : ""}`;
      if (key !== followKeyRef.current) {
        if (followKeyRef.current !== "" && current !== "free") transitionRef.current = { from: cameraRef.current, start: now };
        followKeyRef.current = key;
      }
      let camera = target;
      const moving = transitionRef.current;
      if (moving !== null) {
        const p = (now - moving.start) / TRANSITION_MS;
        if (p >= 1) transitionRef.current = null;
        else camera = easeCamera(moving.from, target, easeOut(p));
      }
      cameraRef.current = camera;

      // ── Trail of the selected car: the last 60 s, as production remembers ──
      const framesBack = Math.round(race.scenario.trail_seconds / (race.stepsPerFrame * race.scenario.dt));
      const lastKey = Math.floor(t);
      const firstKey = Math.max(0, lastKey - framesBack);
      const column = state.output === "steer" ? race.frames.steer : state.output === "throttle" ? race.frames.throttle : null;
      const keys: number[] = [];
      const points: number[][] = [];
      const values: number[] = [];
      if (state.showTrail) {
        for (let k = firstKey; k <= lastKey; k++) {
          keys.push(k);
          points.push([race.frames.x[k][selected], race.frames.y[k][selected]]);
          if (column) values.push(column[k][selected]);
        }
        points.push([cars[selected].x, cars[selected].y]);
      }

      // ── Rays: recorded sensor readings for featured cars, measured for the others ──
      const angles = race.scenario.ray_angles;
      const range = race.scenario.ray_range;
      const distances = detail?.rays ?? angles.map((a) => castRay(track, shownPose.x, shownPose.y, shownPose.psi + a, range));
      const m = state.output === null ? null : state.output === "steer" ? 0 : 1;
      const sensitivity = detail !== null && m !== null ? detail.explain.sensitivity[m].slice(0, 5) : null;
      const highlight: number[] = [];
      if (state.neuron !== null && state.neuron.layer === 0 && state.neuron.index < 5) highlight.push(state.neuron.index);
      if (state.connection !== null && state.connection.layer === 0 && state.connection.from < 5) highlight.push(state.connection.from);

      // ── What-if ghost: hold the real action and the what-if action for 2 s ──
      let ghost: Scene["ghost"] = null;
      if (!state.playing && !looking && record !== null && detail !== null && Object.keys(state.whatIf).length > 0) {
        const answer = whatIf(record, detail, state.whatIf);
        const real = rollout(cars[selected], answer.base, 2, race.scenario.dt);
        const tried = rollout(cars[selected], answer.tried, 2, race.scenario.dt);
        ghost = { real: real.path, whatIf: tried.path, end: tried.end };
      }

      const order = Array.from({ length: info.size }, (_, i) => i).sort((a, b) => info.ranks[b] - info.ranks[a]);
      const theme = readWorldTheme(canvas);
      const role = carRole(data, selected);
      drawScene(ctx, theme, {
        size,
        camera,
        track,
        art: trackArt(track),
        cars,
        colors: cars.map((_, i) => (i < info.size ? theme.rank[rankBand(info.ranks[i], info.size)] : "#c9ccd2")),
        order,
        elite: info.elite,
        ghostCar: info.ghost_car,
        selected,
        rays: state.showRays ? { x: shownPose.x, y: shownPose.y, psi: shownPose.psi, distances, angles, range, highlight, sensitivity } : null,
        trail: state.showTrail ? { points, values: column ? values : null } : null,
        moment: looking ? { x: shownPose.x, y: shownPose.y, psi: shownPose.psi, label: `STEP ${race.frames.step[state.viewKey ?? 0]}` } : null,
        ghost,
        selectedLabel:
          role === "ghost"
            ? carName(data, selected).toUpperCase()
            : `CAR ${selected + 1} · P${info.ranks[selected] + 1}${role === "child" ? "" : ` · ${role.toUpperCase()}`}`,
        ghostLabel: `GHOST · GEN ${info.ghost_unit}`,
        compact,
      });
      trailScreenRef.current = {
        keys,
        screen: keys.map((_, i) => worldToScreen(camera, size, points[i][0], points[i][1])),
      };
      canvas.dataset.cameraMode = current;
      canvas.dataset.cars = cars.map((c) => worldToScreen(camera, size, c.x, c.y).map(Math.round).join(":")).join(",");
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, compact]);

  // Wheel: zoom level while following, around the cursor otherwise (as production, task 4.10).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const factor = wheelFactor(event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY);
      if (mode() === "follow") {
        zoomRef.current = Math.min(40, Math.max(1, zoomRef.current * factor));
        return;
      }
      const rect = canvas.getBoundingClientRect();
      freeRef.current = zoomAt(cameraRef.current, size, event.clientX - rect.left, event.clientY - rect.top, factor, fitScaleRef.current / 2);
      setMode("free");
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size]);

  // The +/− buttons of the toolbar reach the stage through a window event.
  useEffect(() => {
    if (compact) return;
    const onZoom = (event: Event) => {
      const factor = (event as CustomEvent<number>).detail;
      if (mode() === "follow") {
        zoomRef.current = Math.min(40, Math.max(1, zoomRef.current * factor));
        return;
      }
      freeRef.current = zoomAt(cameraRef.current, size, size.w / 2, size.h / 2, factor, fitScaleRef.current / 2);
      setMode("free");
    };
    window.addEventListener("nd-zoom", onZoom);
    return () => window.removeEventListener("nd-zoom", onZoom);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size, compact]);

  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    pressRef.current = { x: event.clientX, y: event.clientY, start: cameraRef.current, dragging: false };
  };

  const onPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const press = pressRef.current;
    if (press === null) return;
    const dx = event.clientX - press.x;
    const dy = event.clientY - press.y;
    if (!press.dragging && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
    if (!press.dragging) {
      press.dragging = true;
      setDragging(true);
    }
    freeRef.current = panBy(press.start, dx, dy);
    transitionRef.current = null;
    if (mode() !== "free") setMode("free");
  };

  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const press = pressRef.current;
    pressRef.current = null;
    setDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (press === null || press.dragging) return;
    const state = useApp.getState();
    const data = state.data;
    if (data === null) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const px = press.x - rect.left;
    const py = press.y - rect.top;
    // A click selects the nearest car within 24 px …
    let best: number | null = null;
    let bestDistance = 24;
    for (let i = 0; i < data.race.scenario.n_cars; i++) {
      const pose = poseAt(data.race, state.t, i);
      const [sx, sy] = worldToScreen(cameraRef.current, size, pose.x, pose.y);
      const d = Math.hypot(sx - px, sy - py);
      if (d <= bestDistance) {
        bestDistance = d;
        best = i;
      }
    }
    if (best !== null && best !== state.car) {
      state.selectCar(best);
      return;
    }
    if (best !== null) {
      state.lookAt(null); // clicking the selected car: back to now
      return;
    }
    // … or, while paused, a point of the trail: look at that step.
    if (state.playing) return;
    const { keys, screen } = trailScreenRef.current;
    let key: number | null = null;
    let nearest = 12;
    screen.forEach(([sx, sy], i) => {
      const d = Math.hypot(sx - px, sy - py);
      if (d <= nearest) {
        nearest = d;
        key = keys[i];
      }
    });
    if (key !== null) state.lookAt(key);
  };

  return (
    <div className="stage-box" ref={boxRef}>
      <canvas
        ref={canvasRef}
        className="stage-canvas"
        data-testid={compact ? "stage-compact" : "stage"}
        data-dragging={dragging ? "true" : undefined}
        style={{ width: size.w, height: size.h }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={() => setMode("follow")}
        role="img"
        aria-label="Top-down view of the track and the cars"
      />
      <div className="stage-vignette" aria-hidden="true" />
    </div>
  );
}
