import { useEffect, useRef, useState } from "react";

import { getWorld } from "../../source/world";
import { useApp } from "../../state/store";
import { type Camera, easeCamera, easeOut, fitCamera, lookAhead, panBy, type Size, wheelFactor, worldToScreen, zoomAt } from "./camera";
import { bodyColor, bodyOf, drawOrder, nameTag } from "./cars";
import { drawScene, readWorldTheme } from "./scene";
import { trackArt } from "./trackArt";

const FOLLOW_ZOOM = 7.5; // pixels per metre
const COMPACT_ZOOM = 4.6;
const DRAG_THRESHOLD = 4;
const TRANSITION_MS = 450;

/** No easing for people who asked their system for less motion (DESIGN_SPEC §11). */
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * The canvas: the track, the cars and everything drawn on them. It redraws every display frame
 * from the world snapshot (src/source/world.ts), so React never re-renders it while the cars
 * move. The mouse works as in production (task 4.10): wheel zooms, drag pans, click selects a car
 * or a trail point, double-click follows the car again.
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
  const trailScreenRef = useRef<{ steps: number[]; screen: [number, number][] }>({ steps: [], screen: [] });
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
      const world = getWorld();
      if (!canvas || !ctx || world === null) return;
      const state = useApp.getState();
      const { track, scenario, cars, selected } = world;
      const info = scenario.population;
      const looking = world.lookStep !== null;
      const carNow = cars[selected] ?? cars[0];
      const shownPose = world.shown?.pose ?? carNow;

      // ── Camera ──
      const margins = compact
        ? { top: 14, right: 14, bottom: 14, left: 14 }
        : { top: 70, right: 28, bottom: 164, left: 28 }; // clear of the telemetry strip and the legend above it
      const fit = fitCamera([...track.left, ...track.right], size, margins);
      fitScaleRef.current = fit.scale;
      const current = mode();
      const [fx, fy] = lookAhead(shownPose.x, shownPose.y, shownPose.psi, looking ? 0 : shownPose.v, zoomRef.current, size);
      const target: Camera =
        current === "free" ? freeRef.current : current === "fit" ? fit : { cx: fx, cy: fy, scale: zoomRef.current };
      // A change of mode, car, moment or track starts a short eased move instead of a jump.
      const key = `${track.name}:${current}:${current === "follow" ? `${selected}:${world.lookStep}` : ""}`;
      if (key !== followKeyRef.current) {
        const sameTrack = followKeyRef.current.startsWith(`${track.name}:`);
        if (followKeyRef.current !== "" && sameTrack && current !== "free" && !reducedMotion())
          transitionRef.current = { from: cameraRef.current, start: now };
        else transitionRef.current = null;
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

      // ── Trail of the selected car: what the source remembers (60 s), up to the car now ──
      const m = state.output === null ? null : state.output === "steer" ? 0 : 1;
      const trail = state.showTrail ? world.trail : null;
      const points: number[][] = [];
      const values: number[] = [];
      const steps: number[] = [];
      if (trail !== null) {
        trail.steps.forEach((s, k) => {
          steps.push(s);
          points.push([trail.x[k], trail.y[k]]);
          if (m !== null) values.push(trail.action[k]?.[m] ?? 0);
        });
        points.push([carNow.x, carNow.y]);
      }

      // ── Rays of the selected car at the shown moment (from the source) ──
      const shown = world.shown;
      const sensitivity = shown?.explain && m !== null ? shown.explain.sensitivity[m].slice(0, 5) : null;
      const highlight: number[] = [];
      if (state.neuron !== null && state.neuron.layer === 0 && state.neuron.index < 5) highlight.push(state.neuron.index);
      if (state.connection !== null && state.connection.layer === 0 && state.connection.from < 5) highlight.push(state.connection.from);
      if (shown?.neuron && state.neuron !== null) highlight.push(...shown.neuron.input_rays);

      const theme = readWorldTheme(canvas);
      drawScene(ctx, theme, {
        size,
        camera,
        track,
        art: trackArt(track),
        cars,
        colors: cars.map((_, i) => bodyColor(theme, bodyOf(scenario, i))),
        order: drawOrder(scenario, cars.length),
        elite: info?.elite ?? [],
        ghostCar: info?.ghost_car ?? null,
        selected,
        rays:
          state.showRays && shown !== null && shown.rays.length > 0
            ? {
                x: shownPose.x,
                y: shownPose.y,
                psi: shownPose.psi,
                distances: shown.rays,
                angles: scenario.ray_angles,
                range: scenario.ray_range,
                highlight,
                sensitivity,
              }
            : null,
        trail: trail !== null ? { points, values: m !== null ? values : null } : null,
        moment: looking ? { x: shownPose.x, y: shownPose.y, psi: shownPose.psi, label: `STEP ${world.lookStep}` } : null,
        ghost: world.whatIf?.ghost ?? null,
        selectedLabel: nameTag(world),
        ghostLabel: `GHOST · GEN ${info?.ghost_unit ?? ""}`,
        compact,
      });
      trailScreenRef.current = {
        steps,
        screen: steps.map((_, i) => worldToScreen(camera, size, points[i][0], points[i][1])),
      };
      canvas.dataset.cameraMode = current;
      canvas.dataset.cars = cars.map((c) => worldToScreen(camera, size, c.x, c.y).map(Math.round).join(":")).join(",");
      canvas.dataset.step = String(world.step);
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
    const world = getWorld();
    if (world === null) return;
    const state = useApp.getState();
    const rect = event.currentTarget.getBoundingClientRect();
    const px = press.x - rect.left;
    const py = press.y - rect.top;
    // A click selects the nearest car within 24 px …
    let best: number | null = null;
    let bestDistance = 24;
    world.cars.forEach((pose, i) => {
      const [sx, sy] = worldToScreen(cameraRef.current, size, pose.x, pose.y);
      const d = Math.hypot(sx - px, sy - py);
      if (d <= bestDistance) {
        bestDistance = d;
        best = i;
      }
    });
    if (best !== null && best !== world.selected) {
      state.selectCar(best);
      return;
    }
    if (best !== null) {
      state.lookAt(null); // clicking the selected car: back to now
      return;
    }
    // … or, while paused, a point of the trail: look at that step.
    if (world.playing) return;
    const { steps, screen } = trailScreenRef.current;
    let step: number | null = null;
    let nearest = 12;
    screen.forEach(([sx, sy], i) => {
      const d = Math.hypot(sx - px, sy - py);
      if (d <= nearest) {
        nearest = d;
        step = steps[i];
      }
    });
    if (step !== null) state.lookAt(step);
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
