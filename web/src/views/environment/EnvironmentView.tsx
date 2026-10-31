import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useLive } from "../../state/live";
import { activeMoment, useMoment, usePlayback } from "../../state/playback";
import { raysFor, useSelection } from "../../state/selection";
import { activeGhost, ghostGap, raySensitivity } from "../network/inspect";
import {
  type Camera,
  fitCamera,
  followCamera,
  nearestCar,
  panBy,
  scaleLimits,
  type Size,
  wheelFactor,
  worldToScreen,
  zoomAt,
} from "./geometry";
import { drawScene, readTheme } from "./draw";
import { Hud } from "./Hud";
import { Controls } from "./Controls";
import { useDrivingKeys } from "./useDrivingKeys";
import { TrailPanel } from "./Trail";
import { nearestTrailStep, trailValues } from "./trailLayout";

// "free": the user moved or zoomed the view with the mouse; it stays put until Follow car /
// Whole track or a double-click (task 4.10).
export type CameraMode = "follow" | "fit" | "free";

const DEFAULT_ZOOM = 6; // pixels per metre when following the car
const COMPACT_ZOOM = 4;
const DRAG_THRESHOLD = 4; // pixels the pointer must move before a press counts as a drag, not a click
const FOLLOW_ZOOM = { min: 1, max: 40 }; // zoom range while following (as the +/− buttons had)

/**
 * The Environment view: the track, the cars and the selected car's rays drawn on a canvas, with
 * a HUD and (in the full version) controls. `compact` is the small version for the companion
 * drawer: a canvas that follows the car, with a one-line HUD. It only displays what the Python
 * server sends (frames in the live store) and sends commands back. In both, the mouse moves the
 * camera (task 4.10): the wheel zooms (around the cursor, or around the car while following), a
 * drag pans, a plain click selects, and a double-click goes back to following the car.
 */
export function EnvironmentView({
  compact = false,
  headerExtra,
}: {
  compact?: boolean;
  headerExtra?: ReactNode;
}) {
  useDrivingKeys();

  const boxRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<Camera>({ cx: 0, cy: 0, scale: 1 });
  // Where each trail point was drawn (CSS pixels), for clicking a point (task 2.11).
  const trailRef = useRef<{ steps: number[]; screen: [number, number][] }>({ steps: [], screen: [] });
  const [size, setSize] = useState<Size>({ w: 0, h: 0 });
  const [mode, setMode] = useState<CameraMode>("follow");
  const [zoom, setZoom] = useState(compact ? COMPACT_ZOOM : DEFAULT_ZOOM);
  // The camera of the free mode, and what the mouse handlers need between events.
  const freeRef = useRef<Camera>({ cx: 0, cy: 0, scale: 1 });
  const fitScaleRef = useRef(1);
  const modeRef = useRef<CameraMode>(mode);
  modeRef.current = mode;
  const scheduleRef = useRef<() => void>(() => {});
  const pressRef = useRef<{ x: number; y: number; start: Camera; dragging: boolean } | null>(null);
  const [dragging, setDragging] = useState(false);

  const geometry = useLive((s) => s.geometry);
  const frame = useLive((s) => s.frame);
  // The rays the shared selection points at (a selected neuron, input or weight), for tests.
  const highlightAttribute = useSelectedRays().join(",");
  // The selected output's sensitivity to each ray, when an output is selected (task 2.10).
  const output = useSelection((s) => s.output);
  // Looking at an earlier step (task 2.11): its explanation replaces the present one.
  const moment = useMoment();
  const liveExplain = useLive((s) => s.frame?.selected?.explain ?? null); // a stable object per frame
  const explain = moment === null ? liveExplain : moment.explain;
  const sensitivityRow = raySensitivity(explain, outputIndex(output));
  const sensitivityAttribute = sensitivityRow === null ? "" : sensitivityRow.join(",");
  // The what-if ghost, while a what-if question about this car at this step is open (ADR-026).
  const ghost = useGhost();
  const ghostAttribute = ghost === null ? "" : ghost.what_if_end.join(",");
  const fitPoints = useMemo(
    () => (geometry === null ? [] : [...geometry.left, ...geometry.right]),
    [geometry],
  );

  // Keep `size` equal to the size of the box the canvas fills.
  useEffect(() => {
    const box = boxRef.current;
    if (box === null) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ w: Math.floor(entry.contentRect.width), h: Math.floor(entry.contentRect.height) });
    });
    observer.observe(box);
    return () => observer.disconnect();
  }, []);

  // Match the canvas's pixel buffer to its size on screen (sharp on high-density displays).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(size.w * ratio));
    canvas.height = Math.max(1, Math.round(size.h * ratio));
  }, [size]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas === null || !ctx || size.w === 0 || size.h === 0) return;
    const live = useLive.getState();
    const { frame, scenario, geometry: track } = live;
    const selected = frame?.selected?.index ?? 0;
    const output = outputIndex(useSelection.getState().output);
    const moment = activeMoment(live, usePlayback.getState().viewStep);
    // Emphasise the rays behind whatever is selected in the Network view (§8.2).
    const highlight = raysFor(useSelection.getState(), frame?.selected);
    const sensitivityRays = raySensitivity(moment === null ? frame?.selected?.explain : moment.explain, output);
    const car = frame === null ? null : frame.cars;
    const trail = live.trail !== null && live.trail.car === selected ? live.trail : null;
    // Follow the selected car, or the place it was at while an earlier step is looked at; or show
    // the whole track; or keep the view the mouse left (free).
    const fit = fitCamera(fitPoints, size);
    fitScaleRef.current = fit.scale;
    const camera =
      mode === "free"
        ? freeRef.current
        : mode === "follow" && moment !== null
          ? followCamera(moment.x, moment.y, zoom)
          : mode === "follow" && car !== null && car.x.length > selected
            ? followCamera(car.x[selected], car.y[selected], zoom)
            : fit;
    cameraRef.current = camera;
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawScene(ctx, readTheme(canvas), {
      size,
      camera,
      geometry: track,
      scenario,
      frame,
      selected,
      compact,
      highlight,
      raySensitivity: sensitivityRays,
      ghost: activeGhost(live, useSelection.getState().whatIf),
      trail:
        trail === null
          ? null
          : {
              steps: trail.steps,
              points: trail.x.map((x, k) => [x, trail.y[k]]),
              values: trailValues(trail, output),
            },
      moment:
        moment === null
          ? null
          : { step: moment.step, x: moment.x, y: moment.y, psi: moment.psi, rays: moment.rays },
    });
    const trailScreen =
      trail === null
        ? []
        : trail.x.map(
            (x, k) => worldToScreen(camera, size, x, trail.y[k]).map((v) => Math.round(v)) as [number, number],
          );
    trailRef.current = { steps: trail?.steps ?? [], screen: trailScreen };
    // For browser tests: each trail point as step:x:y on screen, and the step being looked at.
    canvas.dataset.trail = trailScreen.map(([sx, sy], k) => `${trail?.steps[k]}:${sx}:${sy}`).join(",");
    canvas.dataset.moment = moment === null ? "" : String(moment.step);
    // The camera (centre x, centre y in metres, pixels per metre) and its mode, for browser tests.
    canvas.dataset.camera = `${camera.cx.toFixed(3)}:${camera.cy.toFixed(3)}:${camera.scale.toFixed(4)}`;
    canvas.dataset.cameraMode = mode;
    canvas.dataset.trailValues = (trailValues(trail, output) ?? []).join(","); // what colours it
    // Where each car is on screen (CSS pixels), so browser tests can click a particular car.
    canvas.dataset.cars =
      car === null
        ? ""
        : car.x
            .map((x, i) => worldToScreen(camera, size, x, car.y[i]).map((v) => Math.round(v)).join(":"))
            .join(",");
  }, [size, mode, zoom, compact, fitPoints]);

  // Redraw when anything the picture depends on changes, at most once per display frame.
  useEffect(() => {
    let pending = 0;
    const schedule = () => {
      if (pending === 0) {
        pending = requestAnimationFrame(() => {
          pending = 0;
          draw();
        });
      }
    };
    scheduleRef.current = schedule;
    const unsubscribeLive = useLive.subscribe(schedule);
    const unsubscribeSelection = useSelection.subscribe(schedule);
    const unsubscribePlayback = usePlayback.subscribe(schedule);
    schedule();
    return () => {
      unsubscribeLive();
      unsubscribeSelection();
      unsubscribePlayback();
      cancelAnimationFrame(pending);
    };
  }, [draw]);

  // Mouse camera (task 4.10). The wheel zooms: around the car while following (only the zoom
  // level changes), otherwise around the cursor, which makes the view free. Registered by hand
  // because React's wheel listener cannot stop the page from scrolling.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas === null) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY; // lines → pixels
      const factor = wheelFactor(delta);
      if (modeRef.current === "follow") {
        setZoom((z) => Math.min(FOLLOW_ZOOM.max, Math.max(FOLLOW_ZOOM.min, z * factor)));
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const limits = scaleLimits(fitScaleRef.current);
      freeRef.current = zoomAt(cameraRef.current, size, event.clientX - rect.left, event.clientY - rect.top, factor, limits);
      setMode("free");
      scheduleRef.current();
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [size]);

  // A press starts either a click or, once the pointer has moved DRAG_THRESHOLD pixels, a drag
  // that pans the view from wherever it was.
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
    if (modeRef.current !== "free") setMode("free");
    scheduleRef.current();
  };

  const onPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const press = pressRef.current;
    pressRef.current = null;
    setDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (press !== null && !press.dragging) onClick(press.x, press.y, event.currentTarget);
  };

  // Clicking a car selects it: its rays are drawn and the other views can follow (§8.2).
  // Clicking a point of the trail (while paused) looks at that step (task 2.11).
  const onClick = (clientX: number, clientY: number, target: HTMLCanvasElement) => {
    const frame = useLive.getState().frame;
    if (frame === null) return;
    const rect = target.getBoundingClientRect();
    const px = clientX - rect.left;
    const py = clientY - rect.top;
    const index = nearestCar(cameraRef.current, size, frame.cars.x, frame.cars.y, px, py, 24);
    if (index !== null && index !== frame.selected?.index) {
      useSelection.getState().selectCar(index);
      return;
    }
    const { steps, screen } = trailRef.current;
    const step = frame.playing ? null : nearestTrailStep(screen, steps, px, py, 12);
    // The newest trail points lie under the car: a click nearer the car's centre means the car.
    const distanceTo = ([sx, sy]: number[]) => Math.hypot(sx - px, sy - py);
    const onCar =
      index !== null &&
      (step === null ||
        distanceTo(worldToScreen(cameraRef.current, size, frame.cars.x[index], frame.cars.y[index])) <
          distanceTo(screen[steps.indexOf(step)]));
    if (onCar) usePlayback.getState().backToLive(); // clicking the selected car: back to now
    else if (step !== null) usePlayback.getState().view(step);
  };

  // The +/− buttons: the zoom level while following, otherwise around the centre of the view.
  const zoomBy = (factor: number) => {
    if (mode === "follow") {
      setZoom((z) => Math.min(FOLLOW_ZOOM.max, Math.max(FOLLOW_ZOOM.min, z * factor)));
      return;
    }
    freeRef.current = zoomAt(cameraRef.current, size, size.w / 2, size.h / 2, factor, scaleLimits(fitScaleRef.current));
    setMode("free");
    scheduleRef.current(); // already free: the mode does not change, so ask for the redraw
  };

  return (
    <section
      className={compact ? "view view-compact env-view" : "view env-view"}
      data-testid={compact ? "environment-compact" : "environment-view"}
    >
      <div className="view-header">
        <h2>{compact ? "Environment (compact)" : "Environment"}</h2>
        {headerExtra}
      </div>
      {!compact && (
        <Controls
          mode={mode}
          onModeChange={setMode}
          onZoomIn={() => zoomBy(1.25)}
          onZoomOut={() => zoomBy(0.8)}
        />
      )}
      <div className="env-stage" ref={boxRef}>
        <canvas
          ref={canvasRef}
          className="env-canvas"
          data-testid="environment-canvas"
          data-highlighted-rays={highlightAttribute}
          data-ray-sensitivity={sensitivityAttribute}
          data-ghost={ghostAttribute}
          style={{ width: size.w, height: size.h }}
          data-dragging={dragging ? "true" : undefined}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onDoubleClick={() => setMode("follow")}
          aria-label="Top-down view of the track and the cars"
          role="img"
        />
        <Hud compact={compact} />
        {compact && mode !== "follow" && (
          <button
            type="button"
            className="env-recenter"
            data-testid="compact-follow"
            onClick={() => setMode("follow")}
          >
            Follow car
          </button>
        )}
        {!compact && (ghost !== null || sensitivityRow !== null) && (
          <div className="stage-legends">
            {ghost !== null && (
              <div className="stage-legend" data-testid="ghost-legend">
                <strong>What-if ghost:</strong> where car {(frame?.selected?.index ?? 0) + 1} would
                be in {ghost.seconds} s holding the what-if action (solid line, see-through car)
                instead of its real action (dashed); the two endings are{" "}
                <span data-testid="ghost-gap">{ghostGap(ghost).toFixed(1)}</span> m apart. Held
                actions only: the network would really decide again every step.
              </div>
            )}
            {sensitivityRow !== null && (
              <div className="stage-legend" data-testid="ray-legend">
                Rays coloured by how {output} reacts to them:{" "}
                <span className="ray-legend-pos">blue</span> = a longer ray pushes {output}{" "}
                {output === "steer" ? "right" : "up (accelerate)"},{" "}
                <span className="ray-legend-neg">orange</span> ={" "}
                {output === "steer" ? "left" : "down (brake)"}; thicker = stronger.
              </div>
            )}
          </div>
        )}
      </div>
      {!compact && <TrailPanel />}
      {!compact && (
        <p className="env-hint">
          Drive with the arrow keys or W A S D (up accelerates, down brakes). Press Play first: the
          car only moves while the simulation runs. Scroll over the track to zoom, drag to move the
          view, double-click to follow the car again. Click a car to select it: the Network page
          and the drawer then show that car. While paused, click a point on its trail (or drag Look
          back) to see the network at that earlier step. With the Neuron driver, set its weights on the
          Network page.
        </p>
      )}
    </section>
  );
}

/** The rays highlighted for the current selection, as a React value (re-renders when it changes). */
function useSelectedRays(): number[] {
  const neuron = useSelection((s) => s.neuron);
  const connection = useSelection((s) => s.connection);
  const selected = useLive((s) => s.frame?.selected);
  return raysFor({ neuron, connection }, selected);
}

/** Index of a named output in the action (steer 0, throttle 1), or null. */
function outputIndex(output: "steer" | "throttle" | null): number | null {
  return output === null ? null : output === "steer" ? 0 : 1;
}

/** The open what-if ghost as a React value (re-renders when it appears or goes). */
function useGhost() {
  const result = useLive((s) => s.whatIfResult);
  const frame = useLive((s) => s.frame);
  const overrides = useSelection((s) => s.whatIf);
  return activeGhost({ whatIfResult: result, frame }, overrides);
}
