import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useLive } from "../../state/live";
import { useSelection } from "../../state/selection";
import { type Camera, fitCamera, followCamera, nearestCar, type Size } from "./geometry";
import { drawScene, readTheme } from "./draw";
import { Hud } from "./Hud";
import { Controls } from "./Controls";
import { useDrivingKeys } from "./useDrivingKeys";

type CameraMode = "follow" | "fit";

const DEFAULT_ZOOM = 6; // pixels per metre when following the car
const COMPACT_ZOOM = 4;

/**
 * The Environment view: the track, the cars and the selected car's rays drawn on a canvas, with
 * a HUD and (in the full version) controls. `compact` is the small version for the companion
 * drawer: a canvas that follows the car, with a one-line HUD. It only displays what the Python
 * server sends (frames in the live store) and sends commands back.
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
  const [size, setSize] = useState<Size>({ w: 0, h: 0 });
  const [mode, setMode] = useState<CameraMode>("follow");
  const [zoom, setZoom] = useState(compact ? COMPACT_ZOOM : DEFAULT_ZOOM);

  const geometry = useLive((s) => s.geometry);
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
    const { frame, scenario, geometry: track } = useLive.getState();
    const selected = frame?.selected?.index ?? 0;
    const car = frame === null ? null : frame.cars;
    const camera =
      mode === "follow" && car !== null && car.x.length > selected
        ? followCamera(car.x[selected], car.y[selected], zoom)
        : fitCamera(fitPoints, size);
    cameraRef.current = camera;
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawScene(ctx, readTheme(canvas), { size, camera, geometry: track, scenario, frame, selected, compact });
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
    const unsubscribe = useLive.subscribe(schedule);
    schedule();
    return () => {
      unsubscribe();
      cancelAnimationFrame(pending);
    };
  }, [draw]);

  // Clicking a car selects it: its rays are drawn and the other views can follow (§8.2).
  const onPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const frame = useLive.getState().frame;
    if (frame === null) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const index = nearestCar(
      cameraRef.current,
      size,
      frame.cars.x,
      frame.cars.y,
      event.clientX - rect.left,
      event.clientY - rect.top,
      24,
    );
    if (index !== null) {
      useLive.getState().send({ type: "select_car", index });
      useSelection.getState().select({ carId: index });
    }
  };

  const zoomBy = (factor: number) => setZoom((z) => Math.min(40, Math.max(1, z * factor)));

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
          style={{ width: size.w, height: size.h }}
          onPointerDown={onPointerDown}
          aria-label="Top-down view of the track and the cars"
          role="img"
        />
        <Hud compact={compact} />
      </div>
      {!compact && (
        <p className="env-hint">
          Drive with the arrow keys or W A S D (up accelerates, down brakes). Press Play first: the
          car only moves while the simulation runs. Click a car to select it.
        </p>
      )}
    </section>
  );
}
