import { useMemo, useRef } from "react";

import { carName, carRole, type Track } from "../../source/model";
import { useWorld } from "../../source/world";
import { num, STATUS, useApp } from "../../state/store";
import { Button } from "../../ui/controls";
import { Icon } from "../../ui/Icon";
import { bodyCss, bodyOf } from "./cars";

/** The minimap's frame for a track: computed once per outline. */
function useTrackBox(track: Track | null) {
  return useMemo(() => {
    if (track === null) return null;
    const pts = [...track.left, ...track.right];
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const minX = Math.min(...xs);
    const maxY = Math.max(...ys);
    const w = Math.max(...xs) - minX;
    const h = maxY - Math.min(...ys);
    const path = (loop: number[][]) => `M${loop.map(([x, y]) => `${(x - minX).toFixed(1)} ${(maxY - y).toFixed(1)}`).join("L")}Z`;
    return { minX, maxY, w, h, road: path(track.centerline) };
  }, [track]);
}

/**
 * The minimap: the whole circuit with every car as a dot, shown while the camera is close in.
 * It answers "where on the lap is this?" without leaving follow mode. Clicking a dot selects
 * that car.
 */
export function Minimap() {
  const world = useWorld();
  const selectCar = useApp((s) => s.selectCar);
  const box = useTrackBox(world?.track ?? null);
  if (world === null || box === null) return null;
  const { track, scenario, cars, selected } = world;
  const pad = 14;
  const me = cars[selected];
  return (
    <div className="minimap" data-testid="minimap">
      <svg viewBox={`${-pad} ${-pad} ${box.w + 2 * pad} ${box.h + 2 * pad}`} role="img" aria-label="Map of the whole track">
        <path d={box.road} className="minimap-road" strokeWidth={track.width} />
        <path d={box.road} className="minimap-centre" />
        {cars.map((p, i) =>
          i === selected ? null : (
            <circle
              key={i}
              cx={p.x - box.minX}
              cy={box.maxY - p.y}
              r={5}
              className="minimap-car"
              style={{ fill: bodyCss(bodyOf(scenario, i)) }}
              opacity={p.status === 0 ? 1 : 0.4}
              onClick={() => selectCar(i)}
            />
          ),
        )}
        {me !== undefined && <circle cx={me.x - box.minX} cy={box.maxY - me.y} r={9} className="minimap-selected" />}
      </svg>
    </div>
  );
}

/** Words for the colours on the track. One row; it never covers the road in whole-track view. */
export function RaceLegend() {
  const world = useWorld();
  const output = useApp((s) => s.output);
  if (world === null) return null;
  const info = world.scenario.population;
  const many = info === null && world.scenario.n_cars > 10;
  const rays = output !== null && world.shown?.explain != null;
  const ghost = world.whatIf?.ghost != null;
  // One car on its own, nothing selected: the legend would be an empty pill.
  if (info === null && !many && world.cars.length <= 1 && !rays && !ghost) return null;
  return (
    <div className="stage-legend" data-testid="race-legend">
      {info !== null && (
        <>
          <span className="stage-legend-group">
            <b>Rank</b>
            <span className="rank-ramp" aria-hidden="true">
              {[1, 2, 3, 4, 5].map((k) => (
                <i key={k} style={{ background: `var(--car-rank-${k})` }} />
              ))}
            </span>
            best fifth → worst
          </span>
          <span className="stage-legend-group">
            <i className="legend-ring" aria-hidden="true" /> elite
          </span>
          {info.ghost_unit !== null && (
            <span className="stage-legend-group">
              <i className="legend-ghost" aria-hidden="true" /> ghost, gen {info.ghost_unit} champion
            </span>
          )}
        </>
      )}
      {many && <span className="stage-legend-group">{world.scenario.n_cars} cars: the selected one has the ring and the name tag</span>}
      {world.cars.length > 1 && (
        <span className="stage-legend-group">
          <i className="legend-faded" aria-hidden="true" /> stopped
        </span>
      )}
      {rays && (
        <span className="stage-legend-group" data-testid="ray-legend">
          <i className="legend-line" style={{ background: "var(--track-pos)" }} aria-hidden="true" />
          longer ray → {output === "steer" ? "steer right" : "accelerate"}
          <i className="legend-line" style={{ background: "var(--track-neg)" }} aria-hidden="true" />
          {output === "steer" ? "steer left" : "brake"} · thicker = stronger
        </span>
      )}
      {ghost && (
        <span className="stage-legend-group" data-testid="ghost-legend">
          <i className="legend-line" style={{ background: "var(--track-select)" }} aria-hidden="true" /> what-if action held{" "}
          {world.source === "live" ? "2 s, by Python's physics" : "2 s"}
          <i className="legend-dash" aria-hidden="true" /> real action
        </span>
      )}
    </div>
  );
}

/**
 * One row of the strip as two filled areas, above and below the zero line: the positive part of
 * the signal (max(v, 0)) and the negative part (min(v, 0)). Two paths whatever the number of
 * remembered steps, so 1,800 live steps cost the same as 225 recorded frames.
 */
function areaPaths(values: number[], xs: number[], mid: number, half: number, top: number): { pos: string; neg: string } {
  if (values.length === 0) return { pos: "", neg: "" };
  const y = (v: number) => (mid - (v / top) * half).toFixed(1);
  const x = (i: number) => xs[i].toFixed(1);
  let pos = `M${x(0)} ${mid}`;
  let neg = pos;
  values.forEach((v, i) => {
    pos += `L${x(i)} ${y(Math.max(0, v))}`;
    neg += `L${x(i)} ${y(Math.min(0, v))}`;
  });
  const end = `L${x(values.length - 1)} ${mid}Z`;
  return { pos: pos + end, neg: neg + end };
}

/**
 * The look-back timeline (production: Trail.tsx, a bare slider). Here the slider sits on what it
 * scrubs: the car's steer and throttle over the remembered 60 s, so interesting moments (a hard
 * brake, a steering reversal) can be seen before they are looked at.
 */
export function TrailStrip() {
  const world = useWorld();
  const output = useApp((s) => s.output);
  const lookAt = useApp((s) => s.lookAt);
  const stripRef = useRef<SVGSVGElement>(null);
  if (world === null) return null;
  const trail = world.trail;
  const steps = trail?.steps ?? [];
  const first = steps[0] ?? world.step;
  const last = world.step;
  const span = Math.max(1, last - first);
  const W = 1000;
  const H = 56;
  const x = (s: number) => ((s - first) / span) * W;
  const xs = steps.map(x);
  const steer = trail?.action.map((a) => a[0] ?? 0) ?? [];
  const throttle = trail?.action.map((a) => a[1] ?? 0) ?? [];
  // Each row is scaled to its own largest value in the window (printed beside the row), so
  // small steering still shows.
  const peak = (vs: number[]) => vs.reduce((m, v) => Math.max(m, Math.abs(v)), 0.1);
  const steerTop = peak(steer);
  const throttleTop = peak(throttle);
  const s = areaPaths(steer, xs, 14, 12, steerTop);
  const th = areaPaths(throttle, xs, 42, 12, throttleTop);
  const shown = world.lookStep ?? last;
  const dt = world.scenario.dt;
  const ago = (last - shown) * dt;
  const playing = world.playing;

  const pick = (event: React.PointerEvent<SVGSVGElement>) => {
    if (playing || steps.length === 0) return;
    const rect = stripRef.current!.getBoundingClientRect();
    const u = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    const wanted = first + u * span;
    // Snap to a remembered step: the source can only show steps it remembers.
    let best = steps[0];
    for (const step of steps) if (Math.abs(step - wanted) < Math.abs(best - wanted)) best = step;
    lookAt(best >= last ? null : best);
  };

  return (
    <div className="trailstrip" data-testid="trailstrip" data-disabled={playing}>
      <div className="trailstrip-side">
        <span className="eyebrow">Look back</span>
        <span className="trailstrip-now num">
          {world.lookStep === null ? "now" : `−${ago.toFixed(1)} s`}
          <small>step {shown}</small>
        </span>
      </div>
      <div className="trailstrip-chart">
        <svg
          ref={stripRef}
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            pick(e);
          }}
          onPointerMove={(e) => e.buttons === 1 && pick(e)}
        >
          <line x1={0} x2={W} y1={14} y2={14} className="trailstrip-zero" />
          <line x1={0} x2={W} y1={42} y2={42} className="trailstrip-zero" />
          <g className={`trailstrip-area${output === "throttle" ? " dim" : ""}`}>
            <path d={s.pos} className="ts-pos" />
            <path d={s.neg} className="ts-neg" />
          </g>
          <g className={`trailstrip-area${output === "steer" ? " dim" : ""}`}>
            <path d={th.pos} className="ts-pos" />
            <path d={th.neg} className="ts-neg" />
          </g>
          <line x1={x(shown)} x2={x(shown)} y1={0} y2={H} className="trailstrip-cursor" vectorEffect="non-scaling-stroke" />
        </svg>
        <span className="trailstrip-label" style={{ top: 2 }}>
          steer ±{steerTop.toFixed(2)}
        </span>
        <span className="trailstrip-label" style={{ top: 30 }}>
          throttle ±{throttleTop.toFixed(2)}
        </span>
        <span className="trailstrip-scale num">
          <span>−{(span * dt).toFixed(0)} s</span>
          <span>now</span>
        </span>
      </div>
      <div className="trailstrip-side trailstrip-end">
        {playing ? (
          <span className="trailstrip-hint">Pause to look back at an earlier decision.</span>
        ) : world.lookStep === null ? (
          <span className="trailstrip-hint">Drag here, or click a point of the trail on the track.</span>
        ) : (
          <Button icon="right" onClick={() => lookAt(null)} testId="back-to-now">
            Back to now
          </Button>
        )}
      </div>
    </div>
  );
}

const STATUS_ICON = ["", "cross", "flag", "stall", "clock"] as const;

/**
 * Standings: every car in the order they are on the road right now. In a GA race each row also
 * has the rank the GA gave it (the colour band) and its fitness F; for other drivers, its speed.
 * A click selects the car in every view.
 */
export function Standings() {
  const world = useWorld();
  const selectCar = useApp((s) => s.selectCar);
  if (world === null) return <div className="skeleton" style={{ height: 240, margin: 12 }} />;
  const { scenario, track, cars, selected } = world;
  const info = scenario.population;
  const total = scenario.target_laps * track.length;
  const rows = cars.map((pose, i) => ({ i, pose })).sort((a, b) => b.pose.progress - a.pose.progress);
  const counts = [0, 0, 0, 0, 0];
  rows.forEach((r) => (counts[r.pose.status] += 1));
  return (
    <div className="standings" data-testid="standings">
      <div className="standings-summary">
        <span>
          <b className="num">{counts[0]}</b> running
        </span>
        <span>
          <b className="num">{counts[2]}</b> finished
        </span>
        <span>
          <b className="num">{counts[1]}</b> crashed
        </span>
        <span>
          <b className="num">{counts[4] + counts[3]}</b> stopped
        </span>
      </div>
      <table className="table standings-table">
        <thead>
          <tr>
            <th className="r">Pos</th>
            <th>Car</th>
            <th>Progress</th>
            {info !== null ? (
              <th className="r" data-tip={`The GA's fitness over all ${info.n_starts} starts`}>
                F
              </th>
            ) : (
              <th className="r">km/h</th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ i, pose }, position) => {
            const role = carRole(scenario, i);
            const ghost = role === "ghost";
            const status = STATUS[pose.status] ?? STATUS[0];
            return (
              <tr key={i} data-selected={i === selected} onClick={() => selectCar(i)} data-testid="standings-row">
                <td className="r num">{position + 1}</td>
                <td>
                  <span className="standings-car">
                    {ghost ? (
                      <i className="legend-ghost" aria-hidden="true" />
                    ) : (
                      <i
                        className="standings-swatch"
                        style={{ background: bodyCss(bodyOf(scenario, i)) }}
                        data-elite={info?.elite.includes(i) ?? false}
                        aria-hidden="true"
                      />
                    )}
                    {ghost ? "Ghost" : carName(scenario, i)}
                    {role === "champion" && (
                      <span className="standings-role" data-tip="This generation's champion">
                        <Icon name="crown" size={12} />
                      </span>
                    )}
                    {pose.status !== 0 && (
                      <span className="standings-status" data-kind={status.kind} data-tip={status.label}>
                        <Icon name={STATUS_ICON[pose.status] as "cross"} size={12} />
                      </span>
                    )}
                  </span>
                </td>
                <td>
                  <span className="standings-progress">
                    <span style={{ width: `${Math.max(0, Math.min(1, pose.progress / total)) * 100}%` }} data-kind={status.kind} />
                  </span>
                </td>
                <td className="r num">
                  {info === null ? (pose.v * 3.6).toFixed(0) : ghost ? num(info.ghost_fitness ?? 0, 3) : num(info.fitness[i], 3)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
