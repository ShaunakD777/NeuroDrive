import { useMemo, useRef } from "react";

import { poseAt } from "../../mock/mockServer";
import { carName, carRole, num, STATUS, useApp } from "../../state/store";
import { Button } from "../../ui/controls";
import { Icon } from "../../ui/Icon";
import { rankBand } from "./Stage";

/** A whole-number version of the playback position: re-renders once per recorded frame, not per display frame. */
function useKey(): number {
  return useApp((s) => Math.floor(s.t));
}

/**
 * The minimap: the whole circuit with every car as a dot, shown while the camera is close in.
 * It answers "where on the lap is this?" without leaving follow mode. Clicking a dot selects
 * that car.
 */
export function Minimap() {
  const data = useApp((s) => s.data);
  const car = useApp((s) => s.car);
  const selectCar = useApp((s) => s.selectCar);
  const key = useKey();
  const box = useMemo(() => {
    if (data === null) return null;
    const pts = [...data.track.left, ...data.track.right];
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const minX = Math.min(...xs);
    const maxY = Math.max(...ys);
    const w = Math.max(...xs) - minX;
    const h = maxY - Math.min(...ys);
    const path = (loop: number[][]) => `M${loop.map(([x, y]) => `${(x - minX).toFixed(1)} ${(maxY - y).toFixed(1)}`).join("L")}Z`;
    return { minX, maxY, w, h, road: path(data.track.centerline) };
  }, [data]);
  if (data === null || box === null) return null;
  const info = data.race.scenario.population;
  const pad = 14;
  return (
    <div className="minimap" data-testid="minimap">
      <svg viewBox={`${-pad} ${-pad} ${box.w + 2 * pad} ${box.h + 2 * pad}`} role="img" aria-label="Map of the whole track">
        <path d={box.road} className="minimap-road" strokeWidth={data.track.width} />
        <path d={box.road} className="minimap-centre" />
        {Array.from({ length: data.race.scenario.n_cars }, (_, i) => {
          if (i === car) return null;
          const p = poseAt(data.race, key, i);
          return (
            <circle
              key={i}
              cx={p.x - box.minX}
              cy={box.maxY - p.y}
              r={5}
              className="minimap-car"
              style={{ fill: i < info.size ? `var(--car-rank-${rankBand(info.ranks[i], info.size) + 1})` : "#c9ccd2" }}
              opacity={p.status === 0 ? 1 : 0.4}
              onClick={() => selectCar(i)}
            />
          );
        })}
        {(() => {
          const p = poseAt(data.race, key, car);
          return <circle cx={p.x - box.minX} cy={box.maxY - p.y} r={9} className="minimap-selected" />;
        })()}
      </svg>
    </div>
  );
}

/** Words for the colours on the track. One row; it never covers the road in whole-track view. */
export function RaceLegend() {
  const data = useApp((s) => s.data);
  const output = useApp((s) => s.output);
  const whatIf = useApp((s) => Object.keys(s.whatIf).length > 0);
  if (data === null) return null;
  const info = data.race.scenario.population;
  return (
    <div className="stage-legend" data-testid="race-legend">
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
      <span className="stage-legend-group">
        <i className="legend-ghost" aria-hidden="true" /> ghost, gen {info.ghost_unit} champion
      </span>
      <span className="stage-legend-group">
        <i className="legend-faded" aria-hidden="true" /> stopped
      </span>
      {output !== null && (
        <span className="stage-legend-group" data-testid="ray-legend">
          <i className="legend-line" style={{ background: "var(--track-pos)" }} aria-hidden="true" />
          longer ray → {output === "steer" ? "steer right" : "accelerate"}
          <i className="legend-line" style={{ background: "var(--track-neg)" }} aria-hidden="true" />
          {output === "steer" ? "steer left" : "brake"} · thicker = stronger
        </span>
      )}
      {whatIf && (
        <span className="stage-legend-group" data-testid="ghost-legend">
          <i className="legend-line" style={{ background: "var(--track-select)" }} aria-hidden="true" /> what-if action held 2 s
          <i className="legend-dash" aria-hidden="true" /> real action
        </span>
      )}
    </div>
  );
}

/**
 * The look-back timeline (production: Trail.tsx, a bare slider). Here the slider sits on what it
 * scrubs: the car's steer and throttle over the remembered 60 s, so interesting moments (a hard
 * brake, a steering reversal) can be seen before they are looked at.
 */
export function TrailStrip() {
  const data = useApp((s) => s.data);
  const car = useApp((s) => s.car);
  const playing = useApp((s) => s.playing);
  const viewKey = useApp((s) => s.viewKey);
  const output = useApp((s) => s.output);
  const lookAt = useApp((s) => s.lookAt);
  const key = useKey();
  const stripRef = useRef<SVGSVGElement>(null);
  if (data === null) return null;
  const { race } = data;
  const secondsPerFrame = race.stepsPerFrame * race.scenario.dt;
  const back = Math.round(race.scenario.trail_seconds / secondsPerFrame);
  const first = Math.max(0, key - back);
  const count = key - first;
  const W = 1000;
  const H = 56;
  const x = (k: number) => (count === 0 ? W : ((k - first) / count) * W);

  // One thin bar per remembered frame: height = size, colour = sign. Each row is scaled to its
  // own largest value in the window (printed beside the row), so small steering still shows.
  const peak = (column: number[][]) => {
    let m = 0.1;
    for (let k = first; k <= key; k++) m = Math.max(m, Math.abs(column[k][car]));
    return m;
  };
  const bars = (column: number[][], mid: number, half: number, top: number) => {
    const out = [];
    const w = Math.max(1, W / Math.max(1, count + 1));
    for (let k = first; k <= key; k++) {
      const v = column[k][car];
      const h = (Math.abs(v) / top) * half;
      out.push(<rect key={k} x={x(k) - w / 2} y={v >= 0 ? mid - h : mid} width={w} height={h} className={v >= 0 ? "ts-pos" : "ts-neg"} />);
    }
    return out;
  };
  const steerTop = peak(race.frames.steer);
  const throttleTop = peak(race.frames.throttle);
  const shown = viewKey ?? key;
  const ago = (key - shown) * secondsPerFrame;

  const pick = (event: React.PointerEvent<SVGSVGElement>) => {
    if (playing || count === 0) return;
    const rect = stripRef.current!.getBoundingClientRect();
    const u = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    lookAt(first + Math.round(u * count));
  };

  return (
    <div className="trailstrip" data-testid="trailstrip" data-disabled={playing}>
      <div className="trailstrip-side">
        <span className="eyebrow">Look back</span>
        <span className="trailstrip-now num">
          {viewKey === null ? "now" : `−${ago.toFixed(1)} s`}
          <small>step {race.frames.step[shown]}</small>
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
          <g className={`trailstrip-area${output === "throttle" ? " dim" : ""}`}>{bars(race.frames.steer, 14, 12, steerTop)}</g>
          <g className={`trailstrip-area${output === "steer" ? " dim" : ""}`}>{bars(race.frames.throttle, 42, 12, throttleTop)}</g>
          <line x1={x(shown)} x2={x(shown)} y1={0} y2={H} className="trailstrip-cursor" vectorEffect="non-scaling-stroke" />
        </svg>
        <span className="trailstrip-label" style={{ top: 2 }}>
          steer ±{steerTop.toFixed(2)}
        </span>
        <span className="trailstrip-label" style={{ top: 30 }}>
          throttle ±{throttleTop.toFixed(2)}
        </span>
        <span className="trailstrip-scale num">
          <span>−{(count * secondsPerFrame).toFixed(0)} s</span>
          <span>now</span>
        </span>
      </div>
      <div className="trailstrip-side trailstrip-end">
        {playing ? (
          <span className="trailstrip-hint">Pause to look back at an earlier decision.</span>
        ) : viewKey === null ? (
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
 * Standings: every car of the generation in the order they are on the road right now, with the
 * rank the GA gave it (the colour band), how far it is, and its fitness F. A click selects the
 * car in every view.
 */
export function Standings() {
  const data = useApp((s) => s.data);
  const car = useApp((s) => s.car);
  const selectCar = useApp((s) => s.selectCar);
  const key = useKey();
  if (data === null) return null;
  const { race, track } = data;
  const info = race.scenario.population;
  const total = race.scenario.target_laps * track.length;
  const rows = Array.from({ length: race.scenario.n_cars }, (_, i) => ({ i, pose: poseAt(race, key, i) })).sort(
    (a, b) => b.pose.progress - a.pose.progress,
  );
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
            <th className="r" data-tip="The GA's fitness over all 10 starts">
              F
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ i, pose }, position) => {
            const role = carRole(data, i);
            const ghost = role === "ghost";
            const status = STATUS[pose.status] ?? STATUS[0];
            return (
              <tr key={i} data-selected={i === car} onClick={() => selectCar(i)} data-testid="standings-row">
                <td className="r num">{position + 1}</td>
                <td>
                  <span className="standings-car">
                    {ghost ? (
                      <i className="legend-ghost" aria-hidden="true" />
                    ) : (
                      <i
                        className="standings-swatch"
                        style={{ background: `var(--car-rank-${rankBand(info.ranks[i], info.size) + 1})` }}
                        data-elite={info.elite.includes(i)}
                        aria-hidden="true"
                      />
                    )}
                    {ghost ? "Ghost" : carName(data, i)}
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
                <td className="r num">{ghost ? num(info.ghost_fitness ?? 0, 3) : num(info.fitness[i], 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
