import { carName, carRole } from "../../source/model";
import { useWorld } from "../../source/world";
import { describeSteer, describeThrottle, num, STATUS, useApp } from "../../state/store";
import { Chip } from "../../ui/controls";
import { bodyCss, bodyOf } from "./cars";

/** A signed gauge on the dark HUD: the bar grows from the middle, the tick marks zero. */
function Gauge({ value, left, right }: { value: number; left: string; right: string }) {
  const v = Math.max(-1, Math.min(1, value));
  return (
    <div className="gauge">
      <span className="gauge-end">{left}</span>
      <span className="gauge-track">
        <span
          className={v < 0 ? "gauge-fill gauge-neg" : "gauge-fill gauge-pos"}
          style={{ left: v >= 0 ? "50%" : `${50 + v * 50}%`, width: `${Math.abs(v) * 50}%` }}
        />
      </span>
      <span className="gauge-end">{right}</span>
    </div>
  );
}

/** The five rays as a small fan: length = distance, colour = how close the wall is. */
function RayFan({ distances, angles, range, highlight }: { distances: number[]; angles: number[]; range: number; highlight: number[] }) {
  const R = 46;
  const cx = 60;
  const cy = 54;
  return (
    <svg className="rayfan" viewBox="0 0 120 60" aria-hidden="true">
      <path d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`} className="rayfan-range" />
      <path d={`M ${cx - R / 2} ${cy} A ${R / 2} ${R / 2} 0 0 1 ${cx + R / 2} ${cy}`} className="rayfan-range" />
      {distances.map((d, k) => {
        // Screen up is the car's forward; +angle is to the left.
        const a = Math.PI / 2 + angles[k];
        const f = d / range;
        const x = cx + Math.cos(a) * R * f;
        const y = cy - Math.sin(a) * R * f;
        const hit = d < range - 1e-6;
        const tone = !hit ? "none" : f < 0.2 ? "danger" : f < 0.5 ? "near" : "clear";
        const dim = highlight.length > 0 && !highlight.includes(k);
        return (
          <g key={k} opacity={dim ? 0.3 : 1}>
            <line x1={cx} y1={cy} x2={x} y2={y} className={`rayfan-ray rayfan-${tone}`} strokeWidth={highlight.includes(k) ? 3.5 : 2} />
            {hit && <circle cx={x} cy={y} r={2.6} className={`rayfan-dot rayfan-${tone}`} />}
          </g>
        );
      })}
      <path d={`M ${cx - 3} ${cy + 3} L ${cx} ${cy - 5} L ${cx + 3} ${cy + 3} Z`} className="rayfan-car" />
    </svg>
  );
}

/**
 * The telemetry strip docked along the bottom of the stage. It replaces the tall HUD table that
 * covered the upper-left of the track: one row, grouped as a dashboard (who, how fast, what it is
 * doing, how far it is, what it senses), every number in the tabular number face. The fitness
 * group appears only when a GA generation is racing.
 */
export function Telemetry() {
  const world = useWorld();
  const neuron = useApp((s) => s.neuron);
  if (world === null || world.shown === null) return null;

  const { scenario, track, shown } = world;
  const car = world.selected;
  const pose = shown.pose;
  const info = scenario.population;
  const status = STATUS[pose.status] ?? STATUS[0];
  const role = carRole(scenario, car);
  const isGhost = role === "ghost";
  const laps = pose.progress / track.length;
  const lapNow = Math.min(scenario.target_laps, Math.floor(laps) + 1);
  const highlight = neuron !== null && neuron.layer === 0 && neuron.index < 5 ? [neuron.index] : [];
  const looking = world.lookStep !== null;
  const body = bodyOf(scenario, car);
  const sub =
    info === null
      ? scenario.policy === "keyboard"
        ? "you drive: arrow keys or W A S D"
        : `${scenario.n_cars} car${scenario.n_cars === 1 ? "" : "s"} · ${scenario.track}`
      : isGhost
        ? `not part of generation ${info.unit}`
        : `rank ${info.ranks[car] + 1} of ${info.size} · ${role}`;

  return (
    <div className="telemetry" data-testid="telemetry" data-past={looking}>
      <div className="tele-group tele-id">
        <span className="tele-name">
          {!isGhost && <i className="tele-swatch" style={{ background: bodyCss(body) }} />}
          {carName(scenario, car)}
        </span>
        <span className="tele-sub">{sub}</span>
        <span data-testid="hud-status" data-status={status.key}>
          <Chip kind={status.kind}>{status.label}</Chip>
        </span>
      </div>

      <div className="tele-group tele-speed">
        <span className="eyebrow">Speed</span>
        <span className="tele-big num" data-testid="hud-speed">
          {(pose.v * 3.6).toFixed(0)}
          <small>km/h</small>
        </span>
        <span className="tele-sub num">{pose.v.toFixed(1)} m/s</span>
      </div>

      <div className="tele-group tele-actions">
        <div className="tele-row">
          <span className="eyebrow">Steer</span>
          <Gauge value={pose.steer} left="L" right="R" />
          <span className="num tele-val" data-testid="hud-steer">
            {num(pose.steer)}
          </span>
          <span className="tele-word">{describeSteer(pose.steer)}</span>
        </div>
        <div className="tele-row">
          <span className="eyebrow">Throttle</span>
          <Gauge value={pose.throttle} left="B" right="A" />
          <span className="num tele-val" data-testid="hud-throttle">
            {num(pose.throttle)}
          </span>
          <span className="tele-word">{describeThrottle(pose.throttle)}</span>
        </div>
      </div>

      <div className="tele-group tele-progress">
        <span className="eyebrow">
          Lap {lapNow} of {scenario.target_laps}
        </span>
        <div className="lapbar" aria-hidden="true">
          {Array.from({ length: scenario.target_laps }, (_, lap) => (
            <span key={lap} className="lapbar-lap">
              <span className="lapbar-fill" style={{ width: `${Math.max(0, Math.min(1, laps - lap)) * 100}%` }} />
            </span>
          ))}
        </div>
        <span className="tele-sub num" data-testid="hud-progress">
          {pose.progress.toFixed(0)} m · {laps.toFixed(2)} laps
        </span>
      </div>

      {shown.rays.length > 0 && (
        <div className="tele-group tele-rays">
          <span className="eyebrow">Rays (m)</span>
          <RayFan distances={shown.rays} angles={scenario.ray_angles} range={scenario.ray_range} highlight={highlight} />
          <span className="tele-raynums num">
            {shown.rays.map((d, k) => (
              <span key={k} data-dim={highlight.length > 0 && !highlight.includes(k)} data-testid={`ray-${k}`}>
                {d >= scenario.ray_range - 1e-6 ? `${scenario.ray_range.toFixed(0)}+` : d.toFixed(1)}
              </span>
            ))}
          </span>
        </div>
      )}

      {info !== null && (
        <div className="tele-group tele-fit">
          <span className="eyebrow">Fitness F</span>
          <span className="tele-mid num">{(isGhost ? info.ghost_fitness : info.fitness[car])?.toFixed(3) ?? "–"}</span>
          <span className="tele-sub">
            {isGhost
              ? `in generation ${info.ghost_unit}`
              : `${info.n_starts} starts: ${info.finished[car]} finished, ${info.crashed[car]} crashed`}
          </span>
          <span className="tele-sub">
            this start: f ={" "}
            {world.endFitness?.[car] === undefined ? "after the race" : <b className="num">{world.endFitness[car].toFixed(3)}</b>}
          </span>
        </div>
      )}

      <div className="tele-group tele-clock">
        <span className="eyebrow">{looking ? "Looking back" : "Step"}</span>
        <span className="tele-mid num" data-testid="hud-step">
          {shown.step}
        </span>
        <span className="tele-sub num">{(shown.step * scenario.dt).toFixed(2)} s</span>
      </div>
    </div>
  );
}
