import { Chip, type ChipKind } from "../../components/ui/controls";
import { useLive } from "../../state/live";
import { useMoment } from "../../state/playback";
import { raysFor, useSelection } from "../../state/selection";
import { describeOutput } from "../network/inspect";
import { bodyCss, bodyOf } from "./cars";
import { carRole, roleLabel } from "./population";
import { describeStatus, formatRayAngle, formatSpeed, type StatusText } from "./status";

// The read-outs over the track (task E3, recommendation E11): a telemetry strip docked along the
// bottom of the stage instead of the tall HUD table that covered the top left of the track, a
// one-line read-out in the compact drawer, and the status banner and server error as stage panels.
// Every number is the server's, for the selected car at the shown moment: now, or the earlier
// step being looked at (task 2.11), as the server recomputed it. The test hooks and their text
// formats are the HUD's (`hud`, `hud-speed`, `hud-status`, `hud-steer`, …), unchanged.

const CHIP: Record<StatusText["kind"], ChipKind> = { running: "run", good: "ok", bad: "bad", warn: "warn" };

/** A signed gauge on the dark strip: the bar grows from the middle tick, blue right, orange left. */
function Gauge({ value, left, right, testId }: { value: number; left: string; right: string; testId: string }) {
  const v = Math.max(-1, Math.min(1, value));
  return (
    <span className="gauge" aria-hidden="true" data-testid={testId}>
      <span className="gauge-end">{left}</span>
      <span className="gauge-track">
        <span
          className={v < 0 ? "gauge-fill gauge-neg" : "gauge-fill gauge-pos"}
          style={{ left: v >= 0 ? "50%" : `${50 + v * 50}%`, width: `${Math.abs(v) * 50}%` }}
        />
      </span>
      <span className="gauge-end">{right}</span>
    </span>
  );
}

/**
 * The five rays as a small fan seen from above the car: screen up is the car's forward, +angle to
 * the left; length = distance / range; colour by how close the wall is (the canvas's rule).
 */
function RayFan({ distances, angles, range, highlight }: { distances: number[]; angles: number[]; range: number; highlight: number[] }) {
  const R = 46;
  const cx = 60;
  const cy = 54;
  return (
    <svg className="rayfan" viewBox="0 0 120 60" aria-hidden="true">
      <path d={`M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${cx + R} ${cy}`} className="rayfan-range" />
      <path d={`M ${cx - R / 2} ${cy} A ${R / 2} ${R / 2} 0 0 1 ${cx + R / 2} ${cy}`} className="rayfan-range" />
      {distances.map((d, k) => {
        const a = Math.PI / 2 + (angles[k] ?? 0);
        const f = Math.min(1, d / range);
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

/** The selected car's state at the shown moment, or null before the first frame. */
function useShown() {
  const frame = useLive((s) => s.frame);
  const scenario = useLive((s) => s.scenario);
  const geometry = useLive((s) => s.geometry);
  const moment = useMoment();
  if (frame === null || scenario === null) return null;
  const i = frame.selected?.index ?? 0;
  const code = moment?.status ?? frame.cars.status[i] ?? 0;
  const progress = moment?.progress ?? frame.cars.progress[i] ?? 0;
  const [steer, throttle] = moment?.action ?? frame.cars.action[i] ?? [0, 0];
  return {
    frame,
    scenario,
    geometry,
    moment,
    i,
    code,
    text: describeStatus(code),
    speed: moment?.v ?? frame.cars.v[i] ?? 0,
    steer,
    throttle,
    progress,
    step: moment?.step ?? frame.step,
    time: moment?.time ?? frame.time,
    rays: moment?.rays ?? frame.selected?.rays ?? [],
  };
}

/** The telemetry strip of the full view (or the one-line read-out of the compact drawer). */
export function Telemetry({ compact }: { compact: boolean }) {
  const shown = useShown();
  const status = useLive((s) => s.status);
  const episodeEnd = useLive((s) => s.episodeEnd);
  const neuron = useSelection((s) => s.neuron);
  const connection = useSelection((s) => s.connection);
  const selectedCar = useLive((s) => s.frame?.selected);

  if (shown === null) {
    return (
      <div className={compact ? "hud hud-compact hud-empty" : "telemetry hud hud-empty"} data-testid="hud">
        {status === "connected" ? "Loading the scenario…" : "Waiting for the server…"}
      </div>
    );
  }
  const { frame, scenario, geometry, moment, i, text, speed, steer, throttle, progress, step, time, rays } = shown;
  const chip = (
    <span className="hud-chip" data-testid="hud-status" data-kind={text.kind}>
      <Chip kind={CHIP[text.kind]}>{text.label}</Chip>
    </span>
  );

  if (compact) {
    return (
      <div className="hud hud-compact" data-testid="hud" data-past={moment !== null}>
        <span className="num" data-testid="hud-speed" data-value={speed}>
          {formatSpeed(speed)}
        </span>
        {chip}
      </div>
    );
  }

  const race = scenario.population ?? null;
  const role = carRole(race, i);
  const raceF = episodeEnd?.fitness?.[i] ?? null;
  const laps = geometry === null ? null : progress / geometry.length;
  const targetLaps = Math.max(1, scenario.target_laps);
  const lapNow = laps === null ? 1 : Math.min(targetLaps, Math.floor(laps) + 1);
  const highlight = raysFor({ neuron, connection }, selectedCar);
  const count = frame.cars.x.length;
  const name = race !== null && role?.kind === "ghost" ? "Ghost" : scenario.policy === "keyboard" && count === 1 ? "You" : `Car ${i + 1}`;
  // In a race, the car's place and role always show here; the fitness group may give way on a
  // narrow stage (environment.css), and the Standings tab lists every car's F.
  const raceLine =
    race === null || role === null
      ? null
      : role.kind === "ghost"
        ? `generation ${race.ghost_unit}'s champion`
        : [`rank ${(role.place ?? 0) + 1} of ${race.size}`, role.elite ? "elite" : null, role.champion ? "champion" : null]
            .filter((part) => part !== null)
            .join(" · ");
  const sub =
    raceLine !== null
      ? raceLine
      : race !== null
        ? null
        : scenario.policy === "keyboard"
        ? "arrow keys or W A S D"
        : `${count} car${count === 1 ? "" : "s"} · ${scenario.track}`;

  return (
    <div className="telemetry hud" data-testid="hud" data-past={moment !== null}>
      <div className="tele-group tele-id">
        <span className="tele-name">
          {role?.kind !== "ghost" && <i className="tele-swatch" style={{ background: bodyCss(bodyOf(race, count, i, i)) }} />}
          {name}
        </span>
        {sub !== null && <span className="tele-sub">{sub}</span>}
        {chip}
      </div>

      <div className="tele-group tele-speed">
        <span className="eyebrow">Speed</span>
        <span className="tele-mid num" data-testid="hud-speed" data-value={speed}>
          {formatSpeed(speed)}
        </span>
      </div>

      <div className="tele-group tele-actions">
        <div className="tele-row">
          <span className="eyebrow">Steer</span>
          <Gauge value={steer} left="L" right="R" testId="hud-steer-bar" />
          <span className="num tele-val" data-testid="hud-steer">
            {steer.toFixed(2)}
          </span>
          <span className="tele-word">{describeOutput(0, steer)}</span>
        </div>
        <div className="tele-row">
          <span className="eyebrow">Throttle</span>
          <Gauge value={throttle} left="B" right="A" testId="hud-throttle-bar" />
          <span className="num tele-val" data-testid="hud-throttle">
            {throttle.toFixed(2)}
          </span>
          <span className="tele-word">{describeOutput(1, throttle)}</span>
        </div>
      </div>

      <div className="tele-group tele-progress">
        <span className="eyebrow">
          Lap {lapNow} of {targetLaps}
        </span>
        {laps !== null && (
          <span className="lapbar" aria-hidden="true">
            {Array.from({ length: targetLaps }, (_, lap) => (
              <span key={lap} className="lapbar-lap">
                <span className="lapbar-fill" style={{ width: `${Math.max(0, Math.min(1, laps - lap)) * 100}%` }} />
              </span>
            ))}
          </span>
        )}
        <span className="tele-sub num">
          <span data-testid="hud-progress">{progress.toFixed(1)} m</span>
          {laps !== null && <> · {laps.toFixed(2)} laps</>}
        </span>
      </div>

      {rays.length > 0 && (
        <div className="tele-group tele-rays" data-testid="hud-rays">
          <span className="eyebrow">Rays (m)</span>
          <RayFan distances={rays} angles={scenario.ray_angles} range={scenario.ray_range} highlight={highlight} />
          <span className="tele-raynums num">
            {rays.map((distance, k) => (
              <span
                key={k}
                data-testid={`ray-${k}`}
                data-value={distance}
                data-dim={highlight.length > 0 && !highlight.includes(k)}
                title={`Ray ${formatRayAngle(scenario.ray_angles[k])}: ${distance.toFixed(1)} m`}
              >
                {distance >= scenario.ray_range - 1e-6 ? `${scenario.ray_range.toFixed(0)}+` : distance.toFixed(1)}
              </span>
            ))}
          </span>
        </div>
      )}

      {race !== null && role !== null && (
        <div className="tele-group tele-fit" data-testid="hud-race" data-kind={role.kind} data-place={role.place ?? ""}>
          <span className="tele-sub" data-testid="hud-race-role">
            {roleLabel(race, role)}
          </span>
          <span className="tele-row">
            <span className="eyebrow">Fitness F</span>
            <span className="tele-mid num" data-testid="hud-race-fitness">
              {role.fitness === null ? "–" : role.fitness.toFixed(3)}
            </span>
          </span>
          <span className="tele-sub">
            {role.kind === "ghost"
              ? `in generation ${race.ghost_unit}`
              : `mean of ${race.n_starts} starts: finished ${race.finished[i]}, crashed ${race.crashed[i]}`}
          </span>
          <span className="tele-sub">
            f this start:{" "}
            <b className="num" data-testid="hud-race-start-fitness">
              {raceF === null ? "after the race" : raceF.toFixed(3)}
            </b>
          </span>
        </div>
      )}

      <div className="tele-group tele-clock">
        <span className="eyebrow">{moment === null ? "Step" : "Looking back"}</span>
        <span className="tele-mid num" data-testid="hud-step">
          {step}
        </span>
        <span className="tele-sub num">{time.toFixed(2)} s</span>
        {moment !== null && (
          <span className="tele-sub tele-past" data-testid="hud-past">
            Looking back: now is step {frame.step}
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Why the selected car stopped, as a stage panel (the HUD's status banner, same words), and the
 * server's error. In the compact drawer the banner has no "Press Reset".
 */
export function StageBanners({ compact }: { compact: boolean }) {
  const shown = useShown();
  const error = useLive((s) => s.error);
  const text = shown?.text ?? null;
  return (
    <>
      {text !== null && text.reason !== null && (
        <div className="banner stage-banner" data-kind={text.kind} role="status" data-testid="status-banner">
          <Chip kind={CHIP[text.kind]}>{text.label}</Chip>
          <span>
            {text.reason}
            {!compact && " Press Reset to try again."}
          </span>
        </div>
      )}
      {error !== null && !compact && (
        <div className="banner stage-banner" data-kind="warn" role="alert" data-testid="server-error">
          Server: {error}
        </div>
      )}
    </>
  );
}
