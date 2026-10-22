import { useLive } from "../../state/live";
import { useMoment } from "../../state/playback";
import { describeStatus, formatRayAngle, formatSpeed } from "./status";

/** A horizontal bar showing a value between -1 and +1, filled from the middle. */
function SignedBar({ value, testId }: { value: number; testId: string }) {
  const half = Math.min(1, Math.max(-1, value)) * 50;
  return (
    <span className="bar" aria-hidden="true" data-testid={testId}>
      <span
        className="bar-fill"
        style={{ left: half >= 0 ? "50%" : `${50 + half}%`, width: `${Math.abs(half)}%` }}
      />
    </span>
  );
}

/**
 * The heads-up display over the canvas: speed, the actions being applied, progress, the five ray
 * distances, and why the episode ended once it has. The compact version shows only speed and
 * status. While an earlier step is looked at (task 2.11) every number is that step's, as the
 * server recomputed it (`moment`), and the Step row says so; the action shown is the one the car
 * chose at that step (what the Network view shows there).
 */
export function Hud({ compact }: { compact: boolean }) {
  const frame = useLive((s) => s.frame);
  const scenario = useLive((s) => s.scenario);
  const geometry = useLive((s) => s.geometry);
  const error = useLive((s) => s.error);
  const status = useLive((s) => s.status);
  const moment = useMoment();

  if (frame === null || scenario === null) {
    return (
      <div className="hud hud-empty" data-testid="hud">
        {status === "connected" ? "Loading the scenario…" : "Waiting for the server…"}
      </div>
    );
  }

  const i = frame.selected?.index ?? 0;
  // Now, or the earlier step being looked at: every number comes from the same moment.
  const code = moment?.status ?? frame.cars.status[i] ?? 0;
  const text = describeStatus(code);
  const speed = moment?.v ?? frame.cars.v[i] ?? 0;
  const [steer, throttle] = moment?.action ?? frame.cars.action[i] ?? [0, 0];
  const progress = moment?.progress ?? frame.cars.progress[i] ?? 0;
  const step = moment?.step ?? frame.step;
  const time = moment?.time ?? frame.time;
  const rays = moment?.rays ?? frame.selected?.rays ?? [];
  const laps = geometry === null ? null : progress / geometry.length;

  return (
    <>
      <div className={moment === null ? "hud" : "hud hud-past"} data-testid="hud" data-past={moment !== null}>
        <div className="hud-row">
          <span className="hud-label">Speed</span>
          <span data-testid="hud-speed" data-value={speed}>
            {formatSpeed(speed)}
          </span>
        </div>
        <div className="hud-row">
          <span className="hud-label">Status</span>
          <span className="chip" data-kind={text.kind} data-testid="hud-status">
            {text.label}
          </span>
        </div>
        {!compact && (
          <>
            <div className="hud-row">
              <span className="hud-label">Steer</span>
              <SignedBar value={steer} testId="hud-steer-bar" />
              <span data-testid="hud-steer">{steer.toFixed(2)}</span>
            </div>
            <div className="hud-row">
              <span className="hud-label">Throttle</span>
              <SignedBar value={throttle} testId="hud-throttle-bar" />
              <span data-testid="hud-throttle">{throttle.toFixed(2)}</span>
            </div>
            <div className="hud-row">
              <span className="hud-label">Step</span>
              <span data-testid="hud-step">{step}</span>
              <span className="hud-muted">{time.toFixed(2)} s</span>
            </div>
            {moment !== null && (
              <div className="hud-row hud-past-note" data-testid="hud-past">
                Looking back: now is step {frame.step}
              </div>
            )}
            <div className="hud-row">
              <span className="hud-label">Progress</span>
              <span data-testid="hud-progress">{progress.toFixed(1)} m</span>
              {laps !== null && <span className="hud-muted">{laps.toFixed(2)} laps</span>}
            </div>
            <div className="hud-rays" data-testid="hud-rays">
              {rays.map((distance, k) => (
                <div className="hud-row" key={k}>
                  <span className="hud-label">Ray {formatRayAngle(scenario.ray_angles[k])}</span>
                  <span data-testid={`ray-${k}`} data-value={distance}>
                    {distance.toFixed(1)} m
                  </span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
      {text.reason !== null && (
        <div className="banner" data-kind={text.kind} role="status" data-testid="status-banner">
          {text.reason}
          {!compact && " Press Reset to try again."}
        </div>
      )}
      {error !== null && !compact && (
        <div className="banner" data-kind="warn" role="alert" data-testid="server-error">
          Server: {error}
        </div>
      )}
    </>
  );
}
