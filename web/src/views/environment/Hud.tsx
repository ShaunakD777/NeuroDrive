import { useLive } from "../../state/live";
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
 * status.
 */
export function Hud({ compact }: { compact: boolean }) {
  const frame = useLive((s) => s.frame);
  const scenario = useLive((s) => s.scenario);
  const geometry = useLive((s) => s.geometry);
  const error = useLive((s) => s.error);
  const status = useLive((s) => s.status);

  if (frame === null || scenario === null) {
    return (
      <div className="hud hud-empty" data-testid="hud">
        {status === "connected" ? "Loading the scenario…" : "Waiting for the server…"}
      </div>
    );
  }

  const i = frame.selected?.index ?? 0;
  const code = frame.cars.status[i] ?? 0;
  const text = describeStatus(code);
  const speed = frame.cars.v[i] ?? 0;
  const [steer, throttle] = frame.cars.action[i] ?? [0, 0];
  const progress = frame.cars.progress[i] ?? 0;
  const laps = geometry === null ? null : progress / geometry.length;

  return (
    <>
      <div className="hud" data-testid="hud">
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
              <span data-testid="hud-step">{frame.step}</span>
              <span className="hud-muted">{frame.time.toFixed(2)} s</span>
            </div>
            <div className="hud-row">
              <span className="hud-label">Progress</span>
              <span data-testid="hud-progress">{progress.toFixed(1)} m</span>
              {laps !== null && <span className="hud-muted">{laps.toFixed(2)} laps</span>}
            </div>
            <div className="hud-rays" data-testid="hud-rays">
              {(frame.selected?.rays ?? []).map((distance, k) => (
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
