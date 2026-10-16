// Words for the status codes the server sends in every frame (docs/ARCHITECTURE.md §7).

/** The server's status code for one car: 0 running, 1 crashed, 2 finished, 3 stalled, 4 timeout. */
export type StatusCode = number;

export interface StatusText {
  /** A short word for the HUD chip. */
  label: string;
  /** A sentence saying why the episode ended, or null while the car is still running. */
  reason: string | null;
  /** Which colour the chip uses. */
  kind: "running" | "bad" | "good" | "warn";
}

export function describeStatus(code: StatusCode): StatusText {
  switch (code) {
    case 0:
      return { label: "Running", reason: null, kind: "running" };
    case 1:
      return {
        label: "Crashed",
        reason: "Crashed: the car left the road.",
        kind: "bad",
      };
    case 2:
      return {
        label: "Finished",
        reason: "Finished: the car completed the target laps.",
        kind: "good",
      };
    case 3:
      return {
        label: "Stalled",
        reason: "Stalled: the car stopped making progress.",
        kind: "warn",
      };
    case 4:
      return { label: "Time limit", reason: "Time limit reached.", kind: "warn" };
    default:
      return { label: `Status ${code}`, reason: `Episode ended (status ${code}).`, kind: "warn" };
  }
}

/** Speed in metres per second as "12.3 m/s (44 km/h)". */
export function formatSpeed(metresPerSecond: number): string {
  return `${metresPerSecond.toFixed(1)} m/s (${Math.round(metresPerSecond * 3.6)} km/h)`;
}

/** A ray's angle (radians, relative to the heading, left positive) as "+60°", "0°", "−30°". */
export function formatRayAngle(radians: number): string {
  const degrees = Math.round((radians * 180) / Math.PI);
  if (degrees === 0) return "0°";
  return `${degrees > 0 ? "+" : "−"}${Math.abs(degrees)}°`;
}
