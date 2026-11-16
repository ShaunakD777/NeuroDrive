import type { ConnectionStatus } from "../api/ws";

const LABELS: Record<ConnectionStatus, string> = {
  connecting: "Connecting",
  connected: "Connected",
  disconnected: "Disconnected",
};

const HINTS: Record<ConnectionStatus, string> = {
  connecting: "Connecting to the Python server…",
  connected: "Talking to the Python server (uv run python -m neurodrive serve)",
  disconnected: "No Python server on port 8000; retrying",
};

/** A small pill showing whether the browser is talking to the Python server. */
export function ConnectionBadge({ status }: { status: ConnectionStatus }) {
  return (
    <span
      className="conn"
      data-testid="connection-badge"
      data-status={status}
      title={HINTS[status]}
    >
      <span className="conn-dot" aria-hidden="true" />
      <span className="conn-text">{LABELS[status]}</span>
    </span>
  );
}
