import type { ConnectionStatus } from "../api/ws";

const LABELS: Record<ConnectionStatus, string> = {
  connecting: "Connecting",
  connected: "Connected",
  disconnected: "Disconnected",
};

/** A small badge showing whether the browser is talking to the Python server. */
export function ConnectionBadge({ status }: { status: ConnectionStatus }) {
  return (
    <span className="badge" data-testid="connection-badge" data-status={status}>
      <span className="badge-dot" aria-hidden="true" />
      {LABELS[status]}
    </span>
  );
}
