import { switchSource, useApp } from "../state/store";
import { Button } from "./controls";

/**
 * What a view shows before its source has given it anything to draw: a skeleton while the
 * recording loads or the server answers, or (live, no server) how to start one.
 */
export function SourceWaiting({ what, height }: { what: string; height?: number | string }) {
  const source = useApp((s) => s.source);
  const status = useApp((s) => s.liveStatus);
  if (source === "live" && status === "disconnected") {
    return (
      <div className="empty" data-testid="server-offline">
        <span className="empty-title">No Python server on port 8000</span>
        <span>
          Start it in the repository root with <code>uv run python -m neurodrive serve</code>. The prototype connects by itself as soon as it
          answers.
        </span>
        <div className="empty-actions">
          <Button icon="left" onClick={() => switchSource("recorded")}>
            Use the recorded race instead
          </Button>
        </div>
      </div>
    );
  }
  return (
    <div className="waiting" style={{ height: height ?? "100%" }} aria-busy="true">
      <div className="skeleton" style={{ position: "absolute", inset: 0 }} />
      <span className="waiting-text">{source === "live" ? `Waiting for the server: ${what}` : `Loading the recording: ${what}`}</span>
    </div>
  );
}
