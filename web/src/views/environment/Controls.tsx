import { useLive } from "../../state/live";

const SPEEDS = [0.5, 1, 2, 4, 8];

/** Track choice, play / pause / step / reset, the speed factor and the camera. */
export function Controls({
  mode,
  onModeChange,
  onZoomIn,
  onZoomOut,
}: {
  mode: "follow" | "fit";
  onModeChange: (mode: "follow" | "fit") => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}) {
  const send = useLive((s) => s.send);
  const loadTrack = useLive((s) => s.loadTrack);
  const tracks = useLive((s) => s.tracks);
  const scenario = useLive((s) => s.scenario);
  const frame = useLive((s) => s.frame);
  const ended = useLive((s) => s.episodeEnd !== null);
  const connected = useLive((s) => s.status === "connected");

  const ready = connected && scenario !== null && frame !== null;
  const playing = frame?.playing ?? false;
  const speed = frame?.speed ?? 1;

  return (
    <div className="controls" role="toolbar" aria-label="Simulation controls">
      <label className="control">
        <span className="control-label">Track</span>
        <select
          data-testid="select-track"
          value={scenario?.track ?? ""}
          disabled={!connected || tracks.length === 0}
          onChange={(e) => {
            loadTrack(e.target.value);
            e.target.blur(); // so the arrow keys drive the car instead of changing this menu
          }}
        >
          {scenario === null && <option value="">…</option>}
          {tracks.map((t) => (
            <option key={t.name} value={t.name}>
              {t.name}
            </option>
          ))}
        </select>
      </label>

      <button
        type="button"
        data-testid="btn-play"
        disabled={!ready || ended}
        onClick={() => send({ type: playing ? "pause" : "play" })}
      >
        {playing ? "Pause" : "Play"}
      </button>
      <button
        type="button"
        data-testid="btn-step"
        disabled={!ready || ended || playing}
        onClick={() => send({ type: "step" })}
      >
        Step
      </button>
      <button
        type="button"
        data-testid="btn-reset"
        disabled={!ready}
        onClick={() => send({ type: "reset" })}
      >
        Reset
      </button>

      <label className="control">
        <span className="control-label">Speed</span>
        <select
          data-testid="select-speed"
          value={speed}
          disabled={!ready}
          onChange={(e) => {
            send({ type: "set_speed", speed: Number(e.target.value) });
            e.target.blur(); // so the arrow keys drive the car instead of changing this menu
          }}
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>

      <div className="segmented" role="group" aria-label="Camera">
        <button
          type="button"
          aria-pressed={mode === "follow"}
          data-testid="camera-follow"
          onClick={() => onModeChange("follow")}
        >
          Follow
        </button>
        <button
          type="button"
          aria-pressed={mode === "fit"}
          data-testid="camera-fit"
          onClick={() => onModeChange("fit")}
        >
          Whole track
        </button>
      </div>
      <button
        type="button"
        data-testid="zoom-out"
        aria-label="Zoom out"
        disabled={mode !== "follow"}
        onClick={onZoomOut}
      >
        −
      </button>
      <button
        type="button"
        data-testid="zoom-in"
        aria-label="Zoom in"
        disabled={mode !== "follow"}
        onClick={onZoomIn}
      >
        +
      </button>
    </div>
  );
}
