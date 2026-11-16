import { Button } from "../../components/ui/controls";
import { useLive } from "../../state/live";

// Play, step, reset and the speed factor, floating at the top left of the stage (task E3,
// recommendation E12). Moved from the pre-redesign Controls.tsx with the same rules and test ids;
// the speed stays a menu (`select-speed`), styled for the dark bar.

const SPEEDS = [0.5, 1, 2, 4, 8];

export function Transport() {
  const send = useLive((s) => s.send);
  const ended = useLive((s) => s.episodeEnd !== null);
  const connected = useLive((s) => s.status === "connected");
  const ready = useLive((s) => s.status === "connected" && s.scenario !== null && s.frame !== null);
  const playing = useLive((s) => s.frame?.playing ?? false);
  const speed = useLive((s) => s.frame?.speed ?? 1);

  return (
    <div className="hudbar" role="toolbar" aria-label="Simulation">
      <Button
        variant="hud"
        icon={playing ? "pause" : "play"}
        disabled={!ready || ended}
        title={ended ? "The episode has ended: Reset to drive again" : playing ? "Pause the simulation" : "Run the simulation (the car only moves while it runs)"}
        onClick={() => send({ type: playing ? "pause" : "play" })}
        testId="btn-play"
      >
        {playing ? "Pause" : "Play"}
      </Button>
      <Button
        variant="hud"
        icon="step"
        disabled={!ready || ended || playing}
        title="One step forward"
        onClick={() => send({ type: "step" })}
        testId="btn-step"
      >
        Step
      </Button>
      <Button variant="hud" icon="reset" disabled={!ready} title="Back to the start" onClick={() => send({ type: "reset" })} testId="btn-reset">
        Reset
      </Button>
      <span className="hudbar-sep" aria-hidden="true" />
      <label className="hud-select" title="Simulation speed: how many times faster than real time">
        <span className="hud-select-label">Speed</span>
        <select
          data-testid="select-speed"
          value={speed}
          disabled={!ready || !connected}
          onChange={(e) => {
            send({ type: "set_speed", speed: Number(e.target.value) });
            e.target.blur();
          }}
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
