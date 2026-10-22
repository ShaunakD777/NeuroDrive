import { useLive } from "../../state/live";
import { useMoment, usePlayback } from "../../state/playback";
import { useSelection } from "../../state/selection";
import { describeOutput } from "../network/inspect";
import { num } from "../network/networkLayout";
import { agoText } from "./trailLayout";

/**
 * The decision-trail timeline (task 2.11): a slider over the steps the server remembers for the
 * selected car, a readout of the step being looked at, "Back to now", and what the trail's colours
 * mean. Moving the slider (paused only) asks the server for that moment (`inspect_step`); the
 * Network view and the inspector then show the network as it was at that step.
 */
export function TrailPanel() {
  const frame = useLive((s) => s.frame);
  const trail = useLive((s) => s.trail);
  const dt = useLive((s) => s.scenario?.dt ?? 1 / 30);
  const seconds = useLive((s) => s.scenario?.trail_seconds ?? 60);
  const output = useSelection((s) => s.output);
  const viewStep = usePlayback((s) => s.viewStep);
  const view = usePlayback((s) => s.view);
  const backToLive = usePlayback((s) => s.backToLive);
  const moment = useMoment();

  const car = frame?.selected?.index ?? null;
  if (frame === null || car === null || trail === null || trail.car !== car || trail.steps.length === 0) {
    return null;
  }
  const first = trail.steps[0];
  const last = Math.min(frame.step, trail.steps[trail.steps.length - 1] + 1); // the end means "now"
  const value = viewStep ?? last;

  let readout: React.ReactNode;
  if (frame.playing) {
    readout = <>Pause to look back along the trail.</>;
  } else if (viewStep === null) {
    readout = (
      <>
        Showing now (step {frame.step}). Drag back, or click a point on the trail, to look at an
        earlier decision.
      </>
    );
  } else if (moment === null) {
    readout = <>Loading step {viewStep}…</>;
  } else {
    readout = (
      <>
        <strong>
          Step {moment.step} ({agoText(frame.step - moment.step, dt)}):
        </strong>{" "}
        steer {num(moment.action[0])} ({describeOutput(0, moment.action[0])}), throttle{" "}
        {num(moment.action[1])} ({describeOutput(1, moment.action[1])}). The Network view shows
        the network at this step.
      </>
    );
  }

  return (
    <div className="trail-panel" data-testid="trail-panel">
      <div className="trail-row">
        <label className="trail-slider">
          <span>Look back</span>
          <input
            type="range"
            min={first}
            max={last}
            step={1}
            value={value}
            disabled={frame.playing || last <= first}
            data-testid="trail-slider"
            aria-label="Step to look at"
            onChange={(e) => view(Number(e.target.value))}
          />
        </label>
        <button
          type="button"
          data-testid="trail-live"
          disabled={viewStep === null}
          onClick={() => backToLive()}
        >
          Back to now
        </button>
      </div>
      <p className="trail-readout" data-testid="trail-readout">
        {readout}
      </p>
      <p className="trail-legend hint" data-testid="trail-legend">
        {output === null ? (
          <>
            Trail: where car {car + 1} has been in the last {seconds} s. Click Steer or Throttle on the
            Network page to colour it by that decision.
          </>
        ) : (
          <>
            Trail coloured by {output}: <span className="ray-legend-pos">blue</span> ={" "}
            {output === "steer" ? "steering right" : "accelerating"},{" "}
            <span className="ray-legend-neg">orange</span> ={" "}
            {output === "steer" ? "steering left" : "braking"}; thicker = stronger.
          </>
        )}
      </p>
    </div>
  );
}
