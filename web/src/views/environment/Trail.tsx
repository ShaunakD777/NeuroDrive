import { useLive } from "../../state/live";
import { useMoment, usePlayback } from "../../state/playback";
import { useSelection } from "../../state/selection";
import { describeOutput } from "../network/inspect";
import { num } from "../network/networkLayout";
import { stripPaths, stripPeak, stripX } from "./stageLayout";
import { agoText } from "./trailLayout";

const W = 1000; // the strip's drawing width (SVG units; stretched to the box)
const H = 56;

/**
 * The look-back strip (task 2.11, restyled in task E3 as recommendation E15): the selected car's
 * steer and throttle over the steps the server remembers, drawn as filled areas, so a hard brake
 * or a steering reversal can be seen before it is looked at. A native range input lies over the
 * chart (`trail-slider`, paused only): dragging it asks the server for that moment
 * (`inspect_step`); the Network view and the inspector then show the network as it was. Below it,
 * the readout of the step being looked at, "Back to now", and what the trail's colours mean.
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
  const xs = trail.steps.map((s) => stripX(s, first, last, W));
  const steer = trail.action.map((a) => a[0] ?? 0);
  const throttle = trail.action.map((a) => a[1] ?? 0);
  const steerTop = stripPeak(steer);
  const throttleTop = stripPeak(throttle);
  const steerArea = stripPaths(steer, xs, 14, 12, steerTop);
  const throttleArea = stripPaths(throttle, xs, 42, 12, throttleTop);
  const cursor = stripX(value, first, last, W);

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
    <div className="trail-panel trailstrip" data-testid="trail-panel" data-disabled={frame.playing}>
      <div className="trailstrip-side">
        <span className="eyebrow">Look back</span>
        <span className="trailstrip-now num">
          {viewStep === null ? "now" : `−${((frame.step - value) * dt).toFixed(1)} s`}
          <small>step {value}</small>
        </span>
      </div>
      <div className="trailstrip-chart">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
          <line x1={0} x2={W} y1={14} y2={14} className="trailstrip-zero" />
          <line x1={0} x2={W} y1={42} y2={42} className="trailstrip-zero" />
          <g className={output === "throttle" ? "trailstrip-area dim" : "trailstrip-area"}>
            <path d={steerArea.pos} className="ts-pos" />
            <path d={steerArea.neg} className="ts-neg" />
          </g>
          <g className={output === "steer" ? "trailstrip-area dim" : "trailstrip-area"}>
            <path d={throttleArea.pos} className="ts-pos" />
            <path d={throttleArea.neg} className="ts-neg" />
          </g>
          <line x1={cursor} x2={cursor} y1={0} y2={H} className="trailstrip-cursor" vectorEffect="non-scaling-stroke" />
        </svg>
        <span className="trailstrip-label" style={{ top: 2 }}>
          steer ±{steerTop.toFixed(2)}
        </span>
        <span className="trailstrip-label" style={{ top: 30 }}>
          throttle ±{throttleTop.toFixed(2)}
        </span>
        <span className="trailstrip-scale num" aria-hidden="true">
          <span>−{((last - first) * dt).toFixed(0)} s</span>
          <span>now</span>
        </span>
        <input
          type="range"
          className="trailstrip-input"
          min={first}
          max={last}
          step={1}
          value={value}
          disabled={frame.playing || last <= first}
          data-testid="trail-slider"
          aria-label="Step to look at"
          onChange={(e) => view(Number(e.target.value))}
        />
      </div>
      <div className="trailstrip-side trailstrip-end">
        <button type="button" className="btn" data-testid="trail-live" disabled={viewStep === null} onClick={() => backToLive()}>
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
