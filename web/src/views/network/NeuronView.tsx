import type { ReactNode } from "react";

import type { NeuronReadout } from "../../api/generated";
import { useLive } from "../../state/live";
import { useSelection } from "../../state/selection";
import { formatRayAngle } from "../environment/status";
import {
  type Box,
  curveBounds,
  curvePoints,
  describeSteer,
  dialAngle,
  offChart,
  plotX,
  plotY,
  RANGE_TEXT,
} from "./curve";
import { ParamSlider } from "./ParamSlider";

const ACTIVATIONS = ["tanh", "sigmoid", "relu", "linear"] as const;
const SIGN = (x: number) => (x < 0 ? "−" : "+");
const signed = (x: number, digits = 2) => `${SIGN(x)}${Math.abs(x).toFixed(digits)}`;
const plain = (x: number, digits = 2) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(digits);

/** The steering dial: an arrow that turns right (clockwise) or left with the neuron's output. */
function SteerDial({ output, x, y }: { output: number; x: number; y: number }) {
  return (
    <g data-testid="steer-dial" transform={`translate(${x} ${y})`}>
      <circle r={36} className="dial-ring" />
      <g transform={`rotate(${dialAngle(output)})`} className="dial-arrow">
        <line x1={0} y1={22} x2={0} y2={-20} strokeWidth={4} />
        <polygon points="0,-30 -9,-14 9,-14" />
      </g>
    </g>
  );
}

/** The wiring diagram: two inputs, weighted connections, the neuron, and the steering dial. */
function Diagram({ n, selected, onSelect }: { n: NeuronReadout; selected: boolean; onSelect: () => void }) {
  const inputs = [
    { y: 55, x: n.inputs[0], w: n.weights[0], label: `Left ray ${formatRayAngle(Math.PI / 3)}` },
    { y: 165, x: n.inputs[1], w: n.weights[1], label: `Right ray ${formatRayAngle(-Math.PI / 3)}` },
  ];
  return (
    <svg viewBox="0 0 540 220" className="neuron-diagram" role="img" aria-label="The neuron and its inputs">
      {inputs.map((input, i) => (
        <g key={i}>
          <line
            x1={92}
            y1={input.y}
            x2={290}
            y2={110}
            className={input.w < 0 ? "wire wire-neg" : "wire wire-pos"}
            strokeWidth={1 + Math.min(10, Math.abs(input.w) * 0.5)}
          />
          <text x={190} y={(input.y + 110) / 2 + (i === 0 ? -8 : 16)} className="wire-label">
            w = {plain(input.w, 1)}
          </text>
          <circle cx={70} cy={input.y} r={22} className="node" />
          <circle
            cx={70}
            cy={input.y}
            r={22}
            className="node-fill"
            fillOpacity={Math.min(1, Math.max(0.05, input.x))}
          />
          <text x={70} y={input.y + 4} textAnchor="middle" className="node-text">
            {input.x.toFixed(2)}
          </text>
          <text x={70} y={input.y - 30} textAnchor="middle" className="node-caption">
            {input.label}
          </text>
        </g>
      ))}

      <g
        data-testid="neuron-node"
        data-selected={selected}
        className={selected ? "neuron-node neuron-node-selected" : "neuron-node"}
        onClick={onSelect}
        role="button"
        tabIndex={0}
        aria-pressed={selected}
        aria-label="The neuron (click to highlight the rays it reads)"
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onSelect();
          }
        }}
      >
        <circle cx={330} cy={110} r={44} className="node neuron-body" />
        <text x={330} y={106} textAnchor="middle" className="node-text neuron-sigma">
          {"Σ"}
        </text>
        <text x={330} y={128} textAnchor="middle" className="node-caption">
          {n.activation}
        </text>
      </g>
      <text x={330} y={176} textAnchor="middle" className="node-caption">
        z = {plain(n.z)}
      </text>

      <line x1={374} y1={110} x2={425} y2={110} className="wire wire-out" strokeWidth={3} />
      <SteerDial output={n.output} x={470} y={110} />
      <text x={470} y={166} textAnchor="middle" className="node-caption">
        steer {plain(n.output)}
      </text>
    </svg>
  );
}

/** The activation curve with a dot at the neuron's current (z, output). */
function CurveChart({ n }: { n: NeuronReadout }) {
  const activations = useLive((s) => s.activations);
  const box: Box = { left: 34, top: 12, width: 250, height: 130 };
  const values = activations?.curves[n.activation];
  if (!activations || !values) {
    return <p className="hint">The activation curve is loading…</p>;
  }
  const bounds = curveBounds(values);
  const zeroY = plotY(0, bounds, box);
  const dotX = plotX(n.z, box);
  const dotY = plotY(n.output, bounds, box);
  return (
    <svg viewBox="0 0 300 175" className="curve-chart" role="img" aria-label={`The ${n.activation} curve`}>
      <rect x={box.left} y={box.top} width={box.width} height={box.height} className="chart-frame" />
      <line x1={box.left} x2={box.left + box.width} y1={zeroY} y2={zeroY} className="chart-axis" />
      <line x1={plotX(0, box)} x2={plotX(0, box)} y1={box.top} y2={box.top + box.height} className="chart-axis" />
      <polyline points={curvePoints(activations.z, values, bounds, box)} className="chart-curve" fill="none" />
      <line x1={dotX} x2={dotX} y1={dotY} y2={zeroY} className="chart-guide" />
      <line x1={plotX(-5, box)} x2={dotX} y1={dotY} y2={dotY} className="chart-guide" />
      <circle cx={dotX} cy={dotY} r={6} className="chart-dot" data-testid="curve-dot" />
      <text x={box.left} y={box.top + box.height + 14} className="chart-label">
        z = −5
      </text>
      <text x={box.left + box.width} y={box.top + box.height + 14} textAnchor="end" className="chart-label">
        z = +5
      </text>
      <text x={box.left - 4} y={box.top + 8} textAnchor="end" className="chart-label">
        {bounds.max.toFixed(1)}
      </text>
      <text x={box.left - 4} y={box.top + box.height} textAnchor="end" className="chart-label">
        {bounds.min.toFixed(1)}
      </text>
      <text x={box.left + box.width / 2} y={171} textAnchor="middle" className="chart-label">
        {offChart(n.z) ? "z is beyond the chart" : "weighted sum z"}
      </text>
    </svg>
  );
}

/**
 * The Neuron view: one neuron driving the car. It shows the two ray readings it reads, the weights
 * on their connections (sliders), the weighted sum, the activation curve with a dot at the current
 * value, and an arrow for the steering command. Every number comes from the Python server (the
 * neuron readout in each frame); dragging a slider sends `set_param`. Clicking the neuron selects
 * it, which highlights the two rays it reads in the Environment view.
 */
export function NeuronView({
  compact = false,
  headerExtra,
}: {
  compact?: boolean;
  headerExtra?: ReactNode;
}) {
  const scenario = useLive((s) => s.scenario);
  const frame = useLive((s) => s.frame);
  const send = useLive((s) => s.send);
  const loadScenario = useLive((s) => s.loadScenario);
  const connected = useLive((s) => s.status === "connected");
  const neuronSelected = useSelection((s) => s.neuron !== null);

  const n = frame?.selected?.neuron ?? null;
  const car = frame?.selected?.index ?? 0;
  const toggleNeuron = () =>
    useSelection.getState().select(
      neuronSelected ? { neuron: null, output: null } : { neuron: { layer: 0, index: 0 }, output: "steer" },
    );

  const title = compact ? "Network (compact)" : "Network";
  let body: ReactNode;

  if (scenario === null || frame === null) {
    body = <p className="hint">{connected ? "Loading the scenario…" : "Waiting for the server…"}</p>;
  } else if (n === null) {
    body = (
      <div className="neuron-empty" data-testid="neuron-empty">
        <p>
          This scenario is driven by the {scenario.policy} driver, which has no neuron to show.
          Choose the neuron driver to see and change it.
        </p>
        <button type="button" data-testid="use-neuron" onClick={() => loadScenario({ policy: "neuron" })}>
          Use the neuron driver
        </button>
      </div>
    );
  } else if (compact) {
    body = (
      <div className="neuron-compact" data-testid="neuron-compact">
        <svg viewBox="0 0 100 90" className="neuron-compact-dial">
          <SteerDial output={n.output} x={50} y={45} />
        </svg>
        <div>
          <div>
            z = <span data-testid="neuron-z">{plain(n.z)}</span>
          </div>
          <div>
            {n.activation}(z) = <span data-testid="neuron-output">{plain(n.output)}</span>
          </div>
          <div className="hint">{describeSteer(n.output)}</div>
        </div>
      </div>
    );
  } else {
    body = (
      <>
        <p className="neuron-intro">
          Neuron of car {car + 1}. It reads the far-left and far-right rays; steer = f(w
          <sub>L</sub>·x<sub>L</sub> + w<sub>R</sub>·x<sub>R</sub> + bias). Positive steers right.
        </p>
        <Diagram n={n} selected={neuronSelected} onSelect={toggleNeuron} />
        <div className="neuron-grid">
          <div className="neuron-params">
            <ParamSlider label="w left" name="w_left" value={n.weights[0]} min={-20} max={20} step={0.1} />
            <div className="param-note" data-testid="neuron-line-left">
              x = <span data-testid="neuron-input-0">{n.inputs[0].toFixed(2)}</span> → w·x ={" "}
              <span data-testid="neuron-contribution-0">{signed(n.contributions[0])}</span>
            </div>
            <ParamSlider label="w right" name="w_right" value={n.weights[1]} min={-20} max={20} step={0.1} />
            <div className="param-note" data-testid="neuron-line-right">
              x = <span data-testid="neuron-input-1">{n.inputs[1].toFixed(2)}</span> → w·x ={" "}
              <span data-testid="neuron-contribution-1">{signed(n.contributions[1])}</span>
            </div>
            <ParamSlider label="bias" name="bias" value={n.bias} min={-2} max={2} step={0.01} />
            <div className="param-sum" data-testid="neuron-sum">
              z = {plain(n.contributions[0])} {signed(n.contributions[1])} {signed(n.bias)} ={" "}
              <strong data-testid="neuron-z">{plain(n.z)}</strong>
            </div>
            <label className="param">
              <span className="param-label">activation</span>
              <select
                data-testid="select-activation"
                value={n.activation}
                onChange={(e) => {
                  send({ type: "set_param", name: "activation", value: e.target.value });
                  e.target.blur(); // so the arrow keys drive the car instead of changing this menu
                }}
              >
                {ACTIVATIONS.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </label>
            <div className="param-sum">
              {n.activation}(z) = <strong data-testid="neuron-output">{plain(n.output)}</strong>
              {" → "}
              <span data-testid="neuron-steer-label">{describeSteer(n.output)}</span>
            </div>
            <div className="hint" data-testid="neuron-range">
              {n.activation} can output: {RANGE_TEXT[n.activation]}
            </div>
            <ParamSlider label="throttle" name="throttle" value={n.throttle} min={0} max={1} step={0.01} />
          </div>
          <CurveChart n={n} />
        </div>
      </>
    );
  }

  return (
    <section
      className={compact ? "view view-compact neuron-view" : "view neuron-view"}
      data-testid={compact ? "neuron-view-compact" : "neuron-view"}
    >
      <div className="view-header">
        <h2>{title}</h2>
        {headerExtra}
      </div>
      {body}
    </section>
  );
}
