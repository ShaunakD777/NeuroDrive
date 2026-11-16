import type { NeuronReadout } from "@prod/api/generated";
import { useLive } from "@prod/state/live";
import { RANGE_TEXT } from "@prod/views/network/curve";

import { describeSteer, num, signed } from "../../state/store";
import { Panel, Segmented, ValueBar } from "../../ui/controls";
import { ActivationPlot } from "./Inspector";

/** One parameter: a slider and its value; moving it sends `set_param` (production: ParamSlider). */
function ParamRow({ label, name, value, min, max, step }: { label: string; name: "w_left" | "w_right" | "bias" | "throttle"; value: number; min: number; max: number; step: number }) {
  const send = useLive((s) => s.send);
  return (
    <label className="insp-row whatif-row">
      <span>{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => send({ type: "set_param", name, value: Number(e.target.value) })}
        data-testid={`param-${name}`}
      />
      <span className="num">{num(value)}</span>
    </label>
  );
}

/**
 * The neuron driver (Phase 1): one neuron reads two rays and steers; you set its weights. The
 * redesign of production's NeuronView on the shared components (PRODUCTION_MAPPING.md N17): the
 * receipt reads like the multilayer inspector's (two contributions plus the bias make z, the
 * activation turns z into the steer), and the sliders are the what-if rows' style. Every number
 * is Python's readout; the sliders only send `set_param`.
 */
export function NeuronPanel({ neuron, rays }: { neuron: NeuronReadout; rays: number[] }) {
  const send = useLive((s) => s.send);
  const scale = Math.max(1e-6, ...neuron.contributions.map(Math.abs), Math.abs(neuron.bias));
  const [left, right] = neuron.input_rays;
  return (
    <div className="net-body">
      <div className="net-main">
        <p className="net-intro">
          One neuron: it reads ray {left + 1} and ray {right + 1} (normalised distances), adds a bias, and its activation is the steer.
          Throttle is fixed. Drag the weights and watch the car change its mind.
        </p>
        <Panel title="The neuron now" eyebrow="One neuron, live" testId="neuron-panel" className="neuron-panel">
          <p className="insp-note">
            z = w<sub>left</sub> · x<sub>left</sub> + w<sub>right</sub> · x<sub>right</sub> + b, then steer = {neuron.activation}(z). Each row is one
            term of the sum.
          </p>
          <div className="insp-rows">
            {neuron.contributions.map((c, k) => (
              <div key={k} className="insp-row">
                <span>
                  Ray {neuron.input_rays[k] + 1} ({k === 0 ? "left" : "right"}) · <span className="num">{num(neuron.weights[k])} × {num(neuron.inputs[k], 3)}</span>
                </span>
                <ValueBar value={c / scale} />
                <span className="num">{signed(c, 3)}</span>
              </div>
            ))}
            <div className="insp-row insp-muted">
              <span>bias</span>
              <ValueBar value={neuron.bias / scale} />
              <span className="num">{signed(neuron.bias, 3)}</span>
            </div>
          </div>
          <div className="insp-sum">
            <div className="insp-eq" data-testid="neuron-sum">
              <span className="eyebrow">Weighted sum</span>
              <span className="num insp-eq-line">z = {signed(neuron.z, 3)}</span>
              <span className="eyebrow">Steer</span>
              <span className="num insp-eq-line">
                {neuron.activation}(z) = <b>{signed(neuron.output, 3)}</b> · {describeSteer(neuron.output)}
              </span>
            </div>
            <ActivationPlot z={neuron.z} a={neuron.output} name={neuron.activation} />
          </div>
          <p className="insp-note">
            Rays now (m): {rays.map((d) => d.toFixed(1)).join(" · ")}. {neuron.activation} can output {RANGE_TEXT[neuron.activation]}.
          </p>
        </Panel>
      </div>
      <aside className="inspector" data-testid="neuron-params">
        <section className="insp-section">
          <h4 className="insp-h">
            <span>Weights</span> what the neuron listens to
          </h4>
          <div className="insp-rows">
            <ParamRow label="w left" name="w_left" value={neuron.weights[0]} min={-20} max={20} step={0.1} />
            <ParamRow label="w right" name="w_right" value={neuron.weights[1]} min={-20} max={20} step={0.1} />
            <ParamRow label="bias" name="bias" value={neuron.bias} min={-2} max={2} step={0.01} />
            <ParamRow label="throttle" name="throttle" value={neuron.throttle} min={0} max={1} step={0.01} />
          </div>
        </section>
        <section className="insp-section">
          <h4 className="insp-h">
            <span>Activation</span> how z becomes the steer
          </h4>
          <Segmented
            label="Activation"
            value={neuron.activation}
            onChange={(id) => send({ type: "set_param", name: "activation", value: id })}
            options={(["tanh", "sigmoid", "relu", "linear"] as const).map((id) => ({ id, label: id }))}
          />
        </section>
      </aside>
    </div>
  );
}
