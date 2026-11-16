import { whatIf as askWhatIf } from "../../mock/mockServer";
import type { CarFrame, CarRecord } from "../../mock/types";
import { describeSteer, describeThrottle, INPUT_LABELS, num, OUTPUT_LABELS, signed, useApp } from "../../state/store";
import { Button, ValueBar } from "../../ui/controls";

const TOP_ROWS = 6;
const describe = (m: number, v: number) => (m === 0 ? describeSteer(v) : describeThrottle(v));

/** The two outputs as cards: the entry point of the inspector. Clicking one explains it. */
export function OutputCards({ frame, layers }: { frame: CarFrame; layers: number }) {
  const output = useApp((s) => s.output);
  const toggleNeuron = useApp((s) => s.toggleNeuron);
  return (
    <div className="outcards">
      {OUTPUT_LABELS.map((label, m) => {
        const value = frame.explain.receipts[m].value;
        const active = output === (m === 0 ? "steer" : "throttle");
        return (
          <button
            key={label}
            type="button"
            className="outcard"
            aria-pressed={active}
            onClick={() => toggleNeuron({ layer: layers - 1, index: m }, layers)}
            data-testid={`inspect-${label.toLowerCase()}`}
          >
            <span className="eyebrow">{label}</span>
            <span className="outcard-value num">{num(value)}</span>
            <ValueBar value={value} />
            <span className="outcard-word">{describe(m, value)}</span>
          </button>
        );
      })}
    </div>
  );
}

/**
 * The activation function with the neuron's own point on it: where z lands on the curve is why
 * a large z barely moves the output any more (saturation). Production draws the curve from
 * `GET activations`, which Python computes; the prototype uses tanh directly.
 */
function ActivationPlot({ z, a, name }: { z: number; a: number; name: string }) {
  const W = 150;
  const H = 86;
  const x = (v: number) => 8 + ((v + 3) / 6) * (W - 16);
  const y = (v: number) => H / 2 - v * (H / 2 - 10);
  let d = "";
  for (let i = 0; i <= 60; i++) {
    const v = -3 + (6 * i) / 60;
    d += `${i === 0 ? "M" : "L"}${x(v).toFixed(1)} ${y(Math.tanh(v)).toFixed(1)}`;
  }
  const zc = Math.max(-3, Math.min(3, z));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="actplot" role="img" aria-label={`${name}: z ${num(z, 3)} gives ${num(a, 3)}`}>
      <line x1={8} x2={W - 8} y1={y(0)} y2={y(0)} className="actplot-axis" />
      <line x1={x(0)} x2={x(0)} y1={8} y2={H - 8} className="actplot-axis" />
      <path d={d} className="actplot-curve" />
      <line x1={x(zc)} x2={x(zc)} y1={y(0)} y2={y(a)} className="actplot-guide" />
      <line x1={x(0)} x2={x(zc)} y1={y(a)} y2={y(a)} className="actplot-guide" />
      <circle cx={x(zc)} cy={y(a)} r={4.5} className="actplot-dot" />
      <text x={W - 8} y={y(0) + 11} textAnchor="end" className="actplot-text">
        z
      </text>
      <text x={10} y={16} className="actplot-text">
        {name}(z)
      </text>
    </svg>
  );
}

/**
 * Why the selected output has its value: the receipt. Each row is one hidden neuron's
 * contribution c = weight × its value; the rows, plus the bias, add up to z; the activation
 * function turns z into the output. All numbers are Python's (`selected.explain`).
 */
function Why({ frame, m, record }: { frame: CarFrame; m: number; record: CarRecord }) {
  const set = useApp((s) => s.set);
  const receipt = frame.explain.receipts[m];
  const layers = record.network.layer_sizes.length;
  const fn = record.network.activations[layers - 2];
  const ranked = receipt.contributions
    .map((c, index) => ({ index, c, a: receipt.inputs[index] }))
    .sort((p, q) => Math.abs(q.c) - Math.abs(p.c));
  const shown = ranked.slice(0, TOP_ROWS);
  const rest = ranked.slice(TOP_ROWS);
  const restSum = rest.reduce((sum, r) => sum + r.c, 0);
  const scale = Math.max(1e-6, ...ranked.map((r) => Math.abs(r.c)), Math.abs(receipt.bias), Math.abs(restSum));
  const hidden = layers - 2;
  return (
    <section className="insp-section" data-testid="inspector-why">
      <h4 className="insp-h">
        <span>Why</span> what pushed it
      </h4>
      <div className="insp-rows">
        <div className="insp-row insp-head">
          <span>from</span>
          <span>weight × value</span>
          <span />
        </div>
        {shown.map((r) => (
          <div
            key={r.index}
            className="insp-row insp-hover"
            onMouseEnter={() => set({ peek: { layer: hidden, index: r.index } })}
            onMouseLeave={() => set({ peek: null })}
            data-testid={`receipt-row-${r.index}`}
          >
            <span>
              Hidden {hidden} · {r.index + 1}
            </span>
            <ValueBar value={r.c / scale} />
            <span className="num">{signed(r.c, 3)}</span>
          </div>
        ))}
        {rest.length > 0 && (
          <div className="insp-row insp-muted">
            <span>{rest.length} smaller, together</span>
            <ValueBar value={restSum / scale} />
            <span className="num">{signed(restSum, 3)}</span>
          </div>
        )}
        <div className="insp-row insp-muted">
          <span>bias</span>
          <ValueBar value={receipt.bias / scale} />
          <span className="num">{signed(receipt.bias, 3)}</span>
        </div>
      </div>
      <div className="insp-sum">
        <div className="insp-eq" data-testid="inspector-sum">
          <span className="eyebrow">Weighted sum</span>
          <span className="num insp-eq-line">z = Σ c + b = {signed(receipt.z, 3)}</span>
          <span className="eyebrow">Activation</span>
          <span className="num insp-eq-line">
            {fn}(z) = <b>{signed(receipt.value, 3)}</b>
          </span>
        </div>
        <ActivationPlot z={receipt.z} a={receipt.value} name={fn} />
      </div>
    </section>
  );
}

/** How the output would move if one input changed a little: ∂output / ∂input, from backprop. */
function Sensitivity({ frame, m }: { frame: CarFrame; m: number }) {
  const set = useApp((s) => s.set);
  const slopes = frame.explain.sensitivity[m];
  const scale = Math.max(1e-6, ...slopes.map(Math.abs));
  const strongest = slopes.reduce((best, s, k) => (Math.abs(s) > Math.abs(slopes[best]) ? k : best), 0);
  const s = slopes[strongest];
  const direction =
    m === 0 ? (s > 0 ? "towards the right" : "towards the left") : s > 0 ? "towards accelerating" : "towards braking";
  return (
    <section className="insp-section" data-testid="inspector-sensitivity">
      <h4 className="insp-h">
        <span>Sensitivity</span> how it reacts to each input
      </h4>
      <div className="insp-rows">
        {slopes.map((slope, k) => (
          <div
            key={k}
            className="insp-row insp-hover"
            data-strongest={k === strongest}
            onMouseEnter={() => set({ peek: { layer: 0, index: k } })}
            onMouseLeave={() => set({ peek: null })}
            data-testid={`sensitivity-${k}`}
          >
            <span>{INPUT_LABELS[k]}</span>
            <ValueBar value={slope / scale} />
            <span className="num">{signed(slope, 3)}</span>
          </div>
        ))}
      </div>
      <p className="insp-note" data-testid="inspector-nudge">
        If <b>{INPUT_LABELS[strongest]}</b> rose by 0.1, {OUTPUT_LABELS[m].toLowerCase()} would change by about{" "}
        <b className="num">{signed(s * 0.1, 3)}</b> ({direction}). The rays on the track are coloured by these numbers.
      </p>
    </section>
  );
}

/** Change the inputs and ask what the network would do. The simulation itself never changes. */
function WhatIf({ frame, record }: { frame: CarFrame; record: CarRecord }) {
  const playing = useApp((s) => s.playing);
  const looking = useApp((s) => s.viewKey !== null);
  const overrides = useApp((s) => s.whatIf);
  const setWhatIf = useApp((s) => s.setWhatIf);
  const clearWhatIf = useApp((s) => s.clearWhatIf);
  const pause = useApp((s) => s.pause);
  const lookAt = useApp((s) => s.lookAt);
  const changed = Object.keys(overrides).length > 0;

  if (playing || looking) {
    return (
      <section className="insp-section" data-testid="whatif-panel">
        <h4 className="insp-h">
          <span>What if</span> change an input
        </h4>
        <div className="insp-locked">
          <span>{playing ? "What-if asks about one frozen moment." : "What-if asks about the car now, not an earlier step."}</span>
          <Button icon={playing ? "pause" : "right"} onClick={playing ? pause : () => lookAt(null)}>
            {playing ? "Pause to try it" : "Back to now"}
          </Button>
        </div>
      </section>
    );
  }

  const answer = changed ? askWhatIf(record, frame, overrides) : null;
  return (
    <section className="insp-section" data-testid="whatif-panel">
      <h4 className="insp-h">
        <span>What if</span> change an input
        <Button variant="ghost" icon="reset" disabled={!changed} onClick={clearWhatIf} testId="whatif-reset">
          Real inputs
        </Button>
      </h4>
      <div className="insp-rows">
        {frame.observation.map((value, k) => {
          const min = k === 6 ? -1 : 0;
          const current = k in overrides ? overrides[k] : value;
          return (
            <label key={k} className="insp-row whatif-row" data-changed={k in overrides}>
              <span>{INPUT_LABELS[k]}</span>
              <input
                type="range"
                min={min}
                max={1}
                step={0.01}
                value={current}
                onChange={(e) => setWhatIf(k, Number(e.target.value))}
                data-testid={`whatif-${k}`}
              />
              <span className="num">{num(current)}</span>
            </label>
          );
        })}
      </div>
      {answer === null ? (
        <p className="insp-note">
          Move a slider. Python runs this car&apos;s network on the changed inputs; a ghost car on the track shows where the new action
          would take it in 2 s.
        </p>
      ) : (
        <table className="table whatif-table" data-testid="whatif-result">
          <thead>
            <tr>
              <th />
              <th className="r">now</th>
              <th className="r">what-if</th>
              <th className="r" data-tip="now + sensitivity × change: a straight-line guess">
                predicted
              </th>
            </tr>
          </thead>
          <tbody>
            {OUTPUT_LABELS.map((label, m) => (
              <tr key={label}>
                <td>{label}</td>
                <td className="r num">{num(answer.base[m], 3)}</td>
                <td className="r">
                  <b className="num">{num(answer.tried[m], 3)}</b> <span className="muted">{describe(m, answer.tried[m])}</span>
                </td>
                <td className="r num">{num(answer.predicted[m], 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

/**
 * The inspector rail beside the diagram (production: OutputInspector + WhatIfPanel, below the
 * diagram). Top: the two outputs. With one chosen: why it has its value, what it reacts to, and
 * what it would do with other inputs. Nothing is hidden behind tabs; the rail scrolls.
 */
export function Inspector({ frame, record }: { frame: CarFrame; record: CarRecord }) {
  const output = useApp((s) => s.output);
  const layers = record.network.layer_sizes.length;
  const m = output === null ? null : output === "steer" ? 0 : 1;
  return (
    <aside className="inspector" data-testid="output-inspector">
      <OutputCards frame={frame} layers={layers} />
      {m === null ? (
        <div className="insp-empty" data-testid="inspection-hint">
          <span className="empty-title">Pick an output to explain it</span>
          <span>
            Click <b>Steer</b> or <b>Throttle</b> above, or its circle in the diagram. You will see which neurons pushed it, which
            sensors it reacts to, and what it would do if a sensor read differently.
          </span>
        </div>
      ) : (
        <>
          <p className="insp-headline" data-testid="inspector-headline">
            <span className="num">
              {OUTPUT_LABELS[m]} = {num(frame.explain.receipts[m].value)}
            </span>
            <span>{describe(m, frame.explain.receipts[m].value)}</span>
          </p>
          <Why frame={frame} m={m} record={record} />
          <Sensitivity frame={frame} m={m} />
          <WhatIf frame={frame} record={record} />
        </>
      )}
    </aside>
  );
}
