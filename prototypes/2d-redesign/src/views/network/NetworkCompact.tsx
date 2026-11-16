import { carName, hasNetwork } from "../../source/model";
import { featuredCars } from "../../source/recorded";
import { useWorld } from "../../source/world";
import { describeSteer, describeThrottle, INPUT_LABELS, num, OUTPUT_LABELS, useApp } from "../../state/store";
import { Button, ValueBar } from "../../ui/controls";
import { SourceWaiting } from "../../ui/SourceWaiting";

/**
 * The network in a drawer: a summary built for a narrow column, not a shrunken diagram (as in
 * production, task 2.8). Input bars, one strip of cells per hidden layer, output bars with
 * words. Every row and cell selects that neuron in every view.
 */
export function NetworkCompact() {
  const world = useWorld();
  const data = useApp((s) => s.data);
  const neuron = useApp((s) => s.neuron);
  const toggleNeuron = useApp((s) => s.toggleNeuron);
  const selectCar = useApp((s) => s.selectCar);
  if (world === null) return <SourceWaiting what="a car" height={240} />;
  const network = world.network;
  const frame = world.shown;
  if (network === null || !hasNetwork(frame)) {
    const featured = world.source === "recorded" ? featuredCars(data) : null;
    return (
      <div className="empty">
        <span className="empty-title">
          {featured !== null ? `${carName(world.scenario, world.selected)} was not recorded with its network` : "No network to show"}
        </span>
        {featured !== null ? (
          <>
            <span>The recording kept five networks. Live, any car&apos;s network appears here.</span>
            <Button onClick={() => selectCar(world.scenario.population?.champion ?? featured[0])}>Show the champion</Button>
          </>
        ) : (
          <span>
            {frame?.neuron ? "The neuron driver has one neuron: open the Network view to set its weights." : "This driver decides without a neural network."}
          </span>
        )}
      </div>
    );
  }
  const sizes = network.layer_sizes;
  const layers = sizes.length;
  const isSelected = (layer: number, index: number) => neuron !== null && neuron.layer === layer && neuron.index === index;
  const acts = frame.trace.activations;
  return (
    <div className="cnet" data-testid="network-compact">
      <div className="cnet-group">
        <span className="eyebrow">Inputs · what the car senses</span>
        {acts[0].map((v, k) => (
          <button key={k} type="button" className="cnet-row" aria-pressed={isSelected(0, k)} onClick={() => toggleNeuron({ layer: 0, index: k }, layers)}>
            <span>{INPUT_LABELS[k]}</span>
            <ValueBar value={v} signed={k === 6} tone={k === 6 ? undefined : "pos"} />
            <span className="num">{num(v)}</span>
          </button>
        ))}
      </div>
      {acts.slice(1, -1).map((row, h) => (
        <div className="cnet-group" key={h}>
          <span className="eyebrow">
            Hidden {h + 1} · {row.length} neurons · {network.activations[h]}
          </span>
          <div className="cnet-cells">
            {row.map((v, k) => (
              <button
                key={k}
                type="button"
                className="cnet-cell"
                aria-pressed={isSelected(h + 1, k)}
                aria-label={`Hidden ${h + 1}, neuron ${k + 1}: ${num(v)}`}
                data-tip={`${k + 1}: ${num(v)}`}
                onClick={() => toggleNeuron({ layer: h + 1, index: k }, layers)}
              >
                <i className={v < 0 ? "neg" : "pos"} style={{ opacity: Math.max(0.06, Math.min(1, Math.abs(v))) }} />
              </button>
            ))}
          </div>
        </div>
      ))}
      <div className="cnet-group">
        <span className="eyebrow">Outputs · what the car does</span>
        {acts[layers - 1].map((v, k) => (
          <button
            key={k}
            type="button"
            className="cnet-row cnet-out"
            aria-pressed={isSelected(layers - 1, k)}
            onClick={() => toggleNeuron({ layer: layers - 1, index: k }, layers)}
            data-testid={`cnet-output-${k}`}
          >
            <span>
              <b>{OUTPUT_LABELS[k]}</b>
              <small>{k === 0 ? describeSteer(v) : describeThrottle(v)}</small>
            </span>
            <ValueBar value={v} />
            <span className="num">{num(v)}</span>
          </button>
        ))}
      </div>
      <a className="cnet-link" href="#/network">
        Open the full diagram and the inspector →
      </a>
    </div>
  );
}
