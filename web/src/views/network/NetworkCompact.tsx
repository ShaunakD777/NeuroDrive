import type { NetworkMessage, NetworkTrace } from "../../api/generated";
import { useSelection } from "../../state/selection";
import { describeSteer } from "./curve";
import {
  describeThrottle,
  edgeDescription,
  nodeDescription,
  nodeFill,
  nodeLabel,
  num,
} from "./networkLayout";

/**
 * A bar for one value. Unsigned values (0 to 1) grow from the left edge; signed values (-1 to 1)
 * grow from the middle, to the right for positive and to the left for negative.
 */
function ValueBar({ value, signed, big = false }: { value: number; signed: boolean; big?: boolean }) {
  const v = Math.max(signed ? -1 : 0, Math.min(1, value));
  const style = signed
    ? { left: v >= 0 ? "50%" : `${50 + v * 50}%`, width: `${Math.abs(v) * 50}%` }
    : { left: "0%", width: `${v * 100}%` };
  return (
    <span className={big ? "cnet-bar cnet-bar-big" : "cnet-bar"} aria-hidden="true">
      {signed && <span className="cnet-bar-mid" />}
      <span className={`cnet-bar-fill cnet-fill-${v < 0 ? "neg" : "pos"}`} style={style} />
    </span>
  );
}

const isSame = (a: { layer: number; index: number } | null, layer: number, index: number) =>
  a !== null && a.layer === layer && a.index === index;

/**
 * The compact network for the side drawer, designed for a narrow column instead of shrinking the
 * full diagram: the seven inputs as labelled bars, each hidden layer as a strip of coloured cells
 * (one per neuron: colour = sign, strength = size of its activation), and the two outputs as big
 * bars with words. Everything is clickable and uses the shared selection, so a click here
 * selects the same neuron in the full diagram and highlights its ray on the track.
 */
export function NetworkCompact({ network, trace }: { network: NetworkMessage; trace: NetworkTrace }) {
  const selected = useSelection((s) => s.neuron);
  const connection = useSelection((s) => s.connection);
  const toggleNeuron = useSelection((s) => s.toggleNeuron);
  const layers = network.layer_sizes.length;
  const inputs = trace.activations[0] ?? [];
  const outputs = trace.activations[layers - 1] ?? [];

  const describeNode = (layer: number, index: number) => {
    const a = trace.activations[layer]?.[index] ?? 0;
    if (layer === 0) return nodeDescription(layer, index, layers, a, null, null, null);
    return nodeDescription(
      layer,
      index,
      layers,
      a,
      trace.weighted_sums[layer - 1]?.[index] ?? 0,
      network.biases[layer - 1]?.[index] ?? 0,
      network.activations[layer - 1] ?? null,
    );
  };

  const readout =
    selected !== null
      ? describeNode(selected.layer, selected.index)
      : connection !== null
        ? edgeDescription(
            connection.layer,
            connection.from,
            connection.to,
            layers,
            network.weights[connection.layer]?.[connection.to]?.[connection.from] ?? 0,
          )
        : "Click a row or a cell to select that neuron everywhere.";

  return (
    <div className="cnet" data-testid="network-compact">
      <div className="cnet-section-title">Inputs</div>
      {inputs.map((value, k) => (
        <button
          key={`in-${k}`}
          type="button"
          className={isSame(selected, 0, k) ? "cnet-row cnet-selected" : "cnet-row"}
          data-testid={`cnet-input-${k}`}
          data-value={value}
          aria-pressed={isSame(selected, 0, k)}
          title={describeNode(0, k)}
          onClick={() => toggleNeuron({ layer: 0, index: k }, layers)}
        >
          <span className="cnet-label">{nodeLabel(0, k, layers)}</span>
          <ValueBar value={value} signed={k === 6} />
          <span className="cnet-value">{num(value)}</span>
        </button>
      ))}

      {network.layer_sizes.slice(1, -1).map((size, h) => {
        const layer = h + 1;
        return (
          <div key={`hidden-${layer}`} className="cnet-hidden">
            <div className="cnet-section-title">
              Hidden {layer} <span className="cnet-muted">({size} neurons)</span>
            </div>
            <div className="cnet-cells" style={{ gridTemplateColumns: `repeat(${Math.min(size, 16)}, 1fr)` }}>
              {Array.from({ length: size }, (_, i) => {
                const a = trace.activations[layer]?.[i] ?? 0;
                const fill = nodeFill(a);
                const on = isSame(selected, layer, i);
                return (
                  <button
                    key={i}
                    type="button"
                    className={on ? "cnet-cell cnet-selected" : "cnet-cell"}
                    data-testid={`cnet-cell-${layer}-${i}`}
                    data-activation={a}
                    aria-pressed={on}
                    aria-label={nodeLabel(layer, i, layers)}
                    title={describeNode(layer, i)}
                    onClick={() => toggleNeuron({ layer, index: i }, layers)}
                  >
                    <span className={`cnet-cell-fill cnet-fill-${fill.sign}`} style={{ opacity: fill.opacity }} />
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}

      <div className="cnet-section-title">Outputs</div>
      {outputs.map((value, k) => {
        const on = isSame(selected, layers - 1, k);
        return (
          <button
            key={`out-${k}`}
            type="button"
            className={on ? "cnet-row cnet-output cnet-selected" : "cnet-row cnet-output"}
            data-testid={`cnet-output-${k}`}
            data-value={value}
            aria-pressed={on}
            title={describeNode(layers - 1, k)}
            onClick={() => toggleNeuron({ layer: layers - 1, index: k }, layers)}
          >
            <span className="cnet-label">{nodeLabel(layers - 1, k, layers)}</span>
            <ValueBar value={value} signed big />
            <span className="cnet-value">{num(value)}</span>
            <span className="cnet-words">{k === 0 ? describeSteer(value) : describeThrottle(value)}</span>
          </button>
        );
      })}

      <p className="cnet-readout" data-testid="network-compact-readout">
        {readout}
      </p>
      <a className="cnet-open" href="#/network" data-testid="cnet-open-full">
        Open the full diagram →
      </a>
    </div>
  );
}
