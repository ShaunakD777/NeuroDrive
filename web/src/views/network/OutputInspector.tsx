import type { Explanation, NetworkMessage } from "../../api/generated";
import { useSelection } from "../../state/selection";
import { ValueBar } from "./NetworkCompact";
import { INPUT_LABELS, num, OUTPUT_LABELS } from "./networkLayout";
import { maxAbs, nudgeSentence, outputHeadline, rankedReceipt, topAndRest } from "./inspect";

const TOP_ROWS = 6;

/**
 * The output inspector (task 2.10): why the selected output has its value right now. Three parts,
 * all computed by Python (`selected.explain` in each frame):
 * 1. the value and what it means for the car;
 * 2. the receipt: each hidden neuron's contribution (weight times its value), largest first, plus
 *    the bias, adding up to z;
 * 3. the sensitivity: how much the output would move if each input changed a little.
 */
export function OutputInspector({
  explain,
  network,
  output,
}: {
  explain: Explanation;
  network: NetworkMessage;
  output: number;
}) {
  const toggleNeuron = useSelection((s) => s.toggleNeuron);
  const layers = network.layer_sizes.length;
  const receipt = explain.receipts[output];
  const slopes = explain.sensitivity[output] ?? [];
  if (receipt === undefined) return null;

  const hiddenLayer = layers - 2; // the layer feeding the outputs
  const { shown, rest, restCount } = topAndRest(rankedReceipt(receipt, hiddenLayer), TOP_ROWS);
  const scale = maxAbs([...receipt.contributions, receipt.bias]);
  const slopeScale = maxAbs(slopes);
  const strongest = slopes.reduce((best, s, k) => (Math.abs(s) > Math.abs(slopes[best] ?? 0) ? k : best), 0);
  const activation = network.activations[layers - 2] ?? "f";

  return (
    <section className="inspector" data-testid="output-inspector">
      <div className="inspector-tabs" role="group" aria-label="Output">
        {OUTPUT_LABELS.map((label, m) => (
          <button
            key={label}
            type="button"
            aria-pressed={m === output}
            data-testid={`inspect-${label.toLowerCase()}`}
            onClick={() => m !== output && toggleNeuron({ layer: layers - 1, index: m }, layers)}
          >
            {label}
          </button>
        ))}
      </div>
      <h3 className="inspector-headline" data-testid="inspector-headline">
        {outputHeadline(output, receipt.value)}
      </h3>

      <h4>Why: what pushed it ({activation}(z) with z = contributions + bias)</h4>
      <div className="inspector-rows" data-testid="inspector-receipt">
        {shown.map((row) => (
          <div key={row.index} className="inspector-row" data-testid={`receipt-row-${row.index}`}>
            <span>{row.label}</span>
            <ValueBar value={row.contribution / scale} signed />
            <span className="cnet-value">{num(row.contribution, 3)}</span>
          </div>
        ))}
        {restCount > 0 && (
          <div className="inspector-row inspector-muted">
            <span>{restCount} smaller ones together</span>
            <ValueBar value={rest / scale} signed />
            <span className="cnet-value">{num(rest, 3)}</span>
          </div>
        )}
        <div className="inspector-row inspector-muted">
          <span>bias</span>
          <ValueBar value={receipt.bias / scale} signed />
          <span className="cnet-value">{num(receipt.bias, 3)}</span>
        </div>
        <div className="inspector-sum" data-testid="inspector-sum">
          z = {num(receipt.z, 3)} → {activation}(z) = <strong>{num(receipt.value, 3)}</strong>
        </div>
      </div>

      <h4>How it reacts: sensitivity to each input</h4>
      <div className="inspector-rows" data-testid="inspector-sensitivity">
        {slopes.map((s, k) => (
          <div key={k} className="inspector-row" data-testid={`sensitivity-${k}`} data-value={s}>
            <span>{INPUT_LABELS[k] ?? `Input ${k + 1}`}</span>
            <ValueBar value={s / slopeScale} signed />
            <span className="cnet-value">{num(s, 3)}</span>
          </div>
        ))}
      </div>
      {slopes.length > 0 && (
        <p className="hint" data-testid="inspector-nudge">
          {nudgeSentence(output, strongest, slopes[strongest] ?? 0)}
        </p>
      )}
    </section>
  );
}
