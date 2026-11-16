import { memo, useMemo, useState } from "react";

import type { Diff, Mutation, Network, NetworkDetail } from "../../source/model";
import {
  describeSteer,
  describeThrottle,
  INPUT_LABELS,
  num,
  OUTPUT_LABELS,
  signed,
  useApp,
  type ConnectionRef,
  type LineMode,
  type NeuronRef,
} from "../../state/store";
import { collectEdges, edgePath, FRAME, layoutNodes, NODE_R, nodeName, OUTPUT_R, type Point, receiptPaths, strokeFor, strongPaths } from "./encode";

type Focus = { kind: "node"; layer: number; index: number } | { kind: "edge"; layer: number; from: number; to: number } | null;

const sameEdge = (f: Focus, e: ConnectionRef) =>
  f !== null && f.kind === "edge" && f.layer === e.layer && f.from === e.from && f.to === e.to;

interface EdgesProps {
  network: Network;
  nodes: Point[][];
  mode: LineMode;
  flow: number[][][] | null;
  mutation: Mutation | null;
  diff: Diff | null;
  focus: Focus;
  selected: Focus;
  paths: Set<string> | null;
  onHover: (f: Focus, event?: React.MouseEvent) => void;
}

/**
 * The connections. Each has a visible curve and a wider invisible one on top for the pointer, so
 * a hair-thin weight can still be hovered and clicked. Related lines keep their strength, the
 * rest fade back: with nothing in focus, every line shows.
 */
const Edges = memo(function Edges({ network, nodes, mode, flow, mutation, diff, focus, selected, paths, onHover }: EdgesProps) {
  const toggleConnection = useApp((s) => s.toggleConnection);
  const edges = useMemo(() => collectEdges(network), [network]);
  const layers = network.layer_sizes.length;
  const flowMax = useMemo(
    () => (flow === null ? [] : flow.map((matrix) => Math.max(1e-9, ...matrix.flat().map(Math.abs)))),
    [flow],
  );
  const radius = (layer: number) => (layer === layers - 1 ? OUTPUT_R : NODE_R);
  return (
    <g className="net-edges">
      {edges.map((e) => {
        const a = nodes[e.layer][e.from];
        const b = nodes[e.layer + 1][e.to];
        const key = `${e.layer}-${e.from}-${e.to}`;
        const stroke = strokeFor(mode, e, flow, flowMax, mutation, diff);
        const isSelected = sameEdge(selected, e);
        const related =
          focus === null
            ? true
            : focus.kind === "edge"
              ? sameEdge(focus, e)
              : (focus.layer === e.layer && focus.index === e.from) || (focus.layer === e.layer + 1 && focus.index === e.to);
        const onPath = paths !== null && paths.has(key);
        const lit = related || onPath;
        const d = edgePath(a, b, radius(e.layer), radius(e.layer + 1));
        return (
          <g key={key}>
            <path
              d={d}
              className={`net-edge net-edge-${stroke.tone}${isSelected ? " net-edge-selected" : ""}`}
              strokeWidth={stroke.width + (isSelected ? 1.6 : 0) + (onPath && focus !== null ? 0.6 : 0)}
              strokeOpacity={isSelected ? 1 : lit ? (focus === null ? stroke.opacity : Math.max(stroke.opacity, onPath ? 0.75 : 0.45)) : stroke.opacity * 0.1}
              data-testid={`net-edge-${key}`}
              data-weight={e.weight}
            />
            <path
              d={d}
              className="net-edge-hit"
              onMouseEnter={(event) => onHover({ kind: "edge", layer: e.layer, from: e.from, to: e.to }, event)}
              onMouseMove={(event) => onHover({ kind: "edge", layer: e.layer, from: e.from, to: e.to }, event)}
              onMouseLeave={() => onHover(null)}
              onClick={() => toggleConnection({ layer: e.layer, from: e.from, to: e.to })}
              data-testid={`net-hit-${key}`}
            />
          </g>
        );
      })}
    </g>
  );
});

/**
 * The network diagram. Columns are layers, circles are neurons, curves are weights.
 *   circle fill   the neuron's value now: blue positive, orange negative, stronger = larger
 *   curve         by "Lines show": the weight, what it carries now, or what a mutation changed
 * Every number shown comes from Python (the weights once, the trace in every frame).
 */
export function Diagram({
  network,
  frame,
  mutation,
  diff,
  flow,
  mode,
}: {
  network: Network;
  frame: NetworkDetail;
  mutation: Mutation | null;
  diff: Diff | null;
  /** Every connection's weight × value, or null when the source does not send it (live). */
  flow: number[][][] | null;
  mode: LineMode;
}) {
  const [hover, setHover] = useState<Focus>(null);
  const [tip, setTip] = useState<{ x: number; y: number } | null>(null);
  const neuron = useApp((s) => s.neuron);
  const connection = useApp((s) => s.connection);
  const peek = useApp((s) => s.peek);
  const output = useApp((s) => s.output);
  const toggleNeuron = useApp((s) => s.toggleNeuron);

  const sizes = network.layer_sizes;
  const layers = sizes.length;
  const nodes = useMemo(() => layoutNodes(sizes), [sizes]);
  const trace = frame.trace;
  const selected: Focus = useMemo(
    () => (neuron !== null ? { kind: "node", ...neuron } : connection !== null ? { kind: "edge", ...connection } : null),
    [neuron, connection],
  );
  const peeked: Focus = peek === null ? null : { kind: "node", ...peek };
  const focus = hover ?? peeked ?? selected;

  // The strongest routes to the selected output: back to the inputs when every layer's
  // contributions are known (recorded stand-in), else one layer back from Python's receipt.
  const receipts = frame.explain.receipts;
  const paths = useMemo(() => {
    if (output === null || hover !== null || peek !== null) return null;
    const m = output === "steer" ? 0 : 1;
    return flow !== null ? strongPaths(flow, m) : receiptPaths(receipts, m, layers - 2);
  }, [flow, receipts, output, hover, peek, layers]);

  const onHover = (f: Focus, event?: React.MouseEvent) => {
    setHover(f);
    setTip(f !== null && event ? { x: event.clientX, y: event.clientY } : null);
  };

  const describe = (f: NonNullable<Focus>) => {
    if (f.kind === "node") {
      const a = trace.activations[f.layer][f.index];
      const name = nodeName(f.layer, f.index, layers, INPUT_LABELS, OUTPUT_LABELS);
      if (f.layer === 0) return { title: name, rows: [["value", num(a, 3)]] };
      const z = trace.weighted_sums[f.layer - 1][f.index];
      const bias = network.biases[f.layer - 1][f.index];
      const fn = network.activations[f.layer - 1];
      return {
        title: name,
        rows: [
          ["Σ weight × input", signed(z - bias, 3)],
          ["bias b", signed(bias, 3)],
          ["z = Σ + b", signed(z, 3)],
          [`${fn}(z)`, signed(a, 3)],
        ],
      };
    }
    const w = network.weights[f.layer][f.to][f.from];
    const a = trace.activations[f.layer][f.from];
    const rows = [
      ["weight w", signed(w, 3)],
      ["source value a", signed(a, 3)],
    ];
    // What the connection carries now, only where a source computed it: every layer from the
    // recorded stand-in, the last layer from Python's receipt.
    const carries = flow?.[f.layer]?.[f.to]?.[f.from] ?? (f.layer === layers - 2 ? receipts[f.to]?.contributions[f.from] : undefined);
    if (carries !== undefined) rows.push(["carries w × a", signed(carries, 3)]);
    if (mode === "change" && diff !== null) rows.push([`change since unit ${diff.from_unit}`, signed(diff.weight_delta[f.layer]?.[f.to]?.[f.from] ?? 0, 4)]);
    if (mode === "mutation" && mutation !== null) {
      const mutated = mutation.weight_mutated[f.layer][f.to][f.from];
      const other = mutation.weight_from_other[f.layer][f.to][f.from];
      rows.push(["origin", mutated ? "mutated" : other ? `from row ${mutation.other_parent}` : `from row ${mutation.main_parent}`]);
    }
    return {
      title: `${nodeName(f.layer, f.from, layers, INPUT_LABELS, OUTPUT_LABELS)} → ${nodeName(f.layer + 1, f.to, layers, INPUT_LABELS, OUTPUT_LABELS)}`,
      rows,
    };
  };
  const shown = focus === null ? null : describe(focus);

  const header = (layer: number) =>
    layer === 0 ? "Inputs" : layer === layers - 1 ? "Outputs" : `Hidden ${layer}`;
  const subheader = (layer: number) =>
    layer === 0 ? `${sizes[0]} · observation` : `${sizes[layer]} · ${network.activations[layer - 1]}`;

  return (
    <div className="net-diagram">
      <svg
        viewBox={`0 0 ${FRAME.width} ${FRAME.height}`}
        className="net-svg"
        role="img"
        aria-label="The network of the selected car"
        data-testid="network-diagram"
        data-mode={mode}
        onMouseLeave={() => onHover(null)}
      >
        {sizes.map((_, layer) => (
          <g key={`head-${layer}`}>
            <text x={nodes[layer][0].x} y={18} textAnchor="middle" className="net-colhead">
              {header(layer)}
            </text>
            <text x={nodes[layer][0].x} y={35} textAnchor="middle" className="net-colsub">
              {subheader(layer)}
            </text>
          </g>
        ))}
        {network.weights.map((matrix, layer) => (
          <text
            key={`w-${layer}`}
            x={(nodes[layer][0].x + nodes[layer + 1][0].x) / 2}
            y={FRAME.height - 12}
            textAnchor="middle"
            className="net-colsub"
          >
            W{layer + 1}: {matrix.length * matrix[0].length} weights + {network.biases[layer].length} biases
          </text>
        ))}

        <Edges
          network={network}
          nodes={nodes}
          mode={mode}
          flow={flow}
          mutation={mutation}
          diff={diff}
          focus={focus}
          selected={selected}
          paths={paths}
          onHover={onHover}
        />

        {nodes.map((column, layer) =>
          column.map((p, index) => {
            const a = trace.activations[layer][index];
            const isSelected = selected !== null && selected.kind === "node" && selected.layer === layer && selected.index === index;
            const isFocus = focus !== null && focus.kind === "node" && focus.layer === layer && focus.index === index;
            const r = layer === layers - 1 ? OUTPUT_R : NODE_R;
            const ref: NeuronRef = { layer, index };
            const biasMutated = mode === "mutation" && layer > 0 && (mutation?.bias_mutated[layer - 1]?.[index] ?? false);
            return (
              <g
                key={`${layer}-${index}`}
                className={`net-node${isSelected ? " net-node-selected" : ""}${isFocus ? " net-node-focus" : ""}`}
                data-testid={`net-node-${layer}-${index}`}
                data-activation={a}
                role="button"
                aria-pressed={isSelected}
                aria-label={nodeName(layer, index, layers, INPUT_LABELS, OUTPUT_LABELS)}
                onMouseEnter={(event) => onHover({ kind: "node", ...ref }, event)}
                onMouseMove={(event) => onHover({ kind: "node", ...ref }, event)}
                onMouseLeave={() => onHover(null)}
                onClick={() => toggleNeuron(ref, layers)}
              >
                {isSelected && <circle cx={p.x} cy={p.y} r={r + 6} className="net-node-halo" />}
                <circle cx={p.x} cy={p.y} r={r} className="net-node-base" />
                <circle cx={p.x} cy={p.y} r={r} className={a < 0 ? "net-fill-neg" : "net-fill-pos"} fillOpacity={Math.max(0.04, Math.min(1, Math.abs(a)))} />
                <circle cx={p.x} cy={p.y} r={r} className="net-node-ring" />
                {biasMutated && <circle cx={p.x} cy={p.y} r={r + 3.5} className="net-node-mutated" />}
              </g>
            );
          }),
        )}

        {/* Inputs: name, value and a small bar, so a sensor reading can be read without hovering. */}
        {nodes[0].map((p, k) => {
          const v = trace.activations[0][k];
          const isSigned = k === 6;
          const x0 = p.x - 58;
          const w = 40;
          const dim = focus !== null && !(focus.kind === "node" && focus.layer === 0 && focus.index === k) && selected?.kind === "node" && selected.layer === 0 && selected.index !== k;
          return (
            <g key={`in-${k}`} className="net-io" opacity={dim ? 0.55 : 1}>
              <text x={p.x - 104} y={p.y + 4.5} textAnchor="end" className="net-label">
                {INPUT_LABELS[k]}
              </text>
              <text x={p.x - 65} y={p.y + 4.5} textAnchor="end" className="net-value" data-testid={`net-input-${k}`}>
                {num(v)}
              </text>
              <rect x={x0} y={p.y - 3} width={w} height={6} rx={1.5} className="net-bar-track" />
              {isSigned ? (
                <>
                  <rect
                    x={v >= 0 ? x0 + w / 2 : x0 + w / 2 + Math.max(-1, v) * (w / 2)}
                    y={p.y - 3}
                    width={Math.min(1, Math.abs(v)) * (w / 2)}
                    height={6}
                    rx={1.5}
                    className={v < 0 ? "net-fill-neg" : "net-fill-pos"}
                  />
                  <line x1={x0 + w / 2} x2={x0 + w / 2} y1={p.y - 5} y2={p.y + 5} className="net-bar-zero" />
                </>
              ) : (
                <rect x={x0} y={p.y - 3} width={Math.max(0, Math.min(1, v)) * w} height={6} rx={1.5} className="net-fill-pos" />
              )}
            </g>
          );
        })}

        {/* Outputs: name, value, a signed bar and the meaning in words. */}
        {nodes[layers - 1].map((p, k) => {
          const v = trace.activations[layers - 1][k];
          const x0 = p.x + 24;
          const w = 118;
          return (
            <g key={`out-${k}`} className="net-io">
              <text x={x0} y={p.y - 9} className="net-outname">
                {OUTPUT_LABELS[k]}
              </text>
              <text x={x0 + w} y={p.y - 9} textAnchor="end" className="net-outvalue" data-testid={`net-output-${k}`}>
                {num(v)}
              </text>
              <rect x={x0} y={p.y - 2} width={w} height={7} rx={1.5} className="net-bar-track" />
              <rect
                x={v >= 0 ? x0 + w / 2 : x0 + w / 2 + Math.max(-1, v) * (w / 2)}
                y={p.y - 2}
                width={Math.min(1, Math.abs(v)) * (w / 2)}
                height={7}
                rx={1.5}
                className={v < 0 ? "net-fill-neg" : "net-fill-pos"}
              />
              <line x1={x0 + w / 2} x2={x0 + w / 2} y1={p.y - 5} y2={p.y + 8} className="net-bar-zero" />
              <text x={x0} y={p.y + 23} className="net-colsub">
                {k === 0 ? describeSteer(v) : describeThrottle(v)}
              </text>
            </g>
          );
        })}
      </svg>

      {tip !== null && shown !== null && hover !== null && (
        <div className="float-tip" style={{ left: Math.min(tip.x + 14, window.innerWidth - 250), top: tip.y + 16 }} data-testid="net-tip">
          <h5>{shown.title}</h5>
          {shown.rows.map(([label, value]) => (
            <div className="row" key={label}>
              <span>{label}</span>
              <span className="num">{value}</span>
            </div>
          ))}
        </div>
      )}

      <p className="net-readout" data-testid="network-readout">
        {shown === null ? (
          "Hover a neuron or a connection for its exact numbers; click to keep it selected in every view."
        ) : (
          <>
            <b>{focus === selected ? "Selected · " : ""}{shown.title}</b>
            {shown.rows.map(([label, value]) => (
              <span key={label}>
                {label} <span className="num">{value}</span>
              </span>
            ))}
          </>
        )}
      </p>
    </div>
  );
}
