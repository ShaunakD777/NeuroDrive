import { memo, useMemo, useState } from "react";

import type { NetworkMessage, NetworkTrace } from "../../api/generated";
import { useSelection } from "../../state/selection";
import {
  columnTitle,
  edgeDescription,
  edgeOpacity,
  edgeWidth,
  type Frame,
  layoutNodes,
  nodeDescription,
  nodeFill,
  nodeLabel,
  num,
  type Point,
  signOf,
} from "./networkLayout";

const FRAME: Frame = { width: 880, height: 470, marginX: 165, marginY: 40 };
const NODE_RADIUS = 10;

/** What the diagram is pointing at: the hovered thing, else the selected one (or nothing). */
type Hover =
  | { kind: "node"; layer: number; index: number }
  | { kind: "edge"; layer: number; from: number; to: number }
  | null;

const sameEdge = (h: Hover, e: { layer: number; from: number; to: number }) =>
  h !== null && h.kind === "edge" && h.layer === e.layer && h.from === e.from && h.to === e.to;

/** True when two pointers name the same circle or the same line. */
function sameThing(a: Hover, b: Hover): boolean {
  if (a === null || b === null) return false;
  if (a.kind === "node" && b.kind === "node") return a.layer === b.layer && a.index === b.index;
  return a.kind === "edge" && sameEdge(b, a);
}

interface EdgeData {
  layer: number;
  from: number;
  to: number;
  weight: number;
}

/** Every connection of the network, weakest first so the strong ones are drawn on top. */
function collectEdges(network: NetworkMessage): EdgeData[] {
  const edges: EdgeData[] = [];
  network.weights.forEach((matrix, layer) => {
    matrix.forEach((row, to) => {
      row.forEach((weight, from) => edges.push({ layer, from, to, weight }));
    });
  });
  return edges.sort((a, b) => Math.abs(a.weight) - Math.abs(b.weight));
}

/** True when the hovered thing touches this connection (so the others can be dimmed). */
function isRelated(hover: Hover, e: EdgeData): boolean {
  if (hover === null) return true;
  if (hover.kind === "edge") return hover.layer === e.layer && hover.from === e.from && hover.to === e.to;
  return (
    (hover.layer === e.layer && hover.index === e.from) ||
    (hover.layer === e.layer + 1 && hover.index === e.to)
  );
}

interface EdgesProps {
  network: NetworkMessage;
  nodes: Point[][];
  /** What to emphasise: the hovered thing, else the selected one. */
  hover: Hover;
  /** The selected connection, drawn with an outline. */
  selected: Hover;
  onHover: (h: Hover) => void;
}

/**
 * The lines. They only change when the network, the hover or the selection does, not on every
 * frame. Each weight has a visible line and a wider transparent "hit" line on top, so thin lines
 * can still be pointed at and clicked.
 */
const Edges = memo(function Edges({ network, nodes, hover, selected, onHover }: EdgesProps) {
  const toggleConnection = useSelection((s) => s.toggleConnection);
  const edges = useMemo(() => collectEdges(network), [network]);
  const layers = network.layer_sizes.length;
  return (
    <g className="net-edges">
      {edges.map((e) => {
        const a = nodes[e.layer]?.[e.from];
        const b = nodes[e.layer + 1]?.[e.to];
        if (!a || !b) return null;
        const dimmed = !isRelated(hover, e);
        const isSelected = sameEdge(selected, e);
        const ref = { layer: e.layer, from: e.from, to: e.to };
        return (
          <g key={`${e.layer}-${e.from}-${e.to}`}>
            <line
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              className={`net-edge net-edge-${signOf(e.weight)}${dimmed ? " net-dim" : ""}${isSelected ? " net-edge-selected" : ""}`}
              strokeWidth={edgeWidth(e.weight) + (isSelected ? 2 : 0)}
              strokeOpacity={isSelected ? 1 : edgeOpacity(e.weight)}
              data-testid={`net-edge-${e.layer}-${e.from}-${e.to}`}
              data-weight={e.weight}
              data-selected={isSelected}
            />
            <line
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              className="net-edge-hit"
              data-testid={`net-hit-${e.layer}-${e.from}-${e.to}`}
              onMouseEnter={() => onHover({ kind: "edge", ...ref })}
              onMouseLeave={() => onHover(null)}
              onClick={() => toggleConnection(ref)}
            >
              <title>{edgeDescription(e.layer, e.from, e.to, layers, e.weight)}</title>
            </line>
          </g>
        );
      })}
    </g>
  );
});

function Legend() {
  return (
    <div className="net-legend" data-testid="network-legend">
      <span className="net-legend-item">
        <svg width="38" height="14" aria-hidden="true">
          <line x1="2" y1="7" x2="36" y2="7" className="net-edge net-edge-pos" strokeWidth="3" />
        </svg>
        positive weight
      </span>
      <span className="net-legend-item">
        <svg width="38" height="14" aria-hidden="true">
          <line x1="2" y1="7" x2="36" y2="7" className="net-edge net-edge-neg" strokeWidth="3" />
        </svg>
        negative weight
      </span>
      <span className="net-legend-item">
        <svg width="38" height="14" aria-hidden="true">
          <line x1="2" y1="3" x2="36" y2="3" className="net-edge net-edge-pos" strokeWidth="0.8" />
          <line x1="2" y1="10" x2="36" y2="10" className="net-edge net-edge-pos" strokeWidth="4" />
        </svg>
        thicker = larger |weight|
      </span>
      <span className="net-legend-item">
        <svg width="62" height="18" aria-hidden="true">
          <circle cx="9" cy="9" r="7" className="net-node-ring" />
          <circle cx="9" cy="9" r="7" className="net-fill-pos" fillOpacity="0.25" />
          <circle cx="31" cy="9" r="7" className="net-node-ring" />
          <circle cx="31" cy="9" r="7" className="net-fill-pos" fillOpacity="1" />
          <circle cx="53" cy="9" r="7" className="net-node-ring" />
          <circle cx="53" cy="9" r="7" className="net-fill-neg" fillOpacity="1" />
        </svg>
        fill = activation: stronger colour = larger |value|; blue positive, orange negative
      </span>
    </div>
  );
}

/**
 * The network diagram: one column of circles per layer and one line per weight. Line colour is the
 * weight's sign and thickness its size; a circle's fill is the neuron's current activation. Every
 * number is computed by Python (the weights arrive once, the trace in every frame); this only
 * draws them. Hovering a circle or a line shows its exact values, and dims everything unrelated.
 */
export function NetworkDiagram({
  network,
  trace,
  car,
  compact = false,
}: {
  network: NetworkMessage;
  trace: NetworkTrace;
  car: number;
  compact?: boolean;
}) {
  const [hover, setHover] = useState<Hover>(null);
  const selectedNeuron = useSelection((s) => s.neuron);
  const selectedConnection = useSelection((s) => s.connection);
  const toggleNeuron = useSelection((s) => s.toggleNeuron);
  const selected: Hover = useMemo(
    () =>
      selectedNeuron !== null
        ? { kind: "node", ...selectedNeuron }
        : selectedConnection !== null
          ? { kind: "edge", ...selectedConnection }
          : null,
    [selectedNeuron, selectedConnection],
  );
  // Hovering shows something for a moment; otherwise the selection stays emphasised.
  const focus = hover ?? selected;
  const sizes = network.layer_sizes;
  const layers = sizes.length;
  const nodes = useMemo(() => layoutNodes(sizes, FRAME), [sizes]);
  const parameters = useMemo(
    () =>
      network.weights.reduce((n, m) => n + m.length * (m[0]?.length ?? 0), 0) +
      network.biases.reduce((n, b) => n + b.length, 0),
    [network],
  );

  const describeNode = (layer: number, index: number): string => {
    const a = trace.activations[layer]?.[index] ?? 0;
    if (layer === 0) return nodeDescription(layer, index, layers, a, null, null, null);
    const z = trace.weighted_sums[layer - 1]?.[index] ?? 0;
    const bias = network.biases[layer - 1]?.[index] ?? 0;
    return nodeDescription(layer, index, layers, a, z, bias, network.activations[layer - 1] ?? null);
  };

  const describe = (h: NonNullable<Hover>): string =>
    h.kind === "node"
      ? describeNode(h.layer, h.index)
      : edgeDescription(h.layer, h.from, h.to, layers, network.weights[h.layer]?.[h.to]?.[h.from] ?? 0);
  const hoverText =
    focus === null
      ? "Hover over a circle or a line to see its exact values; click to keep it selected."
      : `${sameThing(focus, selected) ? "Selected: " : ""}${describe(focus)}`;

  return (
    <div className={compact ? "net-diagram net-compact" : "net-diagram"}>
      {!compact && (
        <p className="net-intro">
          The network of car {car + 1}: {sizes.join(" → ")} neurons, {parameters} numbers. Each line is one
          weight, each circle one neuron; a circle shows its value right now.
        </p>
      )}
      <svg
        viewBox={`0 0 ${FRAME.width} ${FRAME.height}`}
        className="net-svg"
        role="img"
        aria-label={`The network of car ${car + 1}`}
        data-testid="network-diagram"
        onMouseLeave={() => setHover(null)}
      >
        {sizes.map((size, layer) => (
          <text
            key={`title-${layer}`}
            x={nodes[layer]?.[0]?.x ?? 0}
            y={16}
            textAnchor="middle"
            className="net-caption"
          >
            {columnTitle(layer, layers, size)}
          </text>
        ))}

        <Edges network={network} nodes={nodes} hover={focus} selected={selected} onHover={setHover} />

        {nodes.map((column, layer) =>
          column.map((p, index) => {
            const a = trace.activations[layer]?.[index] ?? 0;
            const fill = nodeFill(a);
            const isSelected =
              selected !== null && selected.kind === "node" && selected.layer === layer && selected.index === index;
            return (
              <g
                key={`${layer}-${index}`}
                className={isSelected ? "net-node net-node-selected" : "net-node"}
                data-testid={`net-node-${layer}-${index}`}
                data-activation={a}
                data-selected={isSelected}
                role="button"
                aria-pressed={isSelected}
                aria-label={nodeLabel(layer, index, layers)}
                onMouseEnter={() => setHover({ kind: "node", layer, index })}
                onMouseLeave={() => setHover(null)}
                onClick={() => toggleNeuron({ layer, index }, layers)}
              >
                <circle cx={p.x} cy={p.y} r={NODE_RADIUS} className="net-node-ring" />
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={NODE_RADIUS}
                  className={fill.sign === "pos" ? "net-fill-pos" : "net-fill-neg"}
                  fillOpacity={fill.opacity}
                />
                <title>{describeNode(layer, index)}</title>
              </g>
            );
          }),
        )}

        {!compact &&
          nodes[0]?.map((p, k) => (
            <text key={`in-${k}`} x={p.x - 18} y={p.y + 4} textAnchor="end" className="net-label">
              {nodeLabel(0, k, layers)}{" "}
              <tspan className="net-value" data-testid={`net-input-${k}`} data-value={trace.activations[0]?.[k] ?? 0}>
                {num(trace.activations[0]?.[k] ?? 0)}
              </tspan>
            </text>
          ))}
        {!compact &&
          nodes[layers - 1]?.map((p, k) => (
            <text key={`out-${k}`} x={p.x + 18} y={p.y + 4} textAnchor="start" className="net-label">
              {nodeLabel(layers - 1, k, layers)}{" "}
              <tspan
                className="net-value"
                data-testid={`net-output-${k}`}
                data-value={trace.activations[layers - 1]?.[k] ?? 0}
              >
                {num(trace.activations[layers - 1]?.[k] ?? 0)}
              </tspan>
            </text>
          ))}
      </svg>
      {!compact && (
        <>
          <p className="net-readout" data-testid="network-readout">
            {hoverText}
          </p>
          <Legend />
        </>
      )}
    </div>
  );
}
