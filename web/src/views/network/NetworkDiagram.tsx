import { memo, useMemo, useState } from "react";

import type { NetworkMessage, NetworkTrace } from "../../api/generated";
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

type Hover =
  | { kind: "node"; layer: number; index: number }
  | { kind: "edge"; layer: number; from: number; to: number }
  | null;

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
  hover: Hover;
  onHover: (h: Hover) => void;
}

/** The lines. They only change when the network or the hover does, not on every frame. */
const Edges = memo(function Edges({ network, nodes, hover, onHover }: EdgesProps) {
  const edges = useMemo(() => collectEdges(network), [network]);
  const layers = network.layer_sizes.length;
  return (
    <g className="net-edges">
      {edges.map((e) => {
        const a = nodes[e.layer]?.[e.from];
        const b = nodes[e.layer + 1]?.[e.to];
        if (!a || !b) return null;
        const dimmed = !isRelated(hover, e);
        return (
          <line
            key={`${e.layer}-${e.from}-${e.to}`}
            x1={a.x}
            y1={a.y}
            x2={b.x}
            y2={b.y}
            className={`net-edge net-edge-${signOf(e.weight)}${dimmed ? " net-dim" : ""}`}
            strokeWidth={edgeWidth(e.weight)}
            strokeOpacity={edgeOpacity(e.weight)}
            data-testid={`net-edge-${e.layer}-${e.from}-${e.to}`}
            data-weight={e.weight}
            onMouseEnter={() => onHover({ kind: "edge", layer: e.layer, from: e.from, to: e.to })}
            onMouseLeave={() => onHover(null)}
          >
            <title>{edgeDescription(e.layer, e.from, e.to, layers, e.weight)}</title>
          </line>
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

  const hoverText =
    hover === null
      ? "Hover over a circle or a line to see its exact values."
      : hover.kind === "node"
        ? describeNode(hover.layer, hover.index)
        : edgeDescription(
            hover.layer,
            hover.from,
            hover.to,
            layers,
            network.weights[hover.layer]?.[hover.to]?.[hover.from] ?? 0,
          );

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

        <Edges network={network} nodes={nodes} hover={hover} onHover={setHover} />

        {nodes.map((column, layer) =>
          column.map((p, index) => {
            const a = trace.activations[layer]?.[index] ?? 0;
            const fill = nodeFill(a);
            return (
              <g
                key={`${layer}-${index}`}
                className="net-node"
                data-testid={`net-node-${layer}-${index}`}
                data-activation={a}
                onMouseEnter={() => setHover({ kind: "node", layer, index })}
                onMouseLeave={() => setHover(null)}
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
