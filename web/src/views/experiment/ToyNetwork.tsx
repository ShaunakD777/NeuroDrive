import { useMemo, useState } from "react";

import type { ToyCheckpointResponse } from "../../api/generated";
import { layoutNodes, num } from "../network/networkLayout";
import { formatNumber, glowStroke, stepDirection } from "./charts";

const FRAME = { width: 420, height: 260, marginX: 44, marginY: 26 };

interface Edge {
  layer: number;
  from: number;
  to: number;
}

function edgeName(e: Edge): string {
  return `W${e.layer + 1}[${e.to},${e.from}]`;
}

/** Column caption; short forms ("in", "h2", "out") when many columns share the width. */
function columnName(layer: number, layers: number): string {
  const short = layers > 4;
  if (layer === 0) return short ? "in" : "inputs";
  if (layer === layers - 1) return short ? "out" : "output";
  return short ? `h${layer}` : `hidden ${layer}`;
}

/**
 * The network of one toy checkpoint, with every connection glowing by the size of its loss
 * gradient (task 3.8): brightness and thickness from the server's glow |dL/dw| / max|dL/dW|,
 * colour from the direction the next gradient-descent step moves the weight (blue: larger,
 * orange: smaller). Hover a connection to read its weight and gradient. The browser computes
 * nothing about the network; all numbers come from the server.
 */
export function ToyNetwork({ toy }: { toy: ToyCheckpointResponse }) {
  const [hover, setHover] = useState<Edge | null>(null);
  const sizes = toy.network.layer_sizes;
  const nodes = useMemo(() => layoutNodes(sizes, FRAME), [sizes]);
  const edges = useMemo(() => {
    const list: Edge[] = [];
    toy.glow.forEach((matrix, layer) =>
      matrix.forEach((row, to) => row.forEach((_, from) => list.push({ layer, from, to }))),
    );
    // Faint edges first, so the bright ones are drawn on top.
    return list.sort((a, b) => toy.glow[a.layer][a.to][a.from] - toy.glow[b.layer][b.to][b.from]);
  }, [toy]);

  const readout = hover
    ? (() => {
        const w = toy.network.weights[hover.layer][hover.to][hover.from];
        const g = toy.weight_gradients[hover.layer][hover.to][hover.from];
        const glow = toy.glow[hover.layer][hover.to][hover.from];
        const way = stepDirection(g) === "pos" ? "grow" : stepDirection(g) === "neg" ? "shrink" : "stay";
        return `${edgeName(hover)}: weight ${num(w, 3)}, gradient ${formatNumber(g)} (glow ${glow.toFixed(2)}): the next step makes it ${way}.`;
      })()
    : "Hover a connection to read its weight and gradient.";

  return (
    <figure className="toy-figure" data-testid="toy-network">
      <figcaption className="toy-figure-title">Where learning pushes (epoch {toy.unit})</figcaption>
      <svg
        className="toy-svg"
        viewBox={`0 0 ${FRAME.width} ${FRAME.height}`}
        role="img"
        aria-label={`Network ${sizes.join("-")} at epoch ${toy.unit}; connections glow by gradient size`}
        onMouseLeave={() => setHover(null)}
      >
        {sizes.map((_, layer) => (
          <text key={layer} className="toy-caption" x={nodes[layer][0].x} y={12} textAnchor="middle">
            {columnName(layer, sizes.length)}
          </text>
        ))}
        <g>
          {edges.map((e) => {
            const a = nodes[e.layer][e.from];
            const b = nodes[e.layer + 1][e.to];
            const glow = toy.glow[e.layer][e.to][e.from];
            const { opacity, width } = glowStroke(glow);
            const dir = stepDirection(toy.weight_gradients[e.layer][e.to][e.from]);
            const isHover = hover !== null && edgeName(hover) === edgeName(e);
            return (
              <g key={edgeName(e)}>
                <line
                  className={`toy-edge toy-edge-${dir}${isHover ? " toy-edge-hover" : ""}`}
                  data-testid="toy-edge"
                  data-name={edgeName(e)}
                  data-glow={glow}
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  strokeWidth={width}
                  strokeOpacity={opacity}
                />
                <line
                  className="toy-edge-hit"
                  x1={a.x}
                  y1={a.y}
                  x2={b.x}
                  y2={b.y}
                  onMouseEnter={() => setHover(e)}
                />
              </g>
            );
          })}
        </g>
        {nodes.map((column, layer) =>
          column.map((p, i) => (
            <circle key={`${layer}-${i}`} className="toy-node" cx={p.x} cy={p.y} r={7} />
          )),
        )}
      </svg>
      <p className="toy-readout" data-testid="toy-network-readout">
        {readout}
      </p>
      <div className="toy-legend">
        <span className="toy-legend-item">
          <span className="toy-swatch toy-swatch-pos" /> next step makes the weight larger
        </span>
        <span className="toy-legend-item">
          <span className="toy-swatch toy-swatch-neg" /> smaller
        </span>
        <span className="toy-legend-item" data-testid="toy-max-gradient">
          brightest = largest |dL/dw| = {formatNumber(toy.max_gradient)}
        </span>
      </div>
    </figure>
  );
}
