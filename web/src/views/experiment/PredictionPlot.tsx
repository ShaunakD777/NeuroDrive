import { useMemo, useState, type MouseEvent } from "react";

import type { ToyCheckpointResponse } from "../../api/generated";
import { divergingMix, extent, formatNumber, linePath, linearScale, nearestIndex, niceTicks } from "./charts";

const W = 440;
const H = 260;
const M = { left: 48, right: 14, top: 14, bottom: 40 };

/**
 * What the network believes at one checkpoint, next to the data it learns from. One input (line,
 * sine): the data as dots and the network's output over x as a line. Two inputs (XOR, spirals):
 * a colour map of the output over the plane (orange towards the lower label, blue towards the
 * higher, grey halfway) with the data points on top. Every value comes from the server.
 */
export function PredictionPlot({ toy }: { toy: ToyCheckpointResponse }) {
  return (
    <figure className="toy-figure" data-testid="prediction-plot" data-loss={toy.loss}>
      <figcaption className="toy-figure-title">
        What the network predicts (epoch {toy.unit}, loss {formatNumber(toy.loss)})
      </figcaption>
      {toy.prediction_curve ? <Curve toy={toy} curve={toy.prediction_curve} /> : <PlaneMap toy={toy} />}
    </figure>
  );
}

function Curve({ toy, curve }: { toy: ToyCheckpointResponse; curve: number[] }) {
  const [hover, setHover] = useState<number | null>(null);
  const { x, y, xTicks, yTicks } = useMemo(() => {
    const [x0, x1] = extent(toy.grid_x);
    const [lo, hi] = extent([...toy.data_y, ...curve]);
    const pad = (hi - lo) * 0.06 || 1;
    return {
      x: linearScale(x0, x1, M.left, W - M.right),
      y: linearScale(lo - pad, hi + pad, H - M.bottom, M.top),
      xTicks: niceTicks(x0, x1),
      yTicks: niceTicks(lo - pad, hi + pad),
    };
  }, [toy, curve]);

  function onMove(event: MouseEvent<SVGRectElement>) {
    const box = event.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const sx = ((event.clientX - box.left) / box.width) * W;
    const gx = linearScale(M.left, W - M.right, toy.grid_x[0], toy.grid_x[toy.grid_x.length - 1])(sx);
    setHover(nearestIndex(toy.grid_x, gx));
  }

  return (
    <>
      <svg
        className="toy-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Network output over x at epoch ${toy.unit}, drawn over ${toy.data_y.length} data points`}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="toy-grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
            <text className="toy-tick" x={M.left - 6} y={y(t) + 4} textAnchor="end">
              {formatNumber(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} className="toy-tick" x={x(t)} y={H - M.bottom + 16} textAnchor="middle">
            {formatNumber(t)}
          </text>
        ))}
        <text className="toy-axis-title" x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle">
          input x
        </text>
        {toy.data_x.map(([dx], i) => (
          <circle key={i} className="toy-dot" cx={x(dx)} cy={y(toy.data_y[i])} r={3}>
            <title>{`data: x ${formatNumber(dx)}, target ${formatNumber(toy.data_y[i])}`}</title>
          </circle>
        ))}
        <path
          className="toy-line"
          data-testid="prediction-curve"
          d={linePath(toy.grid_x.map(x), curve.map(y))}
        />
        <rect
          className="toy-hit"
          x={M.left}
          y={M.top}
          width={W - M.left - M.right}
          height={H - M.top - M.bottom}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
        {hover !== null && (
          <g className="toy-tooltip" pointerEvents="none">
            <line className="toy-crosshair" x1={x(toy.grid_x[hover])} x2={x(toy.grid_x[hover])} y1={M.top} y2={H - M.bottom} />
            <circle className="toy-hover-dot" cx={x(toy.grid_x[hover])} cy={y(curve[hover])} r={4} />
            <text
              x={x(toy.grid_x[hover]) + (x(toy.grid_x[hover]) > W - 160 ? -8 : 8)}
              y={M.top + 12}
              textAnchor={x(toy.grid_x[hover]) > W - 160 ? "end" : "start"}
            >
              {`x ${formatNumber(toy.grid_x[hover])} → network ${formatNumber(curve[hover])}`}
            </text>
          </g>
        )}
      </svg>
      <div className="toy-legend">
        <span className="toy-legend-item">
          <span className="toy-swatch-dot" /> data (targets)
        </span>
        <span className="toy-legend-item">
          <span className="toy-swatch-line" /> network's prediction
        </span>
      </div>
    </>
  );
}

/** The colour of one value on the diverging map, as a CSS fill. */
function mapFill(value: number, lo: number, hi: number): string {
  const { pole, percent } = divergingMix(value, lo, hi);
  if (pole === "mid") return "var(--chart-mid)";
  const hue = pole === "pos" ? "var(--chart-pos)" : "var(--chart-neg)";
  return `color-mix(in oklab, ${hue} ${percent}%, var(--chart-mid))`;
}

function PlaneMap({ toy }: { toy: ToyCheckpointResponse }) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  const grid = toy.prediction_map ?? [];
  const gy = toy.grid_y ?? [];
  const side = H - M.top - M.bottom;
  const left = (W - side) / 2;
  const n = toy.grid_x.length;
  const cell = side / n;
  const [lo, hi] = extent(toy.data_y);
  const x = linearScale(toy.grid_x[0], toy.grid_x[n - 1], left + cell / 2, left + side - cell / 2);
  const y = linearScale(gy[0], gy[gy.length - 1], M.top + side - cell / 2, M.top + cell / 2);
  const mid = (lo + hi) / 2;

  return (
    <>
      <svg
        className="toy-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Network output over the plane at epoch ${toy.unit}, with ${toy.data_y.length} data points`}
        onMouseLeave={() => setHover(null)}
      >
        <g data-testid="prediction-map">
          {grid.map((row, i) =>
            row.map((v, k) => (
              <rect
                key={`${i}-${k}`}
                x={left + k * cell}
                y={M.top + side - (i + 1) * cell}
                width={cell + 0.3}
                height={cell + 0.3}
                style={{ fill: mapFill(v, lo, hi) }}
                onMouseEnter={() => setHover([i, k])}
              />
            )),
          )}
        </g>
        {toy.data_x.map(([a, b], i) => (
          <circle
            key={i}
            className={toy.data_y[i] > mid ? "toy-point toy-point-pos" : "toy-point toy-point-neg"}
            cx={x(a)}
            cy={y(b)}
            r={toy.data_y.length <= 10 ? 6 : 3.5}
          >
            <title>{`data: (${formatNumber(a)}, ${formatNumber(b)}) → target ${formatNumber(toy.data_y[i])}`}</title>
          </circle>
        ))}
        <text className="toy-axis-title" x={left + side / 2} y={H - 4} textAnchor="middle">
          input x1 →
        </text>
        <text className="toy-axis-title" transform={`translate(${left - 10} ${M.top + side / 2}) rotate(-90)`} textAnchor="middle">
          input x2 →
        </text>
        {hover && (
          <text className="toy-map-readout" x={left + side + 8} y={M.top + 12}>
            <tspan x={left + side + 8}>{`x1 ${formatNumber(toy.grid_x[hover[1]])}`}</tspan>
            <tspan x={left + side + 8} dy={16}>{`x2 ${formatNumber(gy[hover[0]])}`}</tspan>
            <tspan x={left + side + 8} dy={16}>{`→ ${formatNumber(grid[hover[0]][hover[1]])}`}</tspan>
          </text>
        )}
      </svg>
      <div className="toy-legend">
        <span className="toy-legend-item">
          network output: <span className="toy-ramp" aria-hidden="true" /> {formatNumber(lo)} ·{" "}
          {formatNumber(mid)} · {formatNumber(hi)}
        </span>
        <span className="toy-legend-item">dots: data, coloured by their target</span>
      </div>
    </>
  );
}
