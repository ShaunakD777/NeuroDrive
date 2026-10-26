import { useMemo, useState, type MouseEvent } from "react";

import { formatNumber, formatPowerOfTen, linePath, linearScale, logScale, logTicks, nearestIndex } from "./charts";

const W = 440;
const H = 230;
const M = { left: 56, right: 14, top: 14, bottom: 40 };

/** Position of epoch u on a log axis: u + 1, so epoch 0 sits at the left edge. */
const shift = (u: number) => u + 1;

/**
 * The run's loss after every epoch, both axes on a log scale (the loss falls by factors of ten and
 * most of that happens in the first epochs). Dots mark the checkpoints; the selected one has a
 * cursor line. Hovering shows the epoch and loss under the pointer; clicking a dot selects it.
 */
export function LossChart({
  curve,
  checkpoints,
  selected,
  onSelect,
}: {
  curve: { unit: number[]; loss: number[] };
  checkpoints: number[];
  selected: number | null;
  onSelect: (unit: number) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const last = curve.unit.length ? curve.unit[curve.unit.length - 1] : 1;
  const { x, y, yTicks, xTicks } = useMemo(() => {
    const positive = curve.loss.filter((l) => l > 0);
    const lo = positive.length ? Math.min(...positive) : 1e-3;
    const hi = positive.length ? Math.max(...positive) : 1;
    let yTicks = logTicks(lo, hi);
    const y = logScale(yTicks[0], yTicks[yTicks.length - 1], H - M.bottom, M.top);
    if (yTicks.length > 7) yTicks = yTicks.filter((_, i) => i % Math.ceil(yTicks.length / 6) === 0);
    const x = logScale(1, shift(Math.max(last, 1)), M.left, W - M.right);
    const xTicks = [0, ...logTicks(1, Math.max(last, 1)).filter((t) => t <= last)];
    return { x, y, yTicks, xTicks };
  }, [curve, last]);

  const px = curve.unit.map((u) => x(shift(u)));
  const py = curve.loss.map((l) => y(Math.max(l, Number.MIN_VALUE)));
  const lossAt = (u: number): number | null => {
    const i = curve.unit.indexOf(u);
    return i >= 0 ? curve.loss[i] : null;
  };

  function onMove(event: MouseEvent<SVGRectElement>) {
    const box = event.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const sx = ((event.clientX - box.left) / box.width) * W;
    const t = linearScale(M.left, W - M.right, 0, Math.log10(shift(Math.max(last, 1))))(sx);
    setHover(nearestIndex(curve.unit, 10 ** t - 1));
  }

  const hoverPoint = hover !== null && curve.unit.length ? hover : null;
  return (
    <figure className="toy-figure" data-testid="loss-chart">
      <figcaption className="toy-figure-title">Loss after each epoch</figcaption>
      <svg
        className="toy-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Loss curve over ${last} epochs, from ${formatNumber(curve.loss[0] ?? 0)} to ${formatNumber(curve.loss[curve.loss.length - 1] ?? 0)}`}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="toy-grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
            <text className="toy-tick" x={M.left - 6} y={y(t) + 4} textAnchor="end">
              {formatPowerOfTen(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} className="toy-tick" x={x(shift(t))} y={H - M.bottom + 16} textAnchor="middle">
            {t}
          </text>
        ))}
        <text className="toy-axis-title" x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle">
          epoch (log scale)
        </text>
        <text className="toy-axis-title" transform={`translate(12 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
          loss (log scale)
        </text>
        <path className="toy-line" d={linePath(px, py)} data-testid="loss-line" />
        {selected !== null && (
          <line
            className="toy-cursor"
            data-testid="loss-cursor"
            data-unit={selected}
            x1={x(shift(selected))}
            x2={x(shift(selected))}
            y1={M.top}
            y2={H - M.bottom}
          />
        )}
        <rect
          className="toy-hit"
          x={M.left}
          y={M.top}
          width={W - M.left - M.right}
          height={H - M.top - M.bottom}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
        {checkpoints.map((u) => {
          const l = lossAt(u);
          if (l === null) return null;
          const isSelected = u === selected;
          return (
            <circle
              key={u}
              className={isSelected ? "toy-checkpoint toy-checkpoint-selected" : "toy-checkpoint"}
              data-testid="loss-checkpoint"
              cx={x(shift(u))}
              cy={y(Math.max(l, Number.MIN_VALUE))}
              r={isSelected ? 6 : 4.5}
              onClick={() => onSelect(u)}
            >
              <title>{`checkpoint at epoch ${u}: loss ${formatNumber(l)}`}</title>
            </circle>
          );
        })}
        {hoverPoint !== null && (
          <g className="toy-tooltip" pointerEvents="none">
            <circle className="toy-hover-dot" cx={px[hoverPoint]} cy={py[hoverPoint]} r={4} />
            <text
              x={Math.min(px[hoverPoint] + 8, W - M.right - 4)}
              y={Math.max(py[hoverPoint] - 8, M.top + 10)}
              textAnchor={px[hoverPoint] > W - 150 ? "end" : "start"}
            >
              {`epoch ${curve.unit[hoverPoint]}: loss ${formatNumber(curve.loss[hoverPoint])}`}
            </text>
          </g>
        )}
      </svg>
    </figure>
  );
}
