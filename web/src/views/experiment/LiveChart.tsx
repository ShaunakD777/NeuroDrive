import { useMemo, useState, type MouseEvent } from "react";

import { formatNumber, formatPowerOfTen, linePath, linearScale, logScale, logTicks, nearestIndex, niceTicks } from "./charts";

const W = 520;
const H = 250;
const M = { left: 56, right: 14, top: 14, bottom: 40 };
const MAX_DOTS = 120; // a dot per unit while there are few; beyond that, only the line

/**
 * One metric of a training run against its unit, drawn while the run trains (task 5.8). The unit
 * axis runs to the run's unit budget when it has one, so the line visibly grows towards the end.
 * Losses use a log axis (they fall by factors of ten); everything else, such as fitness, a linear
 * one. Ringed dots mark the units whose checkpoint has an exam card. Hovering shows the value.
 */
export function LiveChart({
  unit,
  value,
  metric,
  budgetUnits,
  examUnits,
}: {
  unit: number[];
  value: number[];
  metric: string;
  budgetUnits: number | null;
  examUnits: number[];
}) {
  const [hover, setHover] = useState<number | null>(null);
  const log = metric.includes("loss") && value.length > 0 && value.every((v) => v > 0);
  const lastUnit = unit.length ? unit[unit.length - 1] : 0;
  const xEnd = Math.max(budgetUnits ?? 0, lastUnit, 1);
  const { x, y, xTicks, yTicks } = useMemo(() => {
    const x = linearScale(0, xEnd, M.left, W - M.right);
    const xTicks = niceTicks(0, xEnd, 6).filter((t) => Number.isInteger(t));
    if (log) {
      const yTicks = logTicks(Math.min(...value), Math.max(...value));
      return { x, xTicks, yTicks, y: logScale(yTicks[0], yTicks[yTicks.length - 1], H - M.bottom, M.top) };
    }
    // From 0 (or the lowest value, if negative) up to the first round tick above the highest
    // value, so the line never runs along the top edge.
    let lo = value.length ? Math.min(...value, 0) : 0;
    let hi = value.length ? Math.max(...value) : 1;
    if (hi <= lo) hi = lo + 1;
    const rough = niceTicks(lo, hi, 5);
    const step = rough.length > 1 ? rough[1] - rough[0] : hi - lo;
    lo = Math.floor(lo / step) * step;
    hi = Math.ceil(hi / step + 1e-9) * step;
    return { x, xTicks, yTicks: niceTicks(lo, hi, 5), y: linearScale(lo, hi, H - M.bottom, M.top) };
  }, [log, value, xEnd]);

  const px = unit.map(x);
  const py = value.map(y);
  const valueAt = new Map(unit.map((u, i) => [u, value[i]]));

  function onMove(event: MouseEvent<SVGRectElement>) {
    if (!unit.length) return;
    const box = event.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const sx = ((event.clientX - box.left) / box.width) * W;
    setHover(nearestIndex(unit, linearScale(M.left, W - M.right, 0, xEnd)(sx)));
  }

  const h = hover !== null && hover < unit.length ? hover : null;
  const label = metric.replaceAll("_", " ");
  return (
    <figure className="toy-figure live-figure" data-testid="live-chart" data-points={unit.length} data-metric={metric}>
      <figcaption className="toy-figure-title">
        {label} after each unit
        {unit.length > 0 && (
          <span className="live-latest" data-testid="live-latest">
            {" "}
            · unit {lastUnit}: {formatNumber(value[value.length - 1])}
          </span>
        )}
      </figcaption>
      <svg
        className="toy-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${label} over ${unit.length} units so far${budgetUnits ? ` of ${budgetUnits}` : ""}`}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line className="toy-grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
            <text className="toy-tick" x={M.left - 6} y={y(t) + 4} textAnchor="end">
              {log ? formatPowerOfTen(t) : formatNumber(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} className="toy-tick" x={x(t)} y={H - M.bottom + 16} textAnchor="middle">
            {t}
          </text>
        ))}
        <text className="toy-axis-title" x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle">
          unit{budgetUnits ? ` (budget ${budgetUnits})` : ""}
        </text>
        <text className="toy-axis-title" transform={`translate(12 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
          {label}
          {log ? " (log scale)" : ""}
        </text>
        {unit.length > 1 && <path className="toy-line" d={linePath(px, py)} data-testid="live-line" />}
        {unit.length <= MAX_DOTS &&
          px.map((cx, i) => <circle key={unit[i]} className="live-dot" cx={cx} cy={py[i]} r={2.5} data-testid="live-point" />)}
        {examUnits.map((u) => {
          const v = valueAt.get(u);
          if (v === undefined) return null;
          return (
            <circle key={`e${u}`} className="toy-checkpoint" cx={x(u)} cy={y(v)} r={4.5} data-testid="live-exam-point">
              <title>{`unit ${u} was examined: ${label} ${formatNumber(v)}`}</title>
            </circle>
          );
        })}
        <rect
          className="toy-hit"
          x={M.left}
          y={M.top}
          width={W - M.left - M.right}
          height={H - M.top - M.bottom}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
        {h !== null && (
          <g className="toy-tooltip" pointerEvents="none">
            <circle className="toy-hover-dot" cx={px[h]} cy={py[h]} r={4} />
            <text
              x={Math.min(px[h] + 8, W - M.right - 4)}
              y={Math.max(py[h] - 8, M.top + 10)}
              textAnchor={px[h] > W - 170 ? "end" : "start"}
            >
              {`unit ${unit[h]}: ${formatNumber(value[h])}`}
            </text>
          </g>
        )}
      </svg>
    </figure>
  );
}
