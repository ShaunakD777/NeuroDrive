import { useMemo, useState, type CSSProperties, type MouseEvent } from "react";

import type { CompareResponse } from "../../api/generated";
import {
  bandPath,
  extent,
  formatNumber,
  formatPowerOfTen,
  linePath,
  linearScale,
  logScale,
  logTicks,
  nearestIndex,
  shortName,
} from "./charts";

const W = 640;
const H = 300;
const M = { left: 58, right: 120, top: 14, bottom: 40 };
const LABEL_GAP = 14; // minimum vertical distance between two end-of-line labels, in px

/** An end-of-line label: long names keep their end, where sweeps of one task differ. */
const endLabel = (name: string) => (name.length > 16 ? `…${name.slice(-15)}` : name);

/** Position of epoch u on a log axis: u + 1, so epoch 0 sits at the left edge. */
const shift = (u: number) => u + 1;

/** The CSS custom property that carries a group's series colour (slots 0..7 → --series-1..8). */
const seriesStyle = (slot: number) =>
  ({ "--series": `var(--series-${slot + 1})` }) as CSSProperties;

/**
 * Overlaid learning curves of several groups of runs (task 4.7b, ADR-033): for each group the IQR
 * band (Q1 to Q3 across its runs at every epoch), the median line and each run's own faint line.
 * Both axes are logarithmic. Everything comes from the server; a null median or quartile means
 * +infinity (diverged runs) and is drawn at the top edge. Hovering shows each group's median and
 * IQR at that epoch.
 */
export function CompareChart({
  data,
  slots,
  metricLabel,
}: {
  data: CompareResponse;
  slots: Record<string, number>;
  metricLabel: string;
}) {
  const [hover, setHover] = useState<number | null>(null);

  const geometry = useMemo(() => {
    const finite: number[] = [];
    for (const g of data.groups)
      for (const series of [g.median, g.q1, g.q3])
        for (const v of series) if (v !== null && v > 0) finite.push(v);
    for (const r of data.runs)
      for (const v of r.values) if (v !== null && v > 0) finite.push(v);
    // A loop, not Math.min(...finite): spreading ~100 000 values (20 runs of 5000 epochs, Study
    // S3) into one call overflows the call stack.
    const [lo, hi] = finite.length ? extent(finite) : [1e-3, 1];
    let yTicks = logTicks(lo, hi);
    const yTop = yTicks[yTicks.length - 1];
    const y = logScale(yTicks[0], yTop, H - M.bottom, M.top);
    if (yTicks.length > 7)
      yTicks = yTicks.filter((_, i) => i % Math.ceil(yTicks.length / 6) === 0);
    const units = data.groups.flatMap((g) => g.unit);
    const [first, last] = units.length ? extent(units) : [0, 1];
    const x = logScale(
      shift(first),
      shift(Math.max(last, first + 1)),
      M.left,
      W - M.right,
    );
    const xTicks = [
      first,
      ...logTicks(1, Math.max(last, 1)).filter((t) => t > first && t <= last),
    ];
    // +infinity (null) and anything off the top is drawn at the top edge.
    const py = (v: number | null) =>
      v === null ? M.top : Math.max(M.top, y(Math.max(v, Number.MIN_VALUE)));
    const allUnits = [...new Set(units)].sort((a, b) => a - b);
    return { x, py, yTicks, xTicks, first, last, allUnits, y };
  }, [data]);

  const { x, py, yTicks, xTicks, first, last, allUnits, y } = geometry;

  const groups = data.groups.map((g) => {
    const xs = g.unit.map((u) => x(shift(u)));
    return {
      group: g,
      slot: slots[g.group_id] ?? 0,
      xs,
      band: bandPath(xs, g.q1.map(py), g.q3.map(py)),
      median: linePath(xs, g.median.map(py)),
      end: { x: xs[xs.length - 1], y: py(g.median[g.median.length - 1]) },
    };
  });

  // End-of-line labels (identity is never colour alone): nudged apart so they do not overlap.
  const labels = [...groups]
    .sort((a, b) => a.end.y - b.end.y)
    .map((g) => ({ id: g.group.group_id, y: g.end.y }));
  for (let i = 1; i < labels.length; i++)
    labels[i].y = Math.max(labels[i].y, labels[i - 1].y + LABEL_GAP);
  const labelY = Object.fromEntries(labels.map((l) => [l.id, l.y]));

  function onMove(event: MouseEvent<SVGRectElement>) {
    const box = event.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const sx = ((event.clientX - box.left) / box.width) * W;
    const t = linearScale(
      M.left,
      W - M.right,
      Math.log10(shift(first)),
      Math.log10(shift(Math.max(last, first + 1))),
    )(sx);
    setHover(
      allUnits.length ? allUnits[nearestIndex(allUnits, 10 ** t - 1)] : null,
    );
  }

  const show = (v: number | null) => (v === null ? "∞" : formatNumber(v));
  const hoverRows =
    hover === null
      ? []
      : groups
          .map((g) => ({ g, i: g.group.unit.indexOf(hover) }))
          .filter(({ i }) => i >= 0)
          .map(({ g, i }) => ({
            id: g.group.group_id,
            slot: g.slot,
            name: shortName(g.group.group_id),
            value: `${show(g.group.median[i])} [${show(g.group.q1[i])}, ${show(g.group.q3[i])}]`,
          }));
  // The read-out is an HTML box over the chart (not SVG text), so it keeps its normal size and
  // stays sharp however much the chart is scaled; it sits right of the crosshair, or left of it
  // in the chart's right half.
  const hoverX = hover === null ? 0 : x(shift(hover));
  const hoverLeft = hoverX > (M.left + W - M.right) / 2;

  return (
    <figure className="toy-figure" data-testid="compare-chart">
      <figcaption className="toy-figure-title">
        {metricLabel} after each epoch: median line, IQR band, one thin line per
        run
      </figcaption>
      <div className="cmp-plot">
      <svg
        className="toy-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${metricLabel} of ${data.runs.length} runs in ${data.groups.length} groups, epochs ${first} to ${last}`}
      >
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line
              className="toy-grid"
              x1={M.left}
              x2={W - M.right}
              y1={y(t)}
              y2={y(t)}
            />
            <text
              className="toy-tick"
              x={M.left - 6}
              y={y(t) + 4}
              textAnchor="end"
            >
              {formatPowerOfTen(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text
            key={`x${t}`}
            className="toy-tick"
            x={x(shift(t))}
            y={H - M.bottom + 16}
            textAnchor="middle"
          >
            {t}
          </text>
        ))}
        <text
          className="toy-axis-title"
          x={(M.left + W - M.right) / 2}
          y={H - 4}
          textAnchor="middle"
        >
          epoch (log scale)
        </text>
        <text
          className="toy-axis-title"
          transform={`translate(12 ${(M.top + H - M.bottom) / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {metricLabel} (log scale)
        </text>

        {groups.map((g) => (
          <path
            key={`band-${g.group.group_id}`}
            className="cmp-band"
            style={seriesStyle(g.slot)}
            d={g.band}
            data-testid="compare-band"
            data-group-id={g.group.group_id}
          />
        ))}
        {data.runs.map((r) => {
          const slot = slots[r.group_id] ?? 0;
          const n = r.values.findIndex((v) => v === null);
          const end = n === -1 ? r.values.length : n; // a diverged run's line stops at its last finite value
          const xs = r.unit.slice(0, end).map((u) => x(shift(u)));
          const ys = r.values.slice(0, end).map((v) => py(v));
          return (
            <path
              key={`run-${r.run_id}`}
              className="cmp-run"
              style={seriesStyle(slot)}
              d={linePath(xs, ys)}
              data-testid="compare-run-line"
              data-run-id={r.run_id}
            >
              <title>{`${shortName(r.run_id)}: final ${show(r.final)}${r.diverged ? " (diverged)" : ""}`}</title>
            </path>
          );
        })}
        {groups.map((g) => (
          <path
            key={`median-${g.group.group_id}`}
            className="cmp-median"
            style={seriesStyle(g.slot)}
            d={g.median}
            data-testid="compare-median"
            data-group-id={g.group.group_id}
          />
        ))}
        {groups.map((g) => (
          <g key={`label-${g.group.group_id}`} style={seriesStyle(g.slot)}>
            <line
              className="cmp-label-mark"
              x1={g.end.x + 4}
              x2={g.end.x + 14}
              y1={labelY[g.group.group_id]}
              y2={labelY[g.group.group_id]}
            />
            <text
              className="cmp-label"
              x={g.end.x + 18}
              y={labelY[g.group.group_id] + 4}
            >
              {endLabel(shortName(g.group.group_id))}
            </text>
          </g>
        ))}

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
          <line
            className="toy-crosshair"
            pointerEvents="none"
            data-testid="compare-crosshair"
            x1={hoverX}
            x2={hoverX}
            y1={M.top}
            y2={H - M.bottom}
          />
        )}
      </svg>
      {hover !== null && (
        <div
          className={hoverLeft ? "cmp-hover cmp-hover-left" : "cmp-hover"}
          data-testid="compare-hover"
          style={{ left: `${(hoverX / W) * 100}%`, top: `${(M.top / H) * 100}%` }}
        >
          <div className="cmp-hover-title">epoch {hover}: median [Q1, Q3]</div>
          {hoverRows.map((row) => (
            <div key={row.id} className="cmp-hover-row" style={seriesStyle(row.slot)}>
              <span className="cmp-swatch" />
              <span className="cmp-hover-name">{row.name}</span>
              <span className="cmp-hover-value">{row.value}</span>
            </div>
          ))}
        </div>
      )}
      </div>
      <div className="toy-legend" data-testid="compare-legend">
        {groups.map((g) => (
          <span
            key={g.group.group_id}
            className="toy-legend-item"
            style={seriesStyle(g.slot)}
          >
            <span className="cmp-swatch" />
            {shortName(g.group.group_id)} ({g.group.run_ids.length} run
            {g.group.run_ids.length === 1 ? "" : "s"})
          </span>
        ))}
        <span className="toy-legend-item">
          <span className="cmp-swatch-band" /> IQR band (Q1 to Q3)
        </span>
      </div>
    </figure>
  );
}
