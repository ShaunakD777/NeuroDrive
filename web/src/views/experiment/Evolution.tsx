import { useEffect, useMemo, useState, type CSSProperties, type MouseEvent, type ReactNode } from "react";

import type { EvolutionView } from "../../api/generated";
import { fetchEvolution } from "../../api/rest";
import { ChartHover } from "./ChartHover";
import { formatNumber, linePath, linearScale, nearestIndex, niceTicks, type Scale } from "./charts";
import {
  bandLabel,
  filledBands,
  FITNESS_LINES,
  panelGeneration,
  shadeColour,
  shadeLegend,
  shadeOf,
  spreadLabels,
  topEdge,
  type StatKey,
} from "./evolutionCharts";

const W = 640;
const H = 240;
const M = { left: 58, right: 70, top: 14, bottom: 40 };
const LABEL_GAP = 14;

const seriesStyle = (slot: number) => ({ "--series": `var(--series-${slot + 1})` }) as CSSProperties;

/** 0 up to the first round tick above `hi`, so a line never runs along the top edge. */
function valueAxis(hi: number): { ticks: number[]; y: Scale } {
  const top = hi > 0 ? hi : 1;
  const rough = niceTicks(0, top, 4);
  const step = rough.length > 1 ? rough[1] - rough[0] : top;
  const end = Math.ceil(top / step + 1e-9) * step;
  return { ticks: niceTicks(0, end, 4), y: linearScale(0, end, H - M.bottom, M.top) };
}

/** Where the pointer is along the generation axis (viewBox units), from a mouse event. */
function pointerX(event: MouseEvent<SVGRectElement>): number {
  const box = event.currentTarget.ownerSVGElement!.getBoundingClientRect();
  return ((event.clientX - box.left) / box.width) * W;
}

function pointerY(event: MouseEvent<SVGRectElement>): number {
  const box = event.currentTarget.ownerSVGElement!.getBoundingClientRect();
  return ((event.clientY - box.top) / box.height) * H;
}

/** Axes, grid and the generation cursor shared by the three charts over generations. */
function GenerationFrame({
  x,
  units,
  yTicks,
  y,
  yTitle,
  cursor,
  children,
}: {
  x: Scale;
  units: number[];
  yTicks: number[];
  y: Scale;
  yTitle: string;
  cursor: number | null;
  children: ReactNode;
}) {
  const first = units[0];
  const last = units[units.length - 1];
  const xTicks = niceTicks(first, last, 6).filter((t) => Number.isInteger(t));
  return (
    <>
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line className="toy-grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
          <text className="toy-tick" x={M.left - 6} y={y(t) + 4} textAnchor="end">
            {formatNumber(t)}
          </text>
        </g>
      ))}
      {(xTicks.length ? xTicks : [first]).map((t) => (
        <text key={`x${t}`} className="toy-tick" x={x(t)} y={H - M.bottom + 16} textAnchor="middle">
          {t}
        </text>
      ))}
      <text className="toy-axis-title" x={(M.left + W - M.right) / 2} y={H - 4} textAnchor="middle">
        generation
      </text>
      <text className="toy-axis-title" transform={`translate(12 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
        {yTitle}
      </text>
      {children}
      {cursor !== null && (
        <line className="toy-cursor" data-testid="evolution-cursor" x1={x(cursor)} x2={x(cursor)} y1={M.top} y2={H - M.bottom} />
      )}
    </>
  );
}

/**
 * The evolution panel of a GA run (task 7.7, ADR-058): the generation's best, median, mean and
 * worst fitness, its whole fitness distribution as a heat map (generations across, fitness bands
 * up, shade = how many networks), the chosen generation's histogram, and the diversity. Every
 * number comes from `GET /api/runs/{id}/evolution`; it is fetched again whenever the run's metrics
 * history gains a line, so a run still training grows here. The chosen generation follows the
 * timeline slider; clicking any chart chooses another one.
 */
export function Evolution({
  runId,
  sliderUnit,
  generations,
  onRace,
  racing,
}: {
  runId: string;
  /** The timeline slider's checkpoint unit (null while there is none). */
  sliderUnit: number | null;
  /** How many metrics lines the run has so far (a change fetches the charts again). */
  generations: number;
  onRace: (generation: number) => void;
  /** The generation racing in the Environment view, if it is one of this run's. */
  racing: number | null;
}) {
  const [view, setView] = useState<EvolutionView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<number | null>(null);

  useEffect(() => {
    let live = true;
    fetchEvolution(runId)
      .then((answer) => {
        if (!live) return;
        setView(answer);
        setError(null);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [runId, generations]);
  useEffect(() => setPicked(null), [sliderUnit, runId]);

  if (error) return <p className="toylab-error">Could not load the evolution charts: {error}</p>;
  if (view === null || view.run_id !== runId) return <p className="hint">Loading the evolution charts…</p>;
  if (view.unit.length === 0)
    return (
      <p className="hint" data-testid="evolution-empty">
        No generation has been scored yet.
      </p>
    );

  const shown = picked !== null && view.unit.includes(picked) ? picked : panelGeneration(sliderUnit, view.unit)!;
  const g = view.unit.indexOf(shown);
  return (
    <section className="evolution" data-testid="evolution" data-generations={view.unit.length} data-generation={shown}>
      <h3 className="evolution-title">
        Evolution: {view.unit.length} generation{view.unit.length === 1 ? "" : "s"} of {view.population} networks
        {view.elite > 0 ? `, ${view.elite} elite${view.elite === 1 ? "" : "s"} kept each time` : ""}
      </h3>
      <div className="evolution-actions">
        <span data-testid="evolution-generation">
          Generation {shown}: best {formatNumber(view.best[g])}, median {formatNumber(view.median[g])}, worst{" "}
          {formatNumber(view.worst[g])}
          {view.finishers[g] !== null ? `, ${view.finishers[g]} of ${view.population} finish every start` : ""}
        </span>
        <button type="button" data-testid="evolution-race" disabled={racing === shown} onClick={() => onRace(shown)}>
          {racing === shown ? `Generation ${shown} races in the Environment view` : `Race generation ${shown}`}
        </button>
      </div>
      <p className="hint">Click any chart to choose a generation; moving the timeline slider chooses its generation again.</p>
      <div className="evolution-grid">
        <FitnessChart view={view} cursor={shown} onPick={setPicked} />
        <DistributionChart view={view} cursor={shown} onPick={setPicked} />
        <Histogram view={view} g={g} />
        <DiversityChart view={view} cursor={shown} onPick={setPicked} />
      </div>
      <EvolutionTable view={view} />
    </section>
  );
}

type ChartProps = { view: EvolutionView; cursor: number; onPick: (generation: number) => void };

function generationScale(units: number[]): Scale {
  return linearScale(units[0] - 0.5, units[units.length - 1] + 0.5, M.left, W - M.right);
}

/** Best, median, mean and worst fitness per generation (hover: every statistic of that generation). */
function FitnessChart({ view, cursor, onPick }: ChartProps) {
  const [hover, setHover] = useState<number | null>(null);
  const units = view.unit;
  const x = useMemo(() => generationScale(units), [units]);
  // the same fitness axis as the heat map (its top band is at or above the best), so the grids line up
  const { ticks, y } = useMemo(() => valueAxis(topEdge(view)), [view]);
  const toUnit = linearScale(M.left, W - M.right, units[0] - 0.5, units[units.length - 1] + 0.5);
  const px = units.map(x);
  const lastX = px[px.length - 1];
  const labels = spreadLabels(
    FITNESS_LINES.map((line) => ({ ...line, y: y(view[line.key][units.length - 1]) })),
    LABEL_GAP,
    M.top + 4,
    H - M.bottom,
  );
  const h = hover !== null && hover < units.length ? hover : null;
  const rows: [string, number | null, number | null][] =
    h === null
      ? []
      : [
          ["best so far", view.best_so_far[h], null],
          ["best", view.best[h], 0],
          ["upper quartile", view.q3[h], null],
          ["median", view.median[h], 1],
          ["mean", view.mean[h], 2],
          ["lower quartile", view.q1[h], null],
          ["worst", view.worst[h], 3],
        ];
  return (
    <figure className="toy-figure" data-testid="evolution-fitness" data-points={units.length}>
      <figcaption className="toy-figure-title">Fitness F of each generation's networks</figcaption>
      <div className="cmp-plot">
        <svg className="toy-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Best, median, mean and worst fitness over ${units.length} generations`}>
          <GenerationFrame x={x} units={units} yTicks={ticks} y={y} yTitle="fitness F" cursor={cursor}>
            {FITNESS_LINES.map((line) => {
              const py = view[line.key as StatKey].map(y);
              return units.length > 1 ? (
                <path key={line.key} className="cmp-median" style={seriesStyle(line.slot)} d={linePath(px, py)} data-testid={`evolution-line-${line.key}`} />
              ) : (
                <circle key={line.key} className="evolution-dot" style={seriesStyle(line.slot)} cx={px[0]} cy={py[0]} r={4} />
              );
            })}
            {labels.map((l) => (
              <g key={`l${l.key}`} style={seriesStyle(l.slot)}>
                <line className="cmp-label-mark" x1={lastX + 4} x2={lastX + 12} y1={l.y} y2={l.y} />
                <text className="cmp-label" x={lastX + 15} y={l.y + 4}>
                  {l.label}
                </text>
              </g>
            ))}
          </GenerationFrame>
          {h !== null && <line className="toy-crosshair" x1={px[h]} x2={px[h]} y1={M.top} y2={H - M.bottom} pointerEvents="none" />}
          {h !== null &&
            FITNESS_LINES.map((line) => (
              <circle key={`h${line.key}`} className="evolution-dot" style={seriesStyle(line.slot)} cx={px[h]} cy={y(view[line.key][h])} r={4} pointerEvents="none" />
            ))}
          <rect
            className="toy-hit evolution-hit"
            data-testid="evolution-fitness-hit"
            x={M.left}
            y={M.top}
            width={W - M.left - M.right}
            height={H - M.top - M.bottom}
            onMouseMove={(e) => setHover(nearestIndex(units, toUnit(pointerX(e))))}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => onPick(units[nearestIndex(units, toUnit(pointerX(e)))])}
          />
        </svg>
        {h !== null && (
          <ChartHover x={px[h]} y={M.top} width={W} height={H} testId="evolution-fitness-hover">
            <div className="cmp-hover-title">generation {units[h]}</div>
            {rows.map(([name, value, slot]) => (
              <div key={name} className="cmp-hover-row" style={slot === null ? undefined : seriesStyle(slot)}>
                <span className={slot === null ? "cmp-swatch cmp-swatch-off" : "cmp-swatch"} />
                <span>{name}</span>
                <span className="cmp-hover-value">{value === null ? "–" : formatNumber(value)}</span>
              </div>
            ))}
            {view.finishers[h] !== null && (
              <div className="cmp-hover-row">
                <span className="cmp-swatch cmp-swatch-off" />
                <span>finish every start</span>
                <span className="cmp-hover-value">
                  {view.finishers[h]} of {view.population}
                </span>
              </div>
            )}
          </ChartHover>
        )}
      </div>
      <div className="toy-legend" data-testid="evolution-fitness-legend">
        {FITNESS_LINES.map((line) => (
          <span key={line.key} className="toy-legend-item" style={seriesStyle(line.slot)}>
            <span className="cmp-swatch" /> {line.label}
          </span>
        ))}
      </div>
    </figure>
  );
}

/** The heat map: one column per generation, one row per fitness band, shaded by how many networks it holds. */
function DistributionChart({ view, cursor, onPick }: ChartProps) {
  const [hover, setHover] = useState<{ g: number; k: number } | null>(null);
  const units = view.unit;
  const x = useMemo(() => generationScale(units), [units]);
  const top = topEdge(view);
  const { ticks, y } = valueAxis(top);
  const toUnit = linearScale(M.left, W - M.right, units[0] - 0.5, units[units.length - 1] + 0.5);
  const colWidth = x(units[0] + 0.5) - x(units[0] - 0.5);
  const gap = colWidth > 6 ? 1 : 0; // a thin surface gap between columns while they are wide enough
  const cells: ReactNode[] = [];
  let drawn = 0;
  units.forEach((u, g) => {
    for (const [k, count] of filledBands(view, g)) {
      if (view.edges[k] >= top) continue;
      drawn += count;
      const y0 = y(Math.min(view.edges[k + 1], top));
      const y1 = y(view.edges[k]);
      cells.push(
        <rect
          key={`${u}:${k}`}
          className="evolution-cell"
          data-testid="evolution-cell"
          data-count={count}
          x={x(u - 0.5) + gap}
          y={y0}
          width={Math.max(colWidth - 2 * gap, 0.5)}
          height={Math.max(y1 - y0 - 0.5, 0.5)}
          style={{ fill: shadeColour(shadeOf(count / view.population)) }}
        />,
      );
    }
  });
  const gi = units.indexOf(cursor);
  const hk = hover === null ? null : (view.counts[hover.g]?.[hover.k] ?? 0);
  function locate(e: MouseEvent<SVGRectElement>) {
    const g = nearestIndex(units, toUnit(pointerX(e)));
    const f = linearScale(y(0), y(1), 0, 1)(pointerY(e)); // pixels back to fitness
    let k = view.edges.findIndex((edge, i) => i < view.edges.length - 1 && f >= edge && f < view.edges[i + 1]);
    if (k < 0) k = 0;
    return { g, k };
  }
  return (
    <figure className="toy-figure" data-testid="evolution-distribution" data-networks={drawn}>
      <figcaption className="toy-figure-title">Fitness distribution: networks per band of {view.band_width}</figcaption>
      <div className="cmp-plot">
        <svg className="toy-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Heat map of the fitness of every network in ${units.length} generations`}>
          <GenerationFrame x={x} units={units} yTicks={ticks} y={y} yTitle="fitness F" cursor={null}>
            {cells}
            {gi >= 0 && (
              <rect className="evolution-column" data-testid="evolution-column" x={x(cursor - 0.5)} y={M.top} width={colWidth} height={H - M.top - M.bottom} />
            )}
          </GenerationFrame>
          <rect
            className="toy-hit evolution-hit"
            data-testid="evolution-distribution-hit"
            x={M.left}
            y={M.top}
            width={W - M.left - M.right}
            height={H - M.top - M.bottom}
            onMouseMove={(e) => setHover(locate(e))}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => onPick(units[locate(e).g])}
          />
        </svg>
        {hover !== null && (
          <ChartHover x={x(units[hover.g])} y={M.top} width={W} height={H} testId="evolution-distribution-hover">
            <div className="cmp-hover-title">generation {units[hover.g]}</div>
            <div>
              F {bandLabel(view.edges, hover.k)}: {hk} network{hk === 1 ? "" : "s"} ({Math.round(((hk ?? 0) / view.population) * 100)} %)
            </div>
          </ChartHover>
        )}
      </div>
      <div className="toy-legend" data-testid="evolution-shade-legend">
        <span className="toy-legend-item">share of the generation in a band:</span>
        {shadeLegend(view.population).map((s) => (
          <span key={s.shade} className="toy-legend-item">
            <span className="evolution-swatch" style={{ background: shadeColour(s.shade) }} /> {s.label}
          </span>
        ))}
      </div>
    </figure>
  );
}

/** The chosen generation's networks per band, as bars (a label on each filled band). */
function Histogram({ view, g }: { view: EvolutionView; g: number }) {
  const top = topEdge(view);
  const x = linearScale(0, top, M.left, W - M.right);
  const bands = filledBands(view, g);
  const { ticks, y } = valueAxis(Math.max(1, ...bands.map(([, c]) => c)));
  const xTicks = niceTicks(0, top, 5);
  const width = x(view.band_width) - x(0);
  const u = view.unit[g];
  return (
    <figure className="toy-figure" data-testid="evolution-histogram" data-generation={u}>
      <figcaption className="toy-figure-title">
        Generation {u}: networks per fitness band ({view.zero[g]} at exactly 0)
      </figcaption>
      <svg className="toy-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Histogram of generation ${u}'s fitness`}>
        {ticks.map((t) => (
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
          fitness F
        </text>
        <text className="toy-axis-title" transform={`translate(12 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
          networks
        </text>
        {bands.map(([k, count]) => (
          <g key={k} data-testid="evolution-bar" data-band={k} data-count={count}>
            <rect className="evolution-bar" x={x(view.edges[k]) + 1} y={y(count)} width={Math.max(width - 2, 1)} height={y(0) - y(count)} rx={2}>
              <title>{`F ${bandLabel(view.edges, k)}: ${count} network${count === 1 ? "" : "s"}`}</title>
            </rect>
            <text className="evolution-bar-label" x={x(view.edges[k]) + width / 2} y={y(count) - 4} textAnchor="middle">
              {count}
            </text>
          </g>
        ))}
      </svg>
    </figure>
  );
}

/** D, the mean distance between every pair of genomes, per generation. */
function DiversityChart({ view, cursor, onPick }: ChartProps) {
  const [hover, setHover] = useState<number | null>(null);
  const units = view.unit;
  const x = useMemo(() => generationScale(units), [units]);
  const values = view.diversity.map((d) => d ?? 0);
  const { ticks, y } = valueAxis(Math.max(...values));
  const toUnit = linearScale(M.left, W - M.right, units[0] - 0.5, units[units.length - 1] + 0.5);
  const px = units.map(x);
  const py = values.map(y);
  const h = hover !== null && hover < units.length ? hover : null;
  return (
    <figure className="toy-figure" data-testid="evolution-diversity" data-points={units.length}>
      <figcaption className="toy-figure-title">Diversity D: mean distance between two networks' weights</figcaption>
      <div className="cmp-plot">
        <svg className="toy-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Diversity over ${units.length} generations`}>
          <GenerationFrame x={x} units={units} yTicks={ticks} y={y} yTitle="diversity D" cursor={cursor}>
            {units.length > 1 ? (
              <path className="toy-line" d={linePath(px, py)} data-testid="evolution-line-diversity" />
            ) : (
              <circle className="evolution-dot" cx={px[0]} cy={py[0]} r={4} />
            )}
          </GenerationFrame>
          {h !== null && <circle className="toy-hover-dot" pointerEvents="none" cx={px[h]} cy={py[h]} r={4} />}
          <rect
            className="toy-hit evolution-hit"
            data-testid="evolution-diversity-hit"
            x={M.left}
            y={M.top}
            width={W - M.left - M.right}
            height={H - M.top - M.bottom}
            onMouseMove={(e) => setHover(nearestIndex(units, toUnit(pointerX(e))))}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => onPick(units[nearestIndex(units, toUnit(pointerX(e)))])}
          />
        </svg>
        {h !== null && (
          <ChartHover x={px[h]} y={M.top} width={W} height={H} testId="evolution-diversity-hover">
            <div className="cmp-hover-title">generation {units[h]}</div>
            <div>D {view.diversity[h] === null ? "–" : formatNumber(view.diversity[h]!)}</div>
          </ChartHover>
        )}
      </div>
    </figure>
  );
}

/** The same numbers as a table (for reading exact values, and without colour). */
function EvolutionTable({ view }: { view: EvolutionView }) {
  return (
    <details className="evolution-table">
      <summary>Table of every generation</summary>
      <table className="compare-table" data-testid="evolution-table">
        <thead>
          <tr>
            <th scope="col">Generation</th>
            <th scope="col">Best</th>
            <th scope="col">Upper quartile</th>
            <th scope="col">Median</th>
            <th scope="col">Mean</th>
            <th scope="col">Lower quartile</th>
            <th scope="col">Worst</th>
            <th scope="col">At 0</th>
            <th scope="col">Finish every start</th>
            <th scope="col">Diversity D</th>
          </tr>
        </thead>
        <tbody>
          {view.unit.map((u, g) => (
            <tr key={u}>
              <td>{u}</td>
              <td>{formatNumber(view.best[g])}</td>
              <td>{formatNumber(view.q3[g])}</td>
              <td>{formatNumber(view.median[g])}</td>
              <td>{formatNumber(view.mean[g])}</td>
              <td>{formatNumber(view.q1[g])}</td>
              <td>{formatNumber(view.worst[g])}</td>
              <td>{view.zero[g]}</td>
              <td>{view.finishers[g] ?? "–"}</td>
              <td>{view.diversity[g] === null ? "–" : formatNumber(view.diversity[g]!)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}
