import { useEffect, useMemo, useState, type CSSProperties, type MouseEvent, type ReactNode } from "react";

import type { EvolutionView, GenerationStarts } from "../../api/generated";
import { fetchEvolution, fetchGenerationStarts } from "../../api/rest";
import { ChartHover } from "./ChartHover";
import { formatNumber, linePath, linearScale, nearestIndex, niceTicks, type Scale } from "./charts";
import {
  bandLabel,
  filledBands,
  finishLegend,
  finishShade,
  FITNESS_LINES,
  fitnessName,
  OUTCOMES,
  outcomeOf,
  panelGeneration,
  shadeColour,
  shadeLegend,
  shadeOf,
  spreadLabels,
  startLabel,
  topEdge,
  type StatKey,
} from "./evolutionCharts";

/** The generation and start racing in the Environment view, if they are this run's. */
export type Racing = { unit: number; start: number } | null;

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
 *
 * Since the follow-up (ADR-059, api v1.24) also one start at a time: the start picker redraws the
 * fitness charts with f on that start, the starts map shows which starts each generation finishes,
 * and the outcome grid shows how every network of the chosen generation drove from every start
 * (clicking a cell races that generation from that start with that network selected).
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
  /** Race a generation from a start (0-based, default 0), selecting a car if given. */
  onRace: (generation: number, start?: number, car?: number) => void;
  racing: Racing;
}) {
  const [view, setView] = useState<EvolutionView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<number | null>(null);
  const [start, setStart] = useState<number | null>(null);
  const [grid, setGrid] = useState<GenerationStarts | null>(null);
  const [gridError, setGridError] = useState<string | null>(null);

  useEffect(() => setStart(null), [runId]);
  useEffect(() => {
    let live = true;
    fetchEvolution(runId, start)
      .then((answer) => {
        if (!live) return;
        setView(answer);
        setError(null);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [runId, generations, start]);
  useEffect(() => setPicked(null), [sliderUnit, runId]);

  const units = view !== null && view.run_id === runId ? view.unit : [];
  const shownUnit = picked !== null && units.includes(picked) ? picked : panelGeneration(sliderUnit, units);
  const hasStarts = view !== null && view.run_id === runId && view.n_starts !== null;
  useEffect(() => {
    if (!hasStarts || shownUnit === null) return;
    let live = true;
    fetchGenerationStarts(runId, shownUnit)
      .then((answer) => {
        if (!live) return;
        setGrid(answer);
        setGridError(null);
      })
      .catch((e: Error) => live && setGridError(e.message));
    return () => {
      live = false;
    };
  }, [runId, shownUnit, generations, hasStarts]);

  if (error) return <p className="toylab-error">Could not load the evolution charts: {error}</p>;
  if (view === null || view.run_id !== runId) return <p className="hint">Loading the evolution charts…</p>;
  if (view.unit.length === 0)
    return (
      <p className="hint" data-testid="evolution-empty">
        No generation has been scored yet.
      </p>
    );

  const shown = shownUnit!;
  const g = view.unit.indexOf(shown);
  const k = view.start;
  const startShare = k !== null && view.start_finish_share !== null ? view.start_finish_share[g]?.[k] : undefined;
  const raceStart = k ?? 0;
  const isRacing = racing !== null && racing.unit === shown && racing.start === raceStart;
  const shownGrid = grid !== null && grid.run_id === runId && grid.unit === shown ? grid : null;
  return (
    <section
      className="evolution"
      data-testid="evolution"
      data-generations={view.unit.length}
      data-generation={shown}
      data-start={k ?? "all"}
    >
      <h3 className="evolution-title">
        Evolution: {view.unit.length} generation{view.unit.length === 1 ? "" : "s"} of {view.population} networks
        {view.elite > 0 ? `, ${view.elite} elite${view.elite === 1 ? "" : "s"} kept each time` : ""}
      </h3>
      <div className="evolution-actions">
        {view.n_starts !== null && (
          <label className="picker">
            <span>Fitness on </span>
            <select
              data-testid="evolution-start"
              value={start === null ? "all" : String(start)}
              onChange={(e) => setStart(e.target.value === "all" ? null : Number(e.target.value))}
            >
              <option value="all">all starts (mean F)</option>
              {Array.from({ length: view.n_starts }, (_, i) => (
                <option key={i} value={i}>
                  start {i + 1}
                </option>
              ))}
            </select>
          </label>
        )}
        <span data-testid="evolution-generation">
          Generation {shown}
          {k === null ? "" : ` on start ${k + 1}`}: best {formatNumber(view.best[g])}, median{" "}
          {formatNumber(view.median[g])}, worst {formatNumber(view.worst[g])}
          {k === null && view.finishers[g] !== null ? `, ${view.finishers[g]} of ${view.population} finish every start` : ""}
          {startShare !== undefined ? `, ${Math.round(startShare * view.population)} of ${view.population} finish it` : ""}
        </span>
        <button type="button" data-testid="evolution-race" disabled={isRacing} onClick={() => onRace(shown, raceStart)}>
          {isRacing
            ? `Generation ${shown} races from start ${raceStart + 1} in the Environment view`
            : `Race generation ${shown} from start ${raceStart + 1}`}
        </button>
      </div>
      <p className="hint">
        Click any chart to choose a generation; moving the timeline slider chooses its generation again.
        {view.n_starts !== null
          ? " Each network's fitness F is its mean over the starts; pick a start to see f on that start alone."
          : ""}
      </p>
      {view.n_starts === null && (
        <p className="hint" data-testid="evolution-no-starts">
          This run was trained before every network's drive from every start was saved, so only the mean over the
          starts can be shown. Train it again to see each start.
        </p>
      )}
      <div className="evolution-grid">
        <FitnessChart view={view} cursor={shown} onPick={setPicked} />
        <DistributionChart view={view} cursor={shown} onPick={setPicked} />
        <Histogram view={view} g={g} />
        <DiversityChart view={view} cursor={shown} onPick={setPicked} />
      </div>
      {view.n_starts !== null && (
        <div className="evolution-grid evolution-grid-starts">
          <StartsMap
            view={view}
            cursor={shown}
            names={shownGrid}
            onPick={(u, s) => {
              setPicked(u);
              setStart(s);
            }}
          />
          {shownGrid !== null ? (
            <OutcomeGrid grid={shownGrid} start={k} racing={racing} onRace={onRace} />
          ) : (
            <p className={gridError ? "toylab-error" : "hint"}>
              {gridError ? `Could not load generation ${shown}'s drives: ${gridError}` : `Loading generation ${shown}'s drives…`}
            </p>
          )}
        </div>
      )}
      <EvolutionTable view={view} />
      {shownGrid !== null && <OutcomeTable grid={shownGrid} />}
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
      <figcaption className="toy-figure-title">
        {view.start === null ? "Fitness F of each generation's networks" : `Fitness f on start ${view.start + 1} of each generation's networks`}
      </figcaption>
      <div className="cmp-plot">
        <svg className="toy-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Best, median, mean and worst fitness over ${units.length} generations`}>
          <GenerationFrame x={x} units={units} yTicks={ticks} y={y} yTitle={fitnessName(view.start)} cursor={cursor}>
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
            {view.start !== null && view.start_finish_share !== null && (
              <div className="cmp-hover-row">
                <span className="cmp-swatch cmp-swatch-off" />
                <span>finish start {view.start + 1}</span>
                <span className="cmp-hover-value">
                  {Math.round(view.start_finish_share[h][view.start] * view.population)} of {view.population}
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
      <figcaption className="toy-figure-title">
        Fitness distribution{view.start === null ? "" : ` on start ${view.start + 1}`}: networks per band of {view.band_width}
      </figcaption>
      <div className="cmp-plot">
        <svg className="toy-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Heat map of the fitness of every network in ${units.length} generations`}>
          <GenerationFrame x={x} units={units} yTicks={ticks} y={y} yTitle={fitnessName(view.start)} cursor={null}>
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
              {view.start === null ? "F" : "f"} {bandLabel(view.edges, hover.k)}: {hk} network{hk === 1 ? "" : "s"} ({Math.round(((hk ?? 0) / view.population) * 100)} %)
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
        Generation {u}{view.start === null ? "" : ` on start ${view.start + 1}`}: networks per fitness band ({view.zero[g]} at exactly 0)
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
          {fitnessName(view.start)}
        </text>
        <text className="toy-axis-title" transform={`translate(12 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
          networks
        </text>
        {bands.map(([k, count]) => (
          <g key={k} data-testid="evolution-bar" data-band={k} data-count={count}>
            <rect className="evolution-bar" x={x(view.edges[k]) + 1} y={y(count)} width={Math.max(width - 2, 1)} height={y(0) - y(count)} rx={2}>
              <title>{`${view.start === null ? "F" : "f"} ${bandLabel(view.edges, k)}: ${count} network${count === 1 ? "" : "s"}`}</title>
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

// ── one start at a time (task 7.7 follow-up, ADR-059) ───────────────────────────────────────

const ROW = 18; // the height of one start's row in the starts map and the outcome grid
const MAP_BOTTOM = 40;

function gridHeight(starts: number): number {
  return M.top + starts * ROW + MAP_BOTTOM;
}

/** Start labels down the left side of the starts map and the outcome grid. */
function StartRows({ starts, selected }: { starts: number; selected: number | null }) {
  return (
    <>
      {Array.from({ length: starts }, (_, s) => (
        <text
          key={s}
          className="toy-tick"
          x={M.left - 6}
          y={M.top + s * ROW + ROW / 2 + 4}
          textAnchor="end"
          fontWeight={s === selected ? 700 : undefined}
        >
          start {s + 1}
        </text>
      ))}
      {selected !== null && (
        <rect
          className="evolution-column"
          data-testid="evolution-start-row"
          x={M.left - 2}
          y={M.top + selected * ROW}
          width={W - M.left - M.right + 4}
          height={ROW}
        />
      )}
    </>
  );
}

/**
 * The starts map: one row per start, one column per generation, shaded by the share of the
 * generation that finished that start. Click a cell to choose that generation and that start.
 */
function StartsMap({
  view,
  cursor,
  names,
  onPick,
}: {
  view: EvolutionView;
  cursor: number;
  /** The chosen generation's grid, for where each start is (null while loading). */
  names: GenerationStarts | null;
  onPick: (generation: number, start: number) => void;
}) {
  const [hover, setHover] = useState<{ g: number; s: number } | null>(null);
  const units = view.unit;
  const shares = view.start_finish_share!;
  const means = view.start_mean!;
  const starts = view.n_starts!;
  const h = gridHeight(starts);
  const x = useMemo(() => generationScale(units), [units]);
  const toUnit = linearScale(M.left, W - M.right, units[0] - 0.5, units[units.length - 1] + 0.5);
  const colWidth = x(units[0] + 0.5) - x(units[0] - 0.5);
  const gap = colWidth > 6 ? 1 : 0;
  const xTicks = niceTicks(units[0], units[units.length - 1], 6).filter((t) => Number.isInteger(t));
  function locate(e: MouseEvent<SVGRectElement>) {
    const box = e.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const py = ((e.clientY - box.top) / box.height) * h;
    const s = Math.min(Math.max(Math.floor((py - M.top) / ROW), 0), starts - 1);
    return { g: nearestIndex(units, toUnit(pointerX(e))), s };
  }
  const hs = hover === null ? null : shares[hover.g][hover.s];
  return (
    <figure className="toy-figure" data-testid="evolution-starts" data-starts={starts}>
      <figcaption className="toy-figure-title">Which starts each generation finishes: share of its networks</figcaption>
      <div className="cmp-plot">
        <svg className="toy-svg" viewBox={`0 0 ${W} ${h}`} role="img" aria-label={`Finish share of ${starts} starts over ${units.length} generations`}>
          <rect className="evolution-map-bg" x={M.left} y={M.top} width={W - M.left - M.right} height={starts * ROW} />
          {units.map((u, g) =>
            shares[g].map((share, s) => {
              const shade = finishShade(share);
              return shade === 0 ? null : (
                <rect
                  key={`${u}:${s}`}
                  className="evolution-cell"
                  data-testid="evolution-start-cell"
                  data-share={share}
                  x={x(u - 0.5) + gap}
                  y={M.top + s * ROW + 1}
                  width={Math.max(colWidth - 2 * gap, 0.5)}
                  height={ROW - 2}
                  style={{ fill: shadeColour(shade) }}
                />
              );
            }),
          )}
          <StartRows starts={starts} selected={view.start} />
          {(xTicks.length ? xTicks : [units[0]]).map((t) => (
            <text key={`x${t}`} className="toy-tick" x={x(t)} y={h - MAP_BOTTOM + 16} textAnchor="middle">
              {t}
            </text>
          ))}
          <text className="toy-axis-title" x={(M.left + W - M.right) / 2} y={h - 4} textAnchor="middle">
            generation
          </text>
          <rect
            className="evolution-column"
            data-testid="evolution-starts-column"
            x={x(cursor - 0.5)}
            y={M.top}
            width={colWidth}
            height={starts * ROW}
          />
          <rect
            className="toy-hit evolution-hit"
            data-testid="evolution-starts-hit"
            x={M.left}
            y={M.top}
            width={W - M.left - M.right}
            height={starts * ROW}
            onMouseMove={(e) => setHover(locate(e))}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => {
              const at = locate(e);
              onPick(units[at.g], at.s);
            }}
          />
        </svg>
        {hover !== null && hs !== null && (
          <ChartHover x={x(units[hover.g])} y={M.top} width={W} height={h} testId="evolution-starts-hover">
            <div className="cmp-hover-title">
              generation {units[hover.g]}, {startLabel(hover.s, names?.start_fraction[hover.s], names?.start_lateral_m[hover.s])}
            </div>
            <div>
              {Math.round(hs * view.population)} of {view.population} finish ({Math.round(hs * 100)} %)
            </div>
            <div>mean f {formatNumber(means[hover.g][hover.s])}</div>
          </ChartHover>
        )}
      </div>
      <div className="toy-legend" data-testid="evolution-starts-legend">
        <span className="toy-legend-item">share that finishes:</span>
        {finishLegend().map((l) => (
          <span key={l.shade} className="toy-legend-item">
            <span
              className={l.shade === 0 ? "evolution-swatch evolution-swatch-empty" : "evolution-swatch"}
              style={l.shade === 0 ? undefined : { background: shadeColour(l.shade) }}
            />{" "}
            {l.label}
          </span>
        ))}
      </div>
    </figure>
  );
}

function driveWords(grid: GenerationStarts, i: number, s: number): string {
  const outcome = outcomeOf(grid.status[i][s]).label;
  const t = grid.finish_time[i][s];
  const time = t === null ? "" : `, ${formatNumber(t)} s`;
  return `${outcome}, f ${formatNumber(grid.start_fitness[i][s])}, ${Math.round(grid.progress[i][s] * 100)} % of the way${time}`;
}

/**
 * The chosen generation's outcome grid: one row per start, one column per network (best F on the
 * left), each cell coloured by how that drive ended. Click a cell to race the generation from that
 * start with that network selected.
 */
function OutcomeGrid({
  grid,
  start,
  racing,
  onRace,
}: {
  grid: GenerationStarts;
  start: number | null;
  racing: Racing;
  onRace: (generation: number, start?: number, car?: number) => void;
}) {
  const [hover, setHover] = useState<{ rank: number; s: number } | null>(null);
  const n = grid.population;
  const starts = grid.n_starts;
  const h = gridHeight(starts);
  const colWidth = (W - M.left - M.right) / n;
  const gap = colWidth > 5 ? 1 : 0;
  const x = (rank: number) => M.left + rank * colWidth;
  const ticks = [...new Set([1, ...niceTicks(1, n, 5).filter((t) => Number.isInteger(t) && t > 1 && t < n), n])];
  const raced = racing !== null && racing.unit === grid.unit ? racing.start : null;
  const elite = new Set(grid.elite);
  const hi = hover === null ? null : grid.order[hover.rank];
  return (
    <figure
      className="toy-figure"
      data-testid="evolution-outcomes"
      data-generation={grid.unit}
      data-networks={n}
      data-starts={starts}
    >
      <figcaption className="toy-figure-title">
        Generation {grid.unit}: how each network drove from each start (best F on the left)
      </figcaption>
      <div className="cmp-plot">
        <svg className="toy-svg" viewBox={`0 0 ${W} ${h}`} role="img" aria-label={`Outcome of ${n} networks on ${starts} starts in generation ${grid.unit}`}>
          {grid.order.map((i, rank) =>
            grid.status[i].map((status, s) => (
              <rect
                key={`${i}:${s}`}
                className="evolution-outcome"
                data-testid="evolution-outcome-cell"
                data-status={status}
                data-row={i}
                data-start={s}
                data-rank={rank + 1}
                x={x(rank) + gap / 2}
                y={M.top + s * ROW + 1}
                width={Math.max(colWidth - gap, 0.5)}
                height={ROW - 2}
                style={{ fill: outcomeOf(status).colour }}
                onMouseEnter={() => setHover({ rank, s })}
                onMouseLeave={() => setHover(null)}
                onClick={() => onRace(grid.unit, s, i)}
              />
            )),
          )}
          <StartRows starts={starts} selected={start} />
          {raced !== null && (
            <rect
              className="evolution-raced"
              data-testid="evolution-raced-row"
              x={M.left - 2}
              y={M.top + raced * ROW - 1}
              width={W - M.left - M.right + 4}
              height={ROW + 2}
              pointerEvents="none"
            />
          )}
          {grid.finish_share.map((share, s) => (
            <text key={`f${s}`} className="toy-tick" x={W - M.right + 6} y={M.top + s * ROW + ROW / 2 + 4}>
              {Math.round(share * n)} finish
            </text>
          ))}
          {ticks.map((t) => (
            <text key={`x${t}`} className="toy-tick" x={x(t - 0.5)} y={h - MAP_BOTTOM + 16} textAnchor="middle">
              {t}
            </text>
          ))}
          <text className="toy-axis-title" x={(M.left + W - M.right) / 2} y={h - 4} textAnchor="middle">
            network, by rank of F (1 = best)
          </text>
        </svg>
        {hover !== null && hi !== null && (
          <ChartHover x={x(hover.rank + 0.5)} y={M.top} width={W} height={h} testId="evolution-outcomes-hover">
            <div className="cmp-hover-title">
              rank {hover.rank + 1} (car {hi + 1}){elite.has(hi) ? ", kept as an elite" : ""}, F {formatNumber(grid.fitness[hi])}
            </div>
            <div>{startLabel(hover.s, grid.start_fraction[hover.s], grid.start_lateral_m[hover.s])}</div>
            <div>{driveWords(grid, hi, hover.s)}</div>
            <div>Click to race generation {grid.unit} from start {hover.s + 1} with this car selected.</div>
          </ChartHover>
        )}
      </div>
      <div className="toy-legend" data-testid="evolution-outcomes-legend">
        <span className="toy-legend-item">the drive ended:</span>
        {OUTCOMES.map((o) => (
          <span key={o.key} className="toy-legend-item">
            <span className="evolution-swatch" style={{ background: o.colour }} /> {o.label}
          </span>
        ))}
      </div>
    </figure>
  );
}

/** The outcome grid as a table: every network's drive from every start, in words. */
function OutcomeTable({ grid }: { grid: GenerationStarts }) {
  return (
    <details className="evolution-table">
      <summary>Table of generation {grid.unit}'s drives from every start</summary>
      <table className="compare-table" data-testid="evolution-outcomes-table">
        <thead>
          <tr>
            <th scope="col">Rank</th>
            <th scope="col">Car</th>
            <th scope="col">F</th>
            {Array.from({ length: grid.n_starts }, (_, s) => (
              <th key={s} scope="col">
                Start {s + 1}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {grid.order.map((i, rank) => (
            <tr key={i}>
              <td>{rank + 1}</td>
              <td>{i + 1}</td>
              <td>{formatNumber(grid.fitness[i])}</td>
              {grid.status[i].map((_, s) => (
                <td key={s}>{driveWords(grid, i, s)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}
