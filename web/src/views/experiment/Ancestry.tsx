import { useEffect, useMemo, useState } from "react";

import type { Ancestry as AncestryView, AncestryStepView, EvolutionView } from "../../api/generated";
import { fetchAncestry } from "../../api/rest";
import { ChartHover } from "./ChartHover";
import { formatNumber, linePath, linearScale, nearestIndex } from "./charts";
import { GenerationFrame, generationScale, H, M, pointerX, valueAxis, W } from "./Evolution";
import { lineSummary, MARKS, markOf, stepText, type Mark } from "./ancestryChart";
import { topEdge } from "./evolutionCharts";

/** One marker of the line of descent, its shape by origin (see `MARKS`). */
function Marker({ mark, x, y }: { mark: Mark; x: number; y: number }) {
  const cls = `ancestry-mark ancestry-mark-${mark}`;
  switch (mark) {
    case "random":
      return <rect className={cls} x={x - 4} y={y - 4} width={8} height={8} />;
    case "crossed":
      return <path className={cls} d={`M${x} ${y - 5.5}L${x + 5.5} ${y}L${x} ${y + 5.5}L${x - 5.5} ${y}Z`} />;
    case "unknown":
      return <path className={cls} d={`M${x - 4} ${y - 4}L${x + 4} ${y + 4}M${x + 4} ${y - 4}L${x - 4} ${y + 4}`} />;
    default:
      return <circle className={cls} cx={x} cy={y} r={mark === "elite" ? 4.5 : 4} />;
  }
}

function MarkKey({ mark }: { mark: Mark }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      <Marker mark={mark} x={7} y={7} />
    </svg>
  );
}

/**
 * The champion's ancestry (task 7.8, ADR-060): the line of descent of one network of the chosen
 * generation (its champion unless another row is picked), through each network's main parent back
 * to generation 1, drawn over the generation's best fitness, with a marker per generation shaped by
 * how that ancestor came about, and a table. Clicking a marker races that generation with that
 * ancestor selected. The line comes from `GET .../generations/{g}/ancestry`.
 */
export function Ancestry({
  runId,
  view,
  unit,
  raceStart,
  onRace,
}: {
  runId: string;
  view: EvolutionView;
  unit: number;
  raceStart: number;
  onRace: (generation: number, start?: number, car?: number) => void;
}) {
  const [row, setRow] = useState<number | null>(null);
  const [answer, setAnswer] = useState<AncestryView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => setRow(null), [runId, unit]);
  useEffect(() => {
    let live = true;
    fetchAncestry(runId, unit, row)
      .then((a) => {
        if (!live) return;
        setAnswer(a);
        setError(null);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [runId, unit, row]);

  const units = view.unit;
  const x = useMemo(() => generationScale(units), [units]);
  const { ticks, y } = useMemo(() => valueAxis(topEdge(view)), [view]);
  const toUnit = linearScale(M.left, W - M.right, units[0] - 0.5, units[units.length - 1] + 0.5);

  const header = (
    <div className="evolution-actions">
      <label className="picker">
        <span>Line of descent of </span>
        <select
          data-testid="ancestry-row"
          value={row === null ? "champion" : String(row)}
          onChange={(e) => setRow(e.target.value === "champion" ? null : Number(e.target.value))}
        >
          <option value="champion">generation {unit}&apos;s champion</option>
          {Array.from({ length: view.population }, (_, i) => (
            <option key={i} value={i}>
              row {i} of generation {unit}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
  if (error)
    return (
      <section className="ancestry" data-testid="ancestry" data-steps={0}>
        <h3 className="evolution-title">Ancestry</h3>
        {header}
        <p className="hint" data-testid="ancestry-error">
          {error}
        </p>
      </section>
    );
  // While another line loads, keep showing this run's last one, so the section (and an opened
  // table) stays in place instead of flashing to a loading line.
  if (answer === null || answer.run_id !== runId) return <p className="hint">Loading the line of descent…</p>;
  const steps = answer.steps;
  const sum = lineSummary(steps);
  const byUnit = steps.slice().reverse(); // oldest first, for the path
  const px = byUnit.map((s) => x(s.unit));
  const py = byUnit.map((s) => y(s.fitness));
  const stepUnits = byUnit.map((s) => s.unit);
  const h = hover !== null && hover < byUnit.length ? hover : null;
  const pick = (s: AncestryStepView) => onRace(s.unit, raceStart, s.row);
  const parents = steps.length - 1;
  return (
    <section className="ancestry" data-testid="ancestry" data-steps={steps.length} data-row={answer.row}>
      <h3 className="evolution-title">Ancestry</h3>
      {header}
      <p className="hint" data-testid="ancestry-summary">
        Row {answer.row} of generation {unit} goes back {parents} generation{parents === 1 ? "" : "s"} to a network of
        generation {steps[steps.length - 1].unit}: {sum.elite} elite cop{sum.elite === 1 ? "y" : "ies"}, {sum.child}{" "}
        child{sum.child === 1 ? "" : "ren"} ({sum.crossed} crossed), {sum.mutated} gene{sum.mutated === 1 ? "" : "s"}{" "}
        mutated along the line. Each step goes to the parent that gave more genes. Click a marker to race that
        generation with that ancestor selected.
      </p>
      <figure className="toy-figure" data-testid="ancestry-chart" data-points={steps.length}>
        <figcaption className="toy-figure-title">Fitness F of each ancestor, over each generation&apos;s best</figcaption>
        <div className="cmp-plot">
          <svg
            className="toy-svg"
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label={`Line of descent over ${steps.length} generations`}
          >
            <GenerationFrame x={x} units={units} yTicks={ticks} y={y} yTitle="fitness F" cursor={unit}>
              {units.length > 1 && (
                <path className="ancestry-best" d={linePath(units.map(x), view.best.map(y))} data-testid="ancestry-best" />
              )}
              {byUnit.length > 1 && <path className="ancestry-path" d={linePath(px, py)} data-testid="ancestry-path" />}
              {byUnit.map((s, i) => (
                <g key={s.unit} data-testid={`ancestry-step-${s.unit}`} data-row={s.row} data-origin={markOf(s)}>
                  <Marker mark={markOf(s)} x={px[i]} y={py[i]} />
                </g>
              ))}
            </GenerationFrame>
            {h !== null && <circle className="toy-hover-dot" pointerEvents="none" cx={px[h]} cy={py[h]} r={7} />}
            <rect
              className="toy-hit evolution-hit"
              data-testid="ancestry-hit"
              x={M.left}
              y={M.top}
              width={W - M.left - M.right}
              height={H - M.top - M.bottom}
              onMouseMove={(e) => setHover(nearestIndex(stepUnits, toUnit(pointerX(e))))}
              onMouseLeave={() => setHover(null)}
              onClick={(e) => pick(byUnit[nearestIndex(stepUnits, toUnit(pointerX(e)))])}
            />
          </svg>
          {h !== null && (
            <ChartHover x={px[h]} y={M.top} width={W} height={H} testId="ancestry-hover">
              <div className="cmp-hover-title">generation {byUnit[h].unit}</div>
              <div>{stepText(byUnit[h])}</div>
              {byUnit[h].distance !== null && <div>distance from its parent {formatNumber(byUnit[h].distance!)}</div>}
              <div>Click to race generation {byUnit[h].unit} with this network selected.</div>
            </ChartHover>
          )}
        </div>
        <div className="net-legend">
          {MARKS.filter((m) => m.key !== "unknown" || steps.some((s) => s.origin === "unknown")).map((m) => (
            <span key={m.key} className="net-legend-item">
              <MarkKey mark={m.key} /> {m.label}
            </span>
          ))}
          <span className="net-legend-item">
            <svg width="38" height="14" aria-hidden="true">
              <line x1="2" y1="7" x2="36" y2="7" className="ancestry-best" />
            </svg>
            the generation&apos;s best
          </span>
        </div>
      </figure>
      <details className="evolution-table">
        <summary>Table of the line of descent</summary>
        <table className="compare-table" data-testid="ancestry-table">
          <thead>
            <tr>
              <th scope="col">Generation</th>
              <th scope="col">Row</th>
              <th scope="col">Rank</th>
              <th scope="col">F</th>
              <th scope="col">Origin</th>
              <th scope="col">Parents</th>
              <th scope="col">Genes from the other parent</th>
              <th scope="col">Mutated genes</th>
              <th scope="col">Distance from the parent</th>
              <th scope="col">Race</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s) => (
              <tr key={s.unit}>
                <td>{s.unit}</td>
                <td>{s.row}</td>
                <td>{s.rank + 1}</td>
                <td>{formatNumber(s.fitness)}</td>
                <td>{MARKS.find((m) => m.key === markOf(s))?.label}</td>
                <td>{s.parents.length ? s.parents.join(" and ") : "–"}</td>
                <td>{s.from_other ?? "–"}</td>
                <td>{s.mutated ?? "–"}</td>
                <td>{s.distance === null ? "–" : formatNumber(s.distance)}</td>
                <td>
                  <button type="button" data-testid={`ancestry-race-${s.unit}`} onClick={() => pick(s)}>
                    Race
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </section>
  );
}
