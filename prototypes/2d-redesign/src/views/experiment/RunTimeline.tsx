import { type ReactNode, useState } from "react";

import type { AncestryStep, Evolution, ExamCard, RunData } from "../../mock/types";
import { num, useApp } from "../../state/store";
import { Button, Chip, Field, LegendItem, Select, Stat, Swatch } from "../../ui/controls";
import { type Box, Figure, linear, linePath, bandPath, niceTicks, Plot, Tip, TipRow, type TipState, XAxis, YAxis } from "./charts";

const SERIES = [
  { key: "best", label: "best", color: "var(--series-1)" },
  { key: "median", label: "median", color: "var(--series-2)" },
  { key: "mean", label: "mean", color: "var(--series-3)" },
  { key: "worst", label: "worst", color: "var(--series-4)" },
] as const;

const OUTCOMES = [
  { key: "finished", label: "finished", color: "var(--ok-fill)" },
  { key: "crashed", label: "crashed", color: "var(--bad-fill)" },
  { key: "stalled", label: "stalled", color: "var(--warn-fill)" },
  { key: "timeout", label: "out of time", color: "var(--idle-fill)" },
] as const;

/** Shade 1..5 of the share of a generation in one fitness band (limits 5, 10, 20, 40 %). */
const shadeOf = (share: number) => (share <= 0.05 ? 1 : share <= 0.1 ? 2 : share <= 0.2 ? 3 : share <= 0.4 ? 4 : 5);
/** Shade 0..5 of the share that finished (limits 20, 40, 60, 80 %). */
const finishShade = (share: number) => (share <= 0 ? 0 : share <= 0.2 ? 1 : share <= 0.4 ? 2 : share <= 0.6 ? 3 : share <= 0.8 ? 4 : 5);
/** The ramp token of a shade: the tokens flip with the theme so "more" always stands out. */
const ramp = (shade: number) => `var(--rank-${shade === 0 ? 0 : 6 - shade})`;

const M = { top: 14, right: 62, bottom: 34, left: 46 };

function topOf(view: Evolution): number {
  return Math.ceil((Math.max(...view.best) * 1.08) / 0.2) * 0.2;
}

/** The x position of generation u: one column per generation. */
function genScale(view: Evolution, box: Box) {
  const n = view.unit.length;
  return { x: linear(0.5, n + 0.5, box.left, box.right), column: (box.right - box.left) / n };
}

/** The selected generation as a column across a chart, and a strip that picks one on click or drag. */
function GenerationCursor({ view, box, generation, onPick, onHover }: { view: Evolution; box: Box; generation: number; onPick: (g: number) => void; onHover?: (g: number | null, e?: React.MouseEvent) => void }) {
  const { x, column } = genScale(view, box);
  const at = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
    const u = Math.round(0.5 + ((e.clientX - rect.left) / rect.width) * view.unit.length);
    return Math.max(1, Math.min(view.unit.length, u));
  };
  return (
    <>
      <rect x={x(generation) - column / 2} y={box.top} width={column} height={box.bottom - box.top} className="cursor-column" />
      <rect
        x={box.left}
        y={box.top}
        width={box.right - box.left}
        height={box.bottom - box.top}
        className="hit"
        onClick={(e) => onPick(at(e))}
        onMouseMove={(e) => {
          if (e.buttons === 1) onPick(at(e));
          onHover?.(at(e), e);
        }}
        onMouseLeave={() => onHover?.(null)}
      />
    </>
  );
}

/** Best, median, mean and worst fitness of every generation, with the middle half as a band. */
function FitnessChart({ view, generation, checkpoints, onPick }: { view: Evolution; generation: number; checkpoints: number[]; onPick: (g: number) => void }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const top = topOf(view);
  return (
    <>
      <Plot height={250} margin={M} label="Fitness of each generation" testId="evolution-fitness">
        {(box) => {
          const { x } = genScale(view, box);
          const y = linear(0, top, box.bottom, box.top);
          const xs = view.unit.map(x);
          // End labels, pushed apart so they never overlap.
          const labels = SERIES.map((s) => ({ ...s, y: y(view[s.key][view.unit.length - 1]) })).sort((a, b) => a.y - b.y);
          for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 13);
          return (
            <>
              <YAxis box={box} scale={y} ticks={niceTicks(0, top, 5)} label="fitness F" />
              <XAxis box={box} scale={x} ticks={view.unit.filter((u) => u === 1 || u % 5 === 0)} />
              <path d={bandPath(xs, view.q1.map(y), view.q3.map(y))} className="band" style={{ fill: "var(--series-2)" }} />
              {SERIES.map((s) => (
                <path key={s.key} d={linePath(xs, view[s.key].map(y))} className="line" style={{ stroke: s.color }} data-testid={`evolution-line-${s.key}`} />
              ))}
              {/* Checkpoints: the generations whose champion was saved and examined. */}
              {checkpoints
                .filter((u) => u >= 1)
                .map((u) => (
                  <circle key={u} cx={x(u)} cy={y(view.best[u - 1])} r={3.5} className="checkpoint" />
                ))}
              {labels.map((s) => (
                <g key={s.key}>
                  <circle cx={box.right + 8} cy={s.y} r={3.5} style={{ fill: s.color }} />
                  <text x={box.right + 15} y={s.y + 3.5} className="end-label">
                    {s.label}
                  </text>
                </g>
              ))}
              <GenerationCursor
                view={view}
                box={box}
                generation={generation}
                onPick={onPick}
                onHover={(g, e) =>
                  setTip(
                    g === null || !e
                      ? null
                      : {
                          x: e.clientX,
                          y: e.clientY,
                          content: (
                            <>
                              <h5>Generation {g}</h5>
                              {SERIES.map((s) => (
                                <TipRow key={s.key} color={s.color} label={s.label} value={num(view[s.key][g - 1], 3)} />
                              ))}
                              <TipRow label="middle half" value={`${num(view.q1[g - 1], 2)} to ${num(view.q3[g - 1], 2)}`} />
                              <TipRow label="finish every start" value={`${view.finishers[g - 1]} of ${view.population}`} />
                            </>
                          ),
                        },
                  )
                }
              />
            </>
          );
        }}
      </Plot>
      <Tip tip={tip} />
    </>
  );
}

/** How the whole generation is spread over the fitness axis: a column per generation. */
function DistributionChart({ view, generation, onPick }: { view: Evolution; generation: number; onPick: (g: number) => void }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const top = topOf(view);
  const bands = Math.round(top / view.band_width);
  return (
    <>
      <Plot height={190} margin={M} label="Fitness distribution of each generation" testId="evolution-distribution">
        {(box) => {
          const { x, column } = genScale(view, box);
          const y = linear(0, top, box.bottom, box.top);
          const h = (box.bottom - box.top) / bands;
          return (
            <>
              <YAxis box={box} scale={y} ticks={niceTicks(0, top, 5)} label="fitness F" />
              <XAxis box={box} scale={x} ticks={view.unit.filter((u) => u === 1 || u % 5 === 0)} label="generation" />
              {view.unit.map((u, g) =>
                view.counts[g].slice(0, bands).map((count, k) =>
                  count === 0 ? null : (
                    <rect
                      key={`${u}-${k}`}
                      x={x(u) - column / 2 + 0.5}
                      y={y(view.edges[k + 1]) + 0.5}
                      width={Math.max(1, column - 1)}
                      height={Math.max(1, h - 1)}
                      style={{ fill: ramp(shadeOf(count / view.population)) }}
                      data-testid="evolution-cell"
                    />
                  ),
                ),
              )}
              <GenerationCursor
                view={view}
                box={box}
                generation={generation}
                onPick={onPick}
                onHover={(g, e) => {
                  if (g === null || !e) return setTip(null);
                  const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
                  const k = Math.max(0, Math.min(bands - 1, Math.floor((1 - (e.clientY - rect.top) / rect.height) * bands)));
                  const count = view.counts[g - 1][k];
                  setTip({
                    x: e.clientX,
                    y: e.clientY,
                    content: (
                      <>
                        <h5>
                          Generation {g} · F {view.edges[k].toFixed(2)} to {view.edges[k + 1].toFixed(2)}
                        </h5>
                        <TipRow label="networks" value={`${count} of ${view.population}`} />
                        <TipRow label="share" value={`${((100 * count) / view.population).toFixed(0)} %`} />
                      </>
                    ),
                  });
                }}
              />
            </>
          );
        }}
      </Plot>
      <Tip tip={tip} />
    </>
  );
}

/** One generation's networks per fitness band. */
function Histogram({ view, generation }: { view: Evolution; generation: number }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const top = topOf(view);
  const bands = Math.round(top / view.band_width);
  const counts = view.counts[generation - 1].slice(0, bands);
  const most = Math.max(1, ...counts);
  return (
    <>
      <Plot height={190} margin={{ top: 18, right: 12, bottom: 34, left: 40 }} label={`Networks per fitness band in generation ${generation}`} testId="evolution-histogram">
        {(box) => {
          const x = linear(0, top, box.left, box.right);
          const y = linear(0, most, box.bottom, box.top);
          const w = Math.min(24, (box.right - box.left) / bands - 2);
          return (
            <>
              <YAxis box={box} scale={y} ticks={niceTicks(0, most, 4).filter(Number.isInteger)} label="networks" />
              <XAxis box={box} scale={x} ticks={niceTicks(0, top, 5)} label="fitness F" />
              {counts.map((count, k) => {
                if (count === 0) return null;
                const cx = x(view.edges[k] + view.band_width / 2);
                return (
                  <g key={k} data-testid="evolution-bar">
                    <path
                      d={`M${cx - w / 2} ${box.bottom}V${y(count) + 3}q0 -3 3 -3h${w - 6}q3 0 3 3V${box.bottom}Z`}
                      className="bar"
                      onMouseMove={(e) =>
                        setTip({
                          x: e.clientX,
                          y: e.clientY,
                          content: (
                            <>
                              <h5>
                                F {view.edges[k].toFixed(2)} to {view.edges[k + 1].toFixed(2)}
                              </h5>
                              <TipRow label="networks" value={count} />
                            </>
                          ),
                        })
                      }
                      onMouseLeave={() => setTip(null)}
                    />
                    {count === most && (
                      <text x={cx} y={y(count) - 5} textAnchor="middle" className="bar-label">
                        {count}
                      </text>
                    )}
                  </g>
                );
              })}
            </>
          );
        }}
      </Plot>
      <Tip tip={tip} />
    </>
  );
}

function DiversityChart({ view, generation, onPick }: { view: Evolution; generation: number; onPick: (g: number) => void }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const top = Math.ceil(Math.max(...view.diversity));
  return (
    <>
      <Plot height={190} margin={{ top: 14, right: 14, bottom: 34, left: 40 }} label="Diversity of each generation" testId="evolution-diversity">
        {(box) => {
          const { x } = genScale(view, box);
          const y = linear(0, top, box.bottom, box.top);
          return (
            <>
              <YAxis box={box} scale={y} ticks={niceTicks(0, top, 4)} label="distance D" />
              <XAxis box={box} scale={x} ticks={view.unit.filter((u) => u === 1 || u % 5 === 0)} label="generation" />
              <path d={linePath(view.unit.map(x), view.diversity.map(y))} className="line" style={{ stroke: "var(--text-2)" }} />
              <circle cx={x(generation)} cy={y(view.diversity[generation - 1])} r={4.5} className="marker" />
              <GenerationCursor
                view={view}
                box={box}
                generation={generation}
                onPick={onPick}
                onHover={(g, e) =>
                  setTip(
                    g === null || !e
                      ? null
                      : { x: e.clientX, y: e.clientY, content: (<><h5>Generation {g}</h5><TipRow label="mean distance D" value={num(view.diversity[g - 1], 3)} /></>) },
                  )
                }
              />
            </>
          );
        }}
      </Plot>
      <Tip tip={tip} />
    </>
  );
}

/** Which starts each generation finishes: rows are the 10 evaluation starts. */
function StartsMap({ view, generation, raced, onPick }: { view: Evolution; generation: number; raced: number; onPick: (g: number) => void }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const S = view.n_starts;
  const rowH = 15;
  const height = 14 + S * rowH + 34;
  return (
    <>
      <Plot height={height} margin={{ top: 14, right: 14, bottom: 34, left: 46 }} label="Share of each generation that finishes from each start" testId="evolution-starts">
        {(box) => {
          const { x, column } = genScale(view, box);
          return (
            <>
              {Array.from({ length: S }, (_, k) => (
                <text key={k} x={box.left - 8} y={box.top + k * rowH + 11} textAnchor="end" className={k === raced ? "tick tick-strong" : "tick"}>
                  {k === raced ? "▸ " : ""}
                  {k + 1}
                </text>
              ))}
              <text x={12} y={(box.top + box.bottom) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(box.top + box.bottom) / 2})`} className="axis-label">
                start
              </text>
              <XAxis box={box} scale={x} ticks={view.unit.filter((u) => u === 1 || u % 5 === 0)} label="generation" />
              {view.unit.map((u, g) =>
                view.start_finish_share[g].map((share, k) => (
                  <rect
                    key={`${u}-${k}`}
                    x={x(u) - column / 2 + 0.5}
                    y={box.top + k * rowH + 0.5}
                    width={Math.max(1, column - 1)}
                    height={rowH - 1}
                    style={{ fill: ramp(finishShade(share)) }}
                    onMouseMove={(e) =>
                      setTip({
                        x: e.clientX,
                        y: e.clientY,
                        content: (
                          <>
                            <h5>
                              Generation {u} · start {k + 1}
                            </h5>
                            <TipRow label="finished" value={`${Math.round(share * view.population)} of ${view.population}`} />
                            <TipRow label="mean f on this start" value={num(view.start_mean[g][k], 3)} />
                          </>
                        ),
                      })
                    }
                    onMouseLeave={() => setTip(null)}
                    onClick={() => onPick(u)}
                    className="cell"
                  />
                )),
              )}
              <rect x={x(generation) - column / 2} y={box.top} width={column} height={S * rowH} className="cursor-outline" />
            </>
          );
        }}
      </Plot>
      <Tip tip={tip} />
    </>
  );
}

/** How every network of one generation drove from every start. A cell selects that car. */
function OutcomeGrid({ run }: { run: RunData }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const car = useApp((s) => s.car);
  const selectCar = useApp((s) => s.selectCar);
  const grid = run.starts;
  const S = grid.n_starts;
  const rowH = 15;
  const height = 14 + S * rowH + 34;
  return (
    <>
      <Plot height={height} margin={{ top: 14, right: 14, bottom: 34, left: 46 }} label="How each network drove from each start" testId="evolution-outcomes">
        {(box) => {
          const n = grid.population;
          const column = (box.right - box.left) / n;
          const x = (r: number) => box.left + r * column;
          const selectedRank = grid.order.indexOf(car);
          return (
            <>
              {Array.from({ length: S }, (_, k) => (
                <text key={k} x={box.left - 8} y={box.top + k * rowH + 11} textAnchor="end" className={k === run.start ? "tick tick-strong" : "tick"}>
                  {k === run.start ? "▸ " : ""}
                  {k + 1}
                </text>
              ))}
              <text x={12} y={(box.top + box.bottom) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(box.top + box.bottom) / 2})`} className="axis-label">
                start
              </text>
              <line x1={box.left} x2={box.right} y1={box.bottom} y2={box.bottom} className="baseline" />
              {[1, 10, 20, 30, 40, 50].map((r) => (
                <text key={r} x={x(r - 0.5)} y={box.bottom + 15} textAnchor="middle" className="tick">
                  {r}
                </text>
              ))}
              <text x={(box.left + box.right) / 2} y={box.bottom + 30} textAnchor="middle" className="axis-label">
                network, by rank of F (best at the left)
              </text>
              {grid.order.map((row, r) =>
                grid.status[row].map((status, k) => (
                  <rect
                    key={`${r}-${k}`}
                    x={x(r) + 0.5}
                    y={box.top + k * rowH + 0.5}
                    width={Math.max(1, column - 1)}
                    height={rowH - 1}
                    rx={1}
                    style={{ fill: OUTCOMES.find((o) => o.key === status)?.color ?? "var(--idle-fill)" }}
                    className="cell"
                    data-testid="evolution-outcome-cell"
                    onClick={() => selectCar(row)}
                    onMouseMove={(e) =>
                      setTip({
                        x: e.clientX,
                        y: e.clientY,
                        content: (
                          <>
                            <h5>
                              Car {row + 1} · rank {r + 1} · start {k + 1}
                            </h5>
                            <TipRow label="ended" value={OUTCOMES.find((o) => o.key === status)?.label ?? status} />
                            <TipRow label="progress" value={`${(100 * grid.progress[row][k]).toFixed(0)} % of 2 laps`} />
                            <TipRow label="f on this start" value={num(grid.start_fitness[row][k], 3)} />
                            <TipRow label="F over all starts" value={num(grid.fitness[row], 3)} />
                          </>
                        ),
                      })
                    }
                    onMouseLeave={() => setTip(null)}
                  />
                )),
              )}
              <rect x={box.left} y={box.top + run.start * rowH} width={box.right - box.left} height={rowH} className="cursor-outline cursor-outline-soft" />
              {selectedRank >= 0 && <rect x={x(selectedRank)} y={box.top - 3} width={column} height={S * rowH + 6} className="cursor-outline" />}
            </>
          );
        }}
      </Plot>
      <Tip tip={tip} />
    </>
  );
}

type Mark = "random" | "elite" | "child" | "crossed";
const markOf = (s: AncestryStep): Mark => (s.origin === "child" && s.crossed ? "crossed" : s.origin);
const MARKS: { key: Mark; label: string }[] = [
  { key: "random", label: "random network of generation 1" },
  { key: "elite", label: "elite: an exact copy" },
  { key: "child", label: "child, mutated" },
  { key: "crossed", label: "child of two parents" },
];

function Marker({ mark, x, y }: { mark: Mark; x: number; y: number }) {
  if (mark === "random") return <rect x={x - 4.5} y={y - 4.5} width={9} height={9} className="anc-mark" />;
  if (mark === "elite") return <circle cx={x} cy={y} r={4.5} className="anc-mark anc-hollow" />;
  if (mark === "crossed") return <path d={`M${x} ${y - 6}L${x + 6} ${y}L${x} ${y + 6}L${x - 6} ${y}Z`} className="anc-mark" />;
  return <circle cx={x} cy={y} r={4.5} className="anc-mark" />;
}

/** The line of descent of one network back to generation 1, over each generation's best. */
function AncestryChart({ run, view }: { run: RunData; view: Evolution }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const steps = [...run.ancestry.steps].sort((a, b) => a.unit - b.unit);
  const top = topOf(view);
  return (
    <>
      <Plot height={220} margin={{ top: 14, right: 14, bottom: 34, left: 46 }} label="Line of descent" testId="ancestry-chart">
        {(box) => {
          const { x } = genScale(view, box);
          const y = linear(0, top, box.bottom, box.top);
          return (
            <>
              <YAxis box={box} scale={y} ticks={niceTicks(0, top, 5)} label="fitness F" />
              <XAxis box={box} scale={x} ticks={view.unit.filter((u) => u === 1 || u % 5 === 0)} label="generation" />
              <path d={linePath(view.unit.map(x), view.best.map(y))} className="line line-ref" />
              <path d={linePath(steps.map((s) => x(s.unit)), steps.map((s) => y(s.fitness)))} className="line" style={{ stroke: "var(--accent)" }} />
              {steps.map((s) => (
                <g
                  key={s.unit}
                  className="anc-step"
                  onMouseMove={(e) =>
                    setTip({
                      x: e.clientX,
                      y: e.clientY,
                      content: (
                        <>
                          <h5>
                            Generation {s.unit} · row {s.row}
                          </h5>
                          <TipRow label="fitness F" value={num(s.fitness, 3)} />
                          <TipRow label="rank" value={`${s.rank + 1} of ${view.population}`} />
                          <TipRow label="origin" value={MARKS.find((m) => m.key === markOf(s))?.label.split(":")[0] ?? s.origin} />
                          {s.origin === "child" && <TipRow label="genes mutated" value={s.mutated} />}
                          {s.origin === "child" && s.crossed && <TipRow label="from 2nd parent" value={s.from_other} />}
                        </>
                      ),
                    })
                  }
                  onMouseLeave={() => setTip(null)}
                >
                  <circle cx={x(s.unit)} cy={y(s.fitness)} r={11} fill="transparent" />
                  <Marker mark={markOf(s)} x={x(s.unit)} y={y(s.fitness)} />
                </g>
              ))}
            </>
          );
        }}
      </Plot>
      <Tip tip={tip} />
    </>
  );
}

const pct = (v: number) => `${(100 * v).toFixed(0)} %`;

/** The Driving Exam of one checkpoint: the only numbers that compare with other methods. */
function ExamPanel({ card, unit }: { card: ExamCard; unit: number }) {
  return (
    <div className="exam" data-testid="exam-card">
      <div className="exam-split">
        <div className="exam-block">
          <span className="eyebrow">Training track</span>
          <Stat label="finished" value={pct(card.training.completion_rate)} note={`of ${card.training.episodes} starts`} />
        </div>
        <div className="exam-block" data-heldout="true">
          <span className="eyebrow">Held-out tracks · never trained on</span>
          <div className="exam-stats">
            <Stat label="finished" value={pct(card.heldout.completion_rate)} note={`of ${card.heldout.episodes} starts`} />
            <Stat label="crashed" value={pct(card.heldout.crash_rate)} />
            <Stat
              label="progress, median"
              value={pct(card.heldout.progress_frac_median)}
              note={`middle half ${pct(card.heldout.progress_frac_q1)} to ${pct(card.heldout.progress_frac_q3)}`}
            />
          </div>
        </div>
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>Track</th>
            <th>Finished</th>
            <th className="r">Crashed</th>
            <th className="r" data-tip="Median over the track's 10 starts">Progress</th>
            <th className="r exam-wide">Best lap</th>
          </tr>
        </thead>
        <tbody>
          {card.tracks.map((t) => (
            <tr key={t.track}>
              <td>
                <span className="exam-track">
                  {t.track}
                  <Chip kind={t.split === "train" ? "plain" : "accent"} icon={null}>
                    {t.split === "train" ? "train" : "held-out"}
                  </Chip>
                </span>
              </td>
              <td>
                <span className="exam-meter">
                  <span className="exam-meter-track">
                    <span style={{ width: pct(t.aggregate.completion_rate).replace(" ", "") }} />
                  </span>
                  <span className="num">
                    {Math.round(t.aggregate.completion_rate * t.aggregate.episodes)}/{t.aggregate.episodes}
                  </span>
                </span>
              </td>
              <td className="r num">{pct(t.aggregate.crash_rate)}</td>
              <td className="r num">{pct(t.aggregate.progress_frac_median)}</td>
              <td className="r num exam-wide">{t.aggregate.median_best_lap_s ? `${t.aggregate.median_best_lap_s.toFixed(1)} s` : "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="figure-foot">
        Champion of generation {unit}, {card.env_steps.toLocaleString("en")} exam steps. Fitness says how well it drives the starts it
        was bred on; the exam says how well it drives.
      </p>
    </div>
  );
}

function OnlyRecorded({ generation, recorded, children }: { generation: number; recorded: number; children: ReactNode }) {
  const set = useApp((s) => s.set);
  if (generation === recorded) return <>{children}</>;
  return (
    <div className="empty" style={{ minHeight: 190 }}>
      <span className="empty-title">Recorded for generation {recorded} only</span>
      <span>In the app this panel follows the chosen generation. The prototype recorded one.</span>
      <Button onClick={() => set({ generation: recorded })}>Back to generation {recorded}</Button>
    </div>
  );
}

/**
 * The Timeline tool for a GA run, as a dashboard. Reading order: the run and its headline
 * numbers; fitness over the generations with the generation cursor; the Driving Exam of the
 * chosen checkpoint; then the generation in detail (distribution, starts, outcomes, ancestry).
 * One generation is chosen for the whole page, here or on any chart.
 */
export function RunTimeline() {
  const data = useApp((s) => s.data);
  const generation = useApp((s) => s.generation);
  const evoStart = useApp((s) => s.evoStart);
  const set = useApp((s) => s.set);
  if (data === null) return <div className="skeleton" style={{ height: 400, margin: 16 }} />;
  const run = data.run;
  const view = evoStart === null ? run.evolution : run.evolutionStart;
  const all = run.evolution;
  const g = generation;
  const pick = (u: number) => set({ generation: u });
  const steps = run.ancestry.steps;
  const elites = steps.filter((s) => s.origin === "elite").length;
  const children = steps.filter((s) => s.origin === "child");
  const lastBest = all.best_so_far[all.best_so_far.length - 1];
  const go = () => {
    window.location.hash = "#/environment";
  };

  return (
    <div className="dash" data-testid="timeline">
      <section className="dash-head span-12">
        <div className="dash-run">
          <Field label="Run">
            <Select value={run.run_id} options={[{ value: run.run_id, label: `${run.run_id.replace(/^\d{8}-\d{6}-/, "")} · 26 checkpoints` }]} onChange={() => {}} disabled />
          </Field>
          <Chip kind="plain" icon="dna">
            genetic algorithm
          </Chip>
          <Chip kind="ok">completed</Chip>
          <span className="muted">
            seed {run.config.seed} · {run.config.network.layer_sizes.join(" → ")} · tournament of {run.config.selection.tournament_size} · crossover{" "}
            {run.config.variation.crossover_rate} · mutation {run.config.variation.mutation_rate} × σ {run.config.variation.mutation_strength}
          </span>
        </div>
        <div className="dash-stats">
          <Stat label="Generations" value={all.unit.length} note={`${all.population} networks, ${all.elite} elites`} />
          <Stat label="Best fitness so far" value={lastBest.toFixed(3)} note={`from ${all.best[0].toFixed(3)} in generation 1`} />
          <Stat label="Finish every start" value={`${all.finishers[all.finishers.length - 1]} of ${all.population}`} note="in the last generation" />
          <Stat label="Diversity D" value={all.diversity[all.diversity.length - 1].toFixed(2)} note={`from ${all.diversity[0].toFixed(2)}`} />
        </div>
      </section>

      <section className="dash-bar span-12" data-testid="generation-bar">
        <div className="dash-gen">
          <span className="eyebrow">Generation</span>
          <span className="dash-gen-value num">{g}</span>
          <input
            type="range"
            min={1}
            max={all.unit.length}
            value={g}
            onChange={(e) => pick(Number(e.target.value))}
            aria-label="Generation"
            data-testid="timeline-slider"
          />
          <span className="muted">
            best <b className="num">{num(all.best[g - 1], 3)}</b> · median <b className="num">{num(all.median[g - 1], 3)}</b> ·{" "}
            <b className="num">{all.finishers[g - 1]}</b> of {all.population} finish every start
          </span>
        </div>
        <div className="dash-actions">
          <Field label="Fitness on">
            <Select
              value={evoStart === null ? "all" : "one"}
              options={[
                { value: "all", label: "all 10 starts (mean F)" },
                { value: "one", label: `start ${run.start + 1} alone (f)` },
              ]}
              onChange={(v) => set({ evoStart: v === "all" ? null : run.start })}
            />
          </Field>
          <Button onClick={go} title="Drive this generation's champion from the exam's 10 starts">
            Replay champion
          </Button>
          <Button variant="primary" icon="play" onClick={go} testId="timeline-race">
            Race generation {g}
          </Button>
        </div>
      </section>

      <div className="panel span-7">
        <div className="panel-body">
          <Figure
            title={evoStart === null ? "Fitness F of each generation's networks" : `Fitness f on start ${evoStart + 1} alone`}
            note="Lines: best, median, mean, worst. Band: the middle half. Rings: saved checkpoints. Click or drag to choose a generation."
            objective="fitness"
            testId="fitness-figure"
          >
            <FitnessChart view={view} generation={g} checkpoints={run.checkpoints} onPick={pick} />
          </Figure>
          <Figure
            title="Where the whole generation sits"
            note="Each column is one generation; each cell a fitness band of 0.05, shaded by the share of the generation in it."
            footer={
              <div className="legend">
                <span>share of the generation:</span>
                {["up to 5 %", "10 %", "20 %", "40 %", "more"].map((label, k) => (
                  <LegendItem key={label} swatch={<Swatch color={ramp(k + 1)} />}>
                    {label}
                  </LegendItem>
                ))}
              </div>
            }
          >
            <DistributionChart view={view} generation={g} onPick={pick} />
          </Figure>
        </div>
      </div>

      <div className="panel span-5">
        <div className="panel-body">
          <Figure title={`Driving Exam · checkpoint ${run.generation}`} note="The champion of this generation on all five exam tracks." objective="exam" testId="exam-figure">
            <OnlyRecorded generation={g} recorded={run.generation}>
              <ExamPanel card={run.exam} unit={run.generation} />
            </OnlyRecorded>
          </Figure>
        </div>
      </div>

      <div className="panel span-4">
        <div className="panel-body">
          <Figure title={`Generation ${g}: networks per fitness band`} objective="fitness" note={`${view.zero[g - 1]} networks at exactly 0.`}>
            <Histogram view={view} generation={g} />
          </Figure>
        </div>
      </div>

      <div className="panel span-4">
        <div className="panel-body">
          <Figure title="Diversity: how different the networks are" objective="diversity" note="Mean distance between two networks' weights.">
            <DiversityChart view={all} generation={g} onPick={pick} />
          </Figure>
        </div>
      </div>

      <div className="panel span-4">
        <div className="panel-body">
          <Figure
            title="Which starts each generation finishes"
            note="Rows: the 10 evaluation starts (▸ = the one raced). Shade: share of the generation that finished."
            footer={
              <div className="legend">
                {["none", "up to 20 %", "40 %", "60 %", "80 %", "more"].map((label, k) => (
                  <LegendItem key={label} swatch={<Swatch color={ramp(k)} />}>
                    {label}
                  </LegendItem>
                ))}
              </div>
            }
          >
            <StartsMap view={all} generation={g} raced={run.start} onPick={pick} />
          </Figure>
        </div>
      </div>

      <div className="panel span-7">
        <div className="panel-body">
          <Figure
            title={`Generation ${run.generation}: how each network drove from each start`}
            note="Click a cell to select that car in every view. Dashed outline: the start that is raced (▸). Solid: the selected car."
            footer={
              <div className="legend">
                {OUTCOMES.map((o) => (
                  <LegendItem key={o.key} swatch={<Swatch color={o.color} />}>
                    {o.label}
                  </LegendItem>
                ))}
              </div>
            }
            testId="outcomes-figure"
          >
            <OnlyRecorded generation={g} recorded={run.generation}>
              <OutcomeGrid run={run} />
            </OnlyRecorded>
          </Figure>
        </div>
      </div>

      <div className="panel span-5">
        <div className="panel-body">
          <Figure
            title={`Ancestry of generation ${run.generation}'s champion`}
            objective="fitness"
            note={`Row ${run.ancestry.row} goes back ${steps.length - 1} generations: ${elites} elite copies, ${children.length} children (${children.filter((s) => s.crossed).length} crossed), ${children.reduce((n, s) => n + s.mutated, 0)} genes mutated along the line.`}
            footer={
              <div className="legend">
                {MARKS.map((m) => (
                  <LegendItem
                    key={m.key}
                    swatch={
                      <svg width="14" height="14" aria-hidden="true">
                        <Marker mark={m.key} x={7} y={7} />
                      </svg>
                    }
                  >
                    {m.label}
                  </LegendItem>
                ))}
                <LegendItem swatch={<span className="legend-refline" aria-hidden="true" />}>each generation&apos;s best</LegendItem>
              </div>
            }
            testId="ancestry-figure"
          >
            <OnlyRecorded generation={g} recorded={run.generation}>
              <AncestryChart run={run} view={all} />
            </OnlyRecorded>
          </Figure>
        </div>
      </div>
    </div>
  );
}
