import { useMemo, useState } from "react";

import type { CompareResponse, ExamGroup } from "../../mock/types";
import { useApp } from "../../state/store";
import { Planned, Segmented } from "../../ui/controls";
import { bandPath, Figure, formatPower, linear, linePath, logScale, logTicks, niceTicks, Plot, Tip, TipRow, type TipState, XAxis, YAxis } from "./charts";

const NAMES: [RegExp, string][] = [
  [/sgd$/, "SGD"],
  [/momentum$/, "Momentum"],
  [/rmsprop$/, "RMSProp"],
  [/adam$/, "Adam"],
  [/imitation-s5c-efull$/, "Imitation · expert, full set"],
  [/imitation-s5c-human$/, "Imitation · keyboard driver"],
  [/dagger-s5c-dw$/, "DAgger · wide starts"],
];
const nameOf = (id: string) => NAMES.find(([re]) => re.test(id))?.[1] ?? id.replace(/^\d{8}-\d{6}-/, "");
const series = (k: number) => `var(--series-${k + 1})`;

/** Quartiles by linear interpolation (the same rule as experiment/stats.py). */
function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/**
 * The Driving Exam of every method, one row each: a dot per seed, the median as a tick and the
 * middle half as a bar. This is the only chart in the app that puts different learning methods
 * on one axis, because the exam is the only measure they share.
 */
function ExamStrip({ groups }: { groups: ExamGroup[] }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const rows = useMemo(
    () =>
      groups
        .map((g) => {
          const values = g.cards.map((c) => c.heldout.progress_frac_median).sort((a, b) => a - b);
          const finished = g.cards.map((c) => c.training.completion_rate).sort((a, b) => a - b);
          return { ...g, values, median: quantile(values, 0.5), q1: quantile(values, 0.25), q3: quantile(values, 0.75), train: quantile(finished, 0.5) };
        })
        .sort((a, b) => b.median - a.median),
    [groups],
  );
  const rowH = 30;
  const height = 22 + rows.length * rowH + 36;
  return (
    <>
      <Plot height={height} margin={{ top: 22, right: 176, bottom: 36, left: 150 }} label="Held-out progress of every method" testId="exam-strip">
        {(box) => {
          const x = linear(0, 1, box.left, box.right);
          return (
            <>
              {niceTicks(0, 1, 5).map((t) => (
                <line key={t} x1={x(t)} x2={x(t)} y1={box.top} y2={box.bottom} className="grid" />
              ))}
              <XAxis box={box} scale={x} ticks={niceTicks(0, 1, 5)} format={(v) => `${Math.round(v * 100)} %`} label="progress on the held-out tracks, median of 40 starts (one dot per seed)" />
              <text x={box.right + 14} y={box.top - 8} className="tick tick-strong">
                training starts finished
              </text>
              {rows.map((r, k) => {
                const cy = box.top + k * rowH + rowH / 2;
                return (
                  <g
                    key={r.key}
                    onMouseMove={(e) =>
                      setTip({
                        x: e.clientX,
                        y: e.clientY,
                        content: (
                          <>
                            <h5>{r.label}</h5>
                            <TipRow label="seeds" value={r.seeds} />
                            <TipRow label="held-out progress, median" value={`${(r.median * 100).toFixed(1)} %`} />
                            <TipRow label="middle half of the seeds" value={`${(r.q1 * 100).toFixed(1)} to ${(r.q3 * 100).toFixed(1)} %`} />
                            <TipRow label="training starts finished" value={`${(r.train * 100).toFixed(0)} %`} />
                          </>
                        ),
                      })
                    }
                    onMouseLeave={() => setTip(null)}
                  >
                    <rect x={0} y={cy - rowH / 2} width={box.width} height={rowH} className="row-hit" />
                    <text x={box.left - 12} y={cy + 4} textAnchor="end" className="row-label">
                      {r.label}
                    </text>
                    {r.seeds > 1 && <line x1={x(r.q1)} x2={x(r.q3)} y1={cy} y2={cy} className="iqr" />}
                    {r.values.map((v, i) => (
                      <circle key={i} cx={x(v)} cy={cy} r={3.5} className="seed-dot" />
                    ))}
                    <line x1={x(r.median)} x2={x(r.median)} y1={cy - 9} y2={cy + 9} className="median-tick" />
                    <text x={box.right + 14} y={cy + 4} className="tick">
                      {(r.train * 100).toFixed(0)} %{r.seeds > 1 ? ` · ${r.seeds} seeds` : " · 1 run"}
                    </text>
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

/** Median curves with their middle-half bands, one colour per group (production: CompareChart). */
function BandChart({ answer, hidden, showRuns, logX }: { answer: CompareResponse; hidden: Set<string>; showRuns: boolean; logX: boolean }) {
  const [hover, setHover] = useState<{ unit: number; x: number; y: number } | null>(null);
  const groups = answer.groups.map((g, k) => ({ ...g, color: series(k) })).filter((g) => !hidden.has(g.group_id));
  const finite = groups.flatMap((g) => [...g.q1, ...g.q3]).filter((v): v is number => v !== null && v > 0 && Number.isFinite(v));
  if (groups.length === 0 || finite.length === 0) {
    return (
      <div className="empty" style={{ minHeight: 300 }}>
        <span className="empty-title">Nothing ticked</span>
        <span>Tick a group in the list to draw it.</span>
      </div>
    );
  }
  const lo = 10 ** Math.floor(Math.log10(Math.min(...finite)));
  const hi = 10 ** Math.ceil(Math.log10(Math.max(...finite)));
  const maxUnit = Math.max(...groups.map((g) => g.unit[g.unit.length - 1]));
  return (
    <>
      <Plot height={340} margin={{ top: 14, right: 150, bottom: 40, left: 54 }} label="Median and middle half of each group" testId="compare-chart">
        {(box) => {
          const x = logX ? logScale(1, maxUnit + 1, box.left, box.right) : linear(0, maxUnit, box.left, box.right);
          const px = (u: number) => x(logX ? u + 1 : u);
          const y = logScale(lo, hi, box.bottom, box.top);
          const safe = (v: number | null) => (v === null || !Number.isFinite(v) || v <= 0 ? NaN : y(v));
          const labels = groups
            .map((g) => ({ id: g.group_id, color: g.color, y: safe(g.median[g.median.length - 1]) }))
            .filter((l) => Number.isFinite(l.y))
            .sort((a, b) => a.y - b.y);
          for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 14);
          const nearest = (u: number, units: number[]) => units.reduce((best, v, i) => (Math.abs(v - u) < Math.abs(units[best] - u) ? i : best), 0);
          return (
            <>
              <YAxis box={box} scale={y} ticks={logTicks(lo, hi)} format={formatPower} label={answer.metric === "val_loss" ? "validation loss (log)" : "training loss (log)"} />
              <XAxis
                box={box}
                scale={px}
                ticks={logX ? [0, 9, 99, 999].filter((t) => t <= maxUnit) : niceTicks(0, maxUnit, 6)}
                format={(v) => String(logX ? v + 1 : v)}
                label={logX ? "epoch (log)" : "epoch"}
              />
              {groups.map((g) => {
                const xs = g.unit.map(px);
                return (
                  <g key={g.group_id}>
                    <path d={bandPath(xs, g.q1.map(safe), g.q3.map(safe))} className="band" style={{ fill: g.color }} data-testid="compare-band" />
                    {showRuns &&
                      answer.runs
                        .filter((r) => r.group_id === g.group_id)
                        .map((r) => <path key={r.run_id} d={linePath(r.unit.map(px), r.values.map(safe))} className="line line-run" style={{ stroke: g.color }} />)}
                    <path d={linePath(xs, g.median.map(safe))} className="line" style={{ stroke: g.color }} data-testid="compare-median" />
                  </g>
                );
              })}
              {labels.map((l) => (
                <g key={l.id}>
                  <circle cx={box.right + 8} cy={l.y} r={3.5} style={{ fill: l.color }} />
                  <text x={box.right + 15} y={l.y + 3.5} className="end-label">
                    {nameOf(l.id)}
                  </text>
                </g>
              ))}
              {hover !== null && <line x1={px(hover.unit)} x2={px(hover.unit)} y1={box.top} y2={box.bottom} className="crosshair" />}
              {hover !== null &&
                groups.map((g) => {
                  const i = nearest(hover.unit, g.unit);
                  const cy = safe(g.median[i]);
                  return Number.isFinite(cy) ? <circle key={g.group_id} cx={px(g.unit[i])} cy={cy} r={4.5} className="marker" style={{ fill: g.color }} /> : null;
                })}
              <rect
                x={box.left}
                y={box.top}
                width={box.right - box.left}
                height={box.bottom - box.top}
                className="hit"
                onMouseMove={(e) => {
                  const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
                  const f = (e.clientX - rect.left) / rect.width;
                  const u = logX ? (maxUnit + 1) ** f - 1 : f * maxUnit;
                  const units = groups[0].unit;
                  setHover({ unit: units[nearest(u, units)], x: e.clientX, y: e.clientY });
                }}
                onMouseLeave={() => setHover(null)}
              />
            </>
          );
        }}
      </Plot>
      <Tip
        tip={
          hover === null
            ? null
            : {
                x: hover.x,
                y: hover.y,
                content: (
                  <>
                    <h5>Epoch {hover.unit} · median of the seeds</h5>
                    {groups.map((g) => {
                      const i = g.unit.reduce((best, v, k) => (Math.abs(v - hover.unit) < Math.abs(g.unit[best] - hover.unit) ? k : best), 0);
                      const v = g.median[i];
                      return <TipRow key={g.group_id} color={g.color} label={nameOf(g.group_id)} value={v === null ? "–" : v.toExponential(2)} />;
                    })}
                  </>
                ),
              }
        }
      />
    </>
  );
}

/**
 * Compare runs. Two rules shape the page:
 *   1. one objective per chart: a loss chart holds only losses of one task;
 *   2. methods meet only in the Driving Exam, at the top.
 */
export function Compare() {
  const data = useApp((s) => s.data);
  const [which, setWhich] = useState<"optimizers" | "imitation">("optimizers");
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [showRuns, setShowRuns] = useState(false);
  if (data === null) return <div className="skeleton" style={{ height: 400, margin: 16 }} />;
  const answer = data.compare[which];
  const toggle = (id: string) =>
    setHidden((old) => {
      const next = new Set(old);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="dash" data-testid="compare">
      <div className="panel span-12">
        <div className="panel-body">
          <Figure
            title="Every method on the Driving Exam"
            note="Held-out tracks: hairpins, sweeper, Monza counter-clockwise and the mirrored loop. Tick = median over the seeds, bar = their middle half."
            objective="exam"
            actions={<Planned>roadmap task 8.1</Planned>}
            testId="exam-figure"
          >
            <ExamStrip groups={data.exams} />
          </Figure>
        </div>
      </div>

      <div className="panel span-3">
        <div className="panel-body">
          <div className="figure-head">
            <div className="figure-titles">
              <span className="figure-title">Runs to compare</span>
              <span className="figure-note">One task and one objective at a time.</span>
            </div>
          </div>
          <Segmented
            label="Comparison"
            value={which}
            onChange={(id) => {
              setWhich(id);
              setHidden(new Set());
            }}
            options={[
              { id: "optimizers", label: "Optimizers" },
              { id: "imitation", label: "Teachers" },
            ]}
          />
          <ul className="grouplist" data-testid="runlist">
            {answer.groups.map((g, k) => (
              <li key={g.group_id}>
                <label className="grouplist-row">
                  <input type="checkbox" checked={!hidden.has(g.group_id)} onChange={() => toggle(g.group_id)} />
                  <i className="grouplist-key" style={{ background: series(k) }} />
                  <span className="grouplist-name">{nameOf(g.group_id)}</span>
                  <span className="grouplist-meta num">{g.run_ids.length} seeds</span>
                </label>
              </li>
            ))}
          </ul>
          <label className="check">
            <input type="checkbox" checked={showRuns} onChange={(e) => setShowRuns(e.target.checked)} /> Draw every seed&apos;s own curve
          </label>
          <p className="figure-foot">
            {which === "optimizers"
              ? "Study S3: four optimizers on the toy sine task, each at its best learning rate, 10 fresh seeds."
              : "Study S5c: the same network cloned from different teachers, 10 seeds each."}
          </p>
        </div>
      </div>

      <div className="panel span-9">
        <div className="panel-body">
          <Figure
            title={which === "optimizers" ? "Toy sine: loss after each epoch" : "Imitation: validation loss after each epoch"}
            note="Line = median of the seeds. Band = their middle half (first to third quartile)."
            objective="loss"
            testId="loss-figure"
          >
            <BandChart answer={answer} hidden={hidden} showRuns={showRuns} logX={which === "optimizers"} />
          </Figure>
          <table className="table compare-table" data-testid="compare-table">
            <thead>
              <tr>
                <th>Group</th>
                <th className="r">Seeds</th>
                <th className="r">Final loss, median</th>
                <th className="r">Middle half</th>
                <th className="r">Diverged</th>
              </tr>
            </thead>
            <tbody>
              {answer.groups.map((g, k) => (
                <tr key={g.group_id} style={{ opacity: hidden.has(g.group_id) ? 0.4 : 1 }}>
                  <td>
                    <span className="grouplist-cell">
                      <i className="grouplist-key" style={{ background: series(k) }} />
                      {nameOf(g.group_id)}
                    </span>
                  </td>
                  <td className="r num">{g.run_ids.length}</td>
                  <td className="r num">{g.final_median === null ? "–" : g.final_median.toExponential(2)}</td>
                  <td className="r num">
                    {g.final_q1 === null || g.final_q3 === null ? "–" : `${g.final_q1.toExponential(2)} to ${g.final_q3.toExponential(2)}`}
                  </td>
                  <td className="r num">{g.n_diverged}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
