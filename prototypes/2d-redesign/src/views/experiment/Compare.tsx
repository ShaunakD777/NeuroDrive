import { useEffect, useMemo, useState } from "react";

import type { RunSummary } from "@prod/api/generated";
import { fetchCompare } from "@prod/api/rest";

import { quantile } from "../../mock/mockServer";
import type { CompareResponse, ExamGroup } from "../../mock/types";
import { methodLabel, useAnswer, useRunList } from "../../source/runs";
import { shortRun, useApp } from "../../state/store";
import { Planned, Segmented } from "../../ui/controls";
import { SourceWaiting } from "../../ui/SourceWaiting";
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

/** The chart, its legend and its table: the same for both sources. */
function LossPanel({ answer, title, hidden, showRuns, logX, foot }: { answer: CompareResponse; title: string; hidden: Set<string>; showRuns: boolean; logX: boolean; foot?: string }) {
  return (
    <div className="panel span-9">
      <div className="panel-body">
        <Figure title={title} note="Line = median of the seeds. Band = their middle half (first to third quartile)." objective="loss" testId="loss-figure">
          <BandChart answer={answer} hidden={hidden} showRuns={showRuns} logX={logX} />
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
                <td className="r num">{g.final_q1 === null || g.final_q3 === null ? "–" : `${g.final_q1.toExponential(2)} to ${g.final_q3.toExponential(2)}`}</td>
                <td className="r num">{g.n_diverged}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {foot && <p className="figure-foot">{foot}</p>}
      </div>
    </div>
  );
}

/**
 * Compare runs, recorded: two comparisons recorded from `GET runs/compare` (studies S3 and S5c),
 * and the exam of every method, which the recording gathered card by card.
 */
function RecordedCompare() {
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

      <LossPanel
        answer={answer}
        title={which === "optimizers" ? "Toy sine: loss after each epoch" : "Imitation: validation loss after each epoch"}
        hidden={hidden}
        showRuns={showRuns}
        logX={which === "optimizers"}
      />
    </div>
  );
}

/** Methods whose runs write a loss after each epoch (`GET runs/compare` accepts only these). */
const hasLoss = (method: string) => method.startsWith("toy_") || method === "imitation" || method === "dagger";

interface ListGroup {
  /** The sweep id, or the run id of a run outside any sweep. */
  key: string;
  label: string;
  runs: RunSummary[];
}

/** The runs of one method, as sweeps (with their seeds) and single runs, newest first. */
function groupBySweep(runs: RunSummary[]): ListGroup[] {
  const groups = new Map<string, ListGroup>();
  for (const r of runs) {
    const key = r.sweep ?? r.run_id;
    const group = groups.get(key) ?? { key, label: r.sweep ? r.tag.replace(/-s\d+$/, "") : shortRun(r.run_id), runs: [] };
    group.runs.push(r);
    groups.set(key, group);
  }
  return [...groups.values()];
}

/**
 * Compare runs, live: the server's run folders grouped by method and sweep, with a search box. One
 * task at a time: ticking a run of another task starts a new comparison, because a loss chart
 * holds only losses of one task. The medians and quartiles are Python's (`GET runs/compare`).
 */
function LiveCompare() {
  const { runs, error } = useRunList();
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [metric, setMetric] = useState<"loss" | "val_loss">("loss");
  const [showRuns, setShowRuns] = useState(false);
  const [logX, setLogX] = useState(true);
  const lossRuns = useMemo(() => (runs ?? []).filter((r) => hasLoss(r.method)), [runs]);
  const methods = useMemo(() => [...new Set(lossRuns.map((r) => r.method))], [lossRuns]);
  const others = (runs ?? []).filter((r) => !hasLoss(r.method)).length;

  // Open on the newest sweep, so the page starts with a comparison rather than an empty chart.
  useEffect(() => {
    if (chosen.size > 0 || lossRuns.length === 0) return;
    const sweep = lossRuns.find((r) => r.sweep !== null)?.sweep;
    if (sweep) setChosen(new Set(lossRuns.filter((r) => r.sweep === sweep).map((r) => r.run_id)));
  }, [lossRuns, chosen.size]);

  const chosenRuns = lossRuns.filter((r) => chosen.has(r.run_id));
  const task = chosenRuns[0]?.method ?? null;
  const canVal = task === "imitation" || task === "dagger";
  const shownMetric = canVal ? metric : "loss";
  const ids = chosenRuns.map((r) => r.run_id).sort();
  const answer = useAnswer(ids.length === 0 ? null : `compare/${shownMetric}/${ids.join(",")}`, () => fetchCompare(ids, shownMetric));

  /** Tick or untick a group; a group of another task replaces the selection. */
  const toggle = (group: ListGroup) => {
    const groupIds = group.runs.map((r) => r.run_id);
    const on = groupIds.every((id) => chosen.has(id));
    const otherTask = task !== null && group.runs[0].method !== task;
    setChosen((old) => {
      const next = otherTask ? new Set<string>() : new Set(old);
      for (const id of groupIds) {
        if (on && !otherTask) next.delete(id);
        else next.add(id);
      }
      return next;
    });
  };

  if (error !== null)
    return (
      <div className="empty" style={{ minHeight: 300 }}>
        <span className="empty-title">The run list could not be loaded</span>
        <span>{error}</span>
      </div>
    );
  if (runs === null) return <SourceWaiting what="the run folders" height={400} />;
  const q = query.trim().toLowerCase();

  return (
    <div className="dash" data-testid="compare">
      <div className="panel span-12">
        <div className="panel-body">
          <Figure title="Every method on the Driving Exam" objective="exam" actions={<Planned>roadmap task 8.1</Planned>} testId="exam-figure">
            <div className="empty spec-only" style={{ minHeight: 120 }}>
              <span>
                Comparing methods needs every seed&apos;s exam gathered and summarised by Python: <code>POST compare</code>, planned in{" "}
                <code>ARCHITECTURE.md</code> §7 and built with roadmap task 8.1. The browser does not compute those medians. The recorded source shows
                the intended chart.
              </span>
            </div>
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
          <input className="input search" placeholder="Search runs" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search runs" data-testid="run-search" />
          {canVal && (
            <Segmented
              label="Loss"
              value={metric}
              onChange={setMetric}
              options={[
                { id: "loss", label: "Training" },
                { id: "val_loss", label: "Validation" },
              ]}
            />
          )}
          <div className="grouplist-scroll" data-testid="runlist">
            {methods.map((method) => {
              const groups = groupBySweep(lossRuns.filter((r) => r.method === method && (q === "" || r.run_id.toLowerCase().includes(q))));
              if (groups.length === 0) return null;
              return (
                <div key={method} className="grouplist-method" data-other={task !== null && task !== method}>
                  <span className="eyebrow">{methodLabel(method)}</span>
                  <ul className="grouplist">
                    {groups.map((g) => {
                      const on = g.runs.every((r) => chosen.has(r.run_id));
                      const some = !on && g.runs.some((r) => chosen.has(r.run_id));
                      const k = answer.state === "ready" ? answer.value.groups.findIndex((x) => x.run_ids.some((id) => g.runs.some((r) => r.run_id === id))) : -1;
                      return (
                        <li key={g.key}>
                          <label className="grouplist-row" title={g.runs.map((r) => r.run_id).join("\n")}>
                            <input type="checkbox" checked={on} ref={(el) => {
                                if (el) el.indeterminate = some;
                              }} onChange={() => toggle(g)} />
                            <i className="grouplist-key" style={{ background: on && k >= 0 ? series(k) : "transparent" }} />
                            <span className="grouplist-name">{g.label}</span>
                            <span className="grouplist-meta num">{g.runs.length === 1 ? g.runs[0].status : `${g.runs.length} seeds`}</span>
                          </label>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              );
            })}
          </div>
          <label className="check">
            <input type="checkbox" checked={showRuns} onChange={(e) => setShowRuns(e.target.checked)} /> Draw every seed&apos;s own curve
          </label>
          <label className="check">
            <input type="checkbox" checked={logX} onChange={(e) => setLogX(e.target.checked)} /> Epochs on a log scale
          </label>
          {others > 0 && (
            <p className="figure-foot">
              {others} runs of methods without a loss (random search, GA, baselines) are not listed: they meet the others on the Driving Exam.
            </p>
          )}
        </div>
      </div>

      {answer.state === "ready" ? (
        <LossPanel
          answer={answer.value}
          title={`${task === null ? "" : methodLabel(task)}: ${shownMetric === "val_loss" ? "validation" : "training"} loss after each epoch`}
          hidden={new Set()}
          showRuns={showRuns}
          logX={logX}
          foot={answer.value.groups.some((g) => g.n_diverged > 0) ? "A diverged run has no finite loss: it counts in the table, not in the band." : undefined}
        />
      ) : (
        <div className="panel span-9">
          <div className="panel-body">
            {answer.state === "loading" ? (
              <div className="skeleton" style={{ height: 380 }} />
            ) : (
              <div className="empty" style={{ minHeight: 380 }}>
                <span className="empty-title">{ids.length === 0 ? "Nothing ticked" : "No comparison"}</span>
                <span>{ids.length === 0 ? "Tick a sweep or a run in the list." : answer.why}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Compare runs. Two rules shape the page:
 *   1. one objective per chart: a loss chart holds only losses of one task;
 *   2. methods meet only in the Driving Exam, at the top.
 */
export function Compare() {
  const source = useApp((s) => s.source);
  return source === "live" ? <LiveCompare /> : <RecordedCompare />;
}
