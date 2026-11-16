import { useEffect, useState } from "react";

import { num, useApp } from "../../state/store";
import { Button, Chip, Field, Select, Stat } from "../../ui/controls";
import { Figure, linear, linePath, niceTicks, Plot, XAxis, YAxis } from "./charts";

const CONFIGS = ["ga.yaml", "random_search.yaml", "imitation.yaml", "dagger.yaml", "toy_sine.yaml", "toy_xor.yaml"];
const METRICS = [
  { value: "best", label: "best fitness" },
  { value: "median", label: "median fitness" },
  { value: "diversity", label: "diversity" },
] as const;

/**
 * The Training tool: start a run, watch it grow, stop it. In the prototype the "running" run is
 * the recorded GA run played back one generation at a time; in production the numbers arrive
 * over /ws/runs as the training process writes them (task 5.8).
 */
export function Training() {
  const data = useApp((s) => s.data);
  const [unit, setUnit] = useState(11);
  const [running, setRunning] = useState(true);
  const [metric, setMetric] = useState<(typeof METRICS)[number]["value"]>("best");
  const total = data?.run.evolution.unit.length ?? 30;

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      setUnit((u) => {
        if (u + 1 >= total) setRunning(false);
        return Math.min(total, u + 1);
      });
    }, 1100);
    return () => window.clearInterval(timer);
  }, [running, total]);

  if (data === null) return <div className="skeleton" style={{ height: 400, margin: 16 }} />;
  const view = data.run.evolution;
  const values = view[metric].slice(0, unit);
  const units = view.unit.slice(0, unit);
  const done = unit >= total;
  const status = done ? "completed" : running ? "running" : "stopped";
  const exams = [
    { unit: data.run.generation, card: data.run.exam },
    { unit: total, card: data.run.examFinal },
  ].filter((e) => e.unit <= unit);
  const top = metric === "diversity" ? Math.ceil(Math.max(...view.diversity)) : Math.ceil(Math.max(...view.best) * 5.4) / 5;
  const others = [
    { name: "imitation-s5c-efull-s9", method: "imitation", note: "300 of 300 epochs", status: "completed" },
    { name: "dagger-s5c-dw-s9", method: "dagger", note: "10 of 10 rounds", status: "completed" },
    { name: "random_search-e2estop", method: "random search", note: "stopped at unit 3 of 200", status: "stopped" },
  ];

  return (
    <div className="dash" data-testid="training">
      <div className="panel span-4">
        <div className="panel-body">
          <div className="figure-head">
            <div className="figure-titles">
              <span className="figure-title">Start a training run</span>
              <span className="figure-note">Runs as its own process: it keeps going if the server stops. At most 4 at once.</span>
            </div>
          </div>
          <form
            className="launch"
            onSubmit={(e) => {
              e.preventDefault();
              setUnit(0);
              setRunning(true);
            }}
            data-testid="launch-form"
          >
            <Field label="Config">
              <Select value="ga.yaml" options={CONFIGS.map((c) => ({ value: c, label: c }))} onChange={() => {}} />
            </Field>
            <Field label="Seed">
              <input className="input" defaultValue={0} />
            </Field>
            <Field label="Tag">
              <input className="input input-wide" defaultValue="lineage78" />
            </Field>
            <Field label="Generations">
              <input className="input" defaultValue={30} />
            </Field>
            <button type="submit" className="btn btn-primary" data-testid="launch-start">
              Start training
            </button>
          </form>

          <span className="eyebrow launch-listhead">Runs</span>
          <ul className="runlist" data-testid="training-runs">
            <li className="runlist-row" data-selected="true">
              <div className="runlist-main">
                <span className="runlist-name">ga-lineage78</span>
                <Chip kind={status === "running" ? "run" : status === "completed" ? "ok" : "idle"}>{status}</Chip>
              </div>
              <div className="progress" role="progressbar" aria-valuenow={unit} aria-valuemin={0} aria-valuemax={total}>
                <span style={{ width: `${(100 * unit) / total}%` }} data-live={running && !done} />
              </div>
              <div className="runlist-main">
                <span className="runlist-meta num">
                  generation {unit} of {total}
                </span>
                {running && !done && (
                  <Button variant="ghost" onClick={() => setRunning(false)} title="Finish the current generation, examine it, then stop" testId="training-stop">
                    Stop
                  </Button>
                )}
              </div>
            </li>
            {others.map((r) => (
              <li className="runlist-row" key={r.name}>
                <div className="runlist-main">
                  <span className="runlist-name">{r.name}</span>
                  <Chip kind={r.status === "completed" ? "ok" : "idle"}>{r.status}</Chip>
                </div>
                <span className="runlist-meta">
                  {r.method} · {r.note}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="panel span-8">
        <div className="panel-body">
          <Figure
            title="ga-lineage78, as it trains"
            note="Rings: generations whose champion took the Driving Exam."
            objective={metric === "diversity" ? "diversity" : "fitness"}
            actions={<Select value={metric} options={METRICS.map((m) => ({ value: m.value, label: m.label }))} onChange={setMetric} label="Metric" />}
            testId="live-figure"
          >
            <div className="live-stats">
              <Stat label="Generation" value={unit} unit={`of ${total}`} />
              <Stat label={METRICS.find((m) => m.value === metric)!.label} value={values.length === 0 ? "–" : num(values[values.length - 1], 3)} />
              <Stat label="Finish every start" value={unit === 0 ? "–" : view.finishers[unit - 1]} unit={`of ${view.population}`} />
              <Stat label="Networks evaluated" value={(unit * view.population).toLocaleString("en")} />
            </div>
            {unit === 0 ? (
              <div className="empty" style={{ minHeight: 260 }} data-testid="live-waiting">
                <span className="skeleton" style={{ width: 180, height: 10 }} />
                <span className="empty-title">Waiting for the first generation</span>
                <span>The chart starts when the run writes its first line of metrics.</span>
              </div>
            ) : (
              <Plot height={260} margin={{ top: 14, right: 20, bottom: 36, left: 46 }} label="The watched run's metric so far" testId="live-chart">
                {(box) => {
                  const x = linear(0.5, total + 0.5, box.left, box.right);
                  const y = linear(0, top, box.bottom, box.top);
                  return (
                    <>
                      <YAxis box={box} scale={y} ticks={niceTicks(0, top, 5)} />
                      <XAxis box={box} scale={x} ticks={view.unit.filter((u) => u === 1 || u % 5 === 0)} label="generation" />
                      <rect x={x(unit + 0.5)} y={box.top} width={Math.max(0, box.right - x(unit + 0.5))} height={box.bottom - box.top} className="pending" />
                      <path d={linePath(units.map(x), values.map(y))} className="line" style={{ stroke: "var(--series-1)" }} />
                      {exams.map((e) => (
                        <circle key={e.unit} cx={x(e.unit)} cy={y(values[e.unit - 1])} r={4.5} className="checkpoint" />
                      ))}
                      <circle cx={x(unit)} cy={y(values[unit - 1])} r={4.5} className="marker" style={{ fill: "var(--series-1)" }} data-live={running && !done} />
                    </>
                  );
                }}
              </Plot>
            )}
          </Figure>
          <Figure title="Driving Exams so far" objective="exam">
            {exams.length === 0 ? (
              <p className="figure-foot">No checkpoint has been examined yet. Each new best network is examined as it appears.</p>
            ) : (
              <table className="table" data-testid="exam-table">
                <thead>
                  <tr>
                    <th className="r">Generation</th>
                    <th className="r">Training: finished</th>
                    <th className="r">Held-out: finished</th>
                    <th className="r">Held-out: crashed</th>
                    <th className="r">Held-out: progress, median</th>
                  </tr>
                </thead>
                <tbody>
                  {exams.map((e) => (
                    <tr key={e.unit}>
                      <td className="r num">{e.unit}</td>
                      <td className="r num">{(100 * e.card.training.completion_rate).toFixed(0)} %</td>
                      <td className="r num">{(100 * e.card.heldout.completion_rate).toFixed(0)} %</td>
                      <td className="r num">{(100 * e.card.heldout.crash_rate).toFixed(0)} %</td>
                      <td className="r num">{(100 * e.card.heldout.progress_frac_median).toFixed(0)} %</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Figure>
        </div>
      </div>
    </div>
  );
}
