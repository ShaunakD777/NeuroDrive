import { type FormEvent, useEffect, useMemo, useState } from "react";

import type { ConfigInfo } from "@prod/api/generated";
import { fetchConfigs, launchRun, stopRun } from "@prod/api/rest";
import { defaultMetric, plottableKeys, runsNewestFirst, series, statusLabel } from "@prod/state/runsWatch";

import { useRunsSocket } from "../../source/runs";
import { num, shortRun, useApp } from "../../state/store";
import { Button, Chip, Field, Select, Stat } from "../../ui/controls";
import { Figure, linear, linePath, niceTicks, Plot, XAxis, YAxis } from "./charts";

const CONFIGS = ["ga.yaml", "random_search.yaml", "imitation.yaml", "dagger.yaml", "toy_sine.yaml", "toy_xor.yaml"];
const METRICS = [
  { value: "best", label: "best fitness" },
  { value: "median", label: "median fitness" },
  { value: "diversity", label: "diversity" },
] as const;

/**
 * The Training tool, recorded: the recorded GA run played back one generation at a time, so the
 * page can be seen moving without a server. The Live source shows the real thing.
 */
function RecordedTraining() {
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

/** A whole number typed in a box, or null when the box is empty (the config's own value is used). */
function wholeOrNull(text: string): number | null {
  const n = Number(text.trim());
  return text.trim() === "" || !Number.isInteger(n) ? null : n;
}

const KIND: Record<string, "run" | "ok" | "bad" | "idle"> = { running: "run", completed: "ok", diverged: "bad", failed: "bad" };

/**
 * The Training tool, live (task 5.8): start `nd train` with a config from configs/ as its own
 * process, watch its metrics lines, checkpoints and exams arrive over /ws/runs, stop it after the
 * current unit. The same endpoints and socket as production's TrainingRuns.
 */
function LiveTraining() {
  const [configs, setConfigs] = useState<ConfigInfo[] | null>(null);
  const [configName, setConfigName] = useState("");
  const [seed, setSeed] = useState("");
  const [tag, setTag] = useState("");
  const [length, setLength] = useState("");
  const [watched, setWatched] = useState<string | null>(null);
  const [metric, setMetric] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);
  const { watch, status } = useRunsSocket(watched);

  useEffect(() => {
    fetchConfigs()
      .then((list) => {
        setConfigs(list.configs);
        setConfigName((c) => c || (list.configs.find((x) => x.method === "ga") ?? list.configs[0])?.name || "");
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const runs = runsNewestFirst(watch);
  // Watch the newest running run (else the newest run) until the user picks one.
  useEffect(() => {
    if (watched === null && runs.length > 0) setWatched((runs.find((r) => r.status === "running") ?? runs[0]).run_id);
  }, [watched, runs]);

  const config = configs?.find((c) => c.name === configName) ?? null;
  const current = watched !== null ? (watch.runs[watched] ?? null) : null;
  const keys = plottableKeys(watch.records);
  const shown = metric !== null && keys.includes(metric) ? metric : defaultMetric(keys);
  const drawn = useMemo(() => (shown === null ? { unit: [], value: [] } : series(watch.records, shown)), [watch.records, shown]);
  const total = Math.max(current?.budget_units ?? 0, drawn.unit[drawn.unit.length - 1] ?? 0, 1);
  const live = current?.status === "running" && current.exit_code === null;
  const lengthLabel = config?.length_field === "epochs" ? "Epochs" : config?.method === "ga" ? "Generations" : "Units";

  function onLaunch(event: FormEvent) {
    event.preventDefault();
    if (config === null) return;
    const n = wholeOrNull(length);
    setLaunching(true);
    setError(null);
    launchRun({
      config: config.name,
      seed: wholeOrNull(seed),
      tag: tag.trim() === "" ? null : tag.trim(),
      epochs: config.length_field === "epochs" ? n : null,
      units: config.length_field === "units" ? n : null,
    })
      .then((answer) => setWatched(answer.run_id))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLaunching(false));
  }

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
          <form className="launch" onSubmit={onLaunch} data-testid="launch-form">
            <Field label="Config">
              <Select value={configName} options={(configs ?? []).map((c) => ({ value: c.name, label: c.name }))} onChange={setConfigName} disabled={!configs?.length} label="Config" />
            </Field>
            <Field label="Seed">
              <input className="input" placeholder={config ? String(config.seed) : ""} value={seed} onChange={(e) => setSeed(e.target.value)} data-testid="launch-seed" />
            </Field>
            <Field label="Tag">
              <input className="input input-wide" placeholder={config?.tag ?? ""} value={tag} onChange={(e) => setTag(e.target.value)} data-testid="launch-tag" />
            </Field>
            <Field label={lengthLabel}>
              <input
                className="input"
                placeholder={config === null || config.length === null ? "" : String(config.length)}
                value={length}
                onChange={(e) => setLength(e.target.value)}
                data-testid="launch-length"
              />
            </Field>
            <button type="submit" className="btn btn-primary" disabled={config === null || launching} data-testid="launch-start">
              {launching ? "Starting…" : "Start training"}
            </button>
          </form>
          {error !== null && (
            <p className="figure-foot launch-error" role="alert" data-testid="launch-error">
              {error}
            </p>
          )}

          <span className="eyebrow launch-listhead">Runs{status !== "connected" ? ` · ${status}` : ""}</span>
          <ul className="runlist" data-testid="training-runs">
            {runs.length === 0 && (
              <li className="runlist-row muted" data-testid="training-none">
                No run is training. This list shows runs that are running now, and runs started here since the server started; every
                finished run is in the Timeline.
              </li>
            )}
            {runs.slice(0, 40).map((r) => {
              const running = r.status === "running" && r.exit_code === null;
              const budget = r.budget_units ?? 0;
              return (
                <li key={r.run_id} className="runlist-row" data-selected={r.run_id === watched} onClick={() => setWatched(r.run_id)}>
                  <div className="runlist-main">
                    <span className="runlist-name">{shortRun(r.run_id)}</span>
                    <Chip kind={KIND[r.status] ?? "idle"}>{statusLabel(r)}</Chip>
                  </div>
                  {budget > 0 && (
                    <div className="progress" role="progressbar" aria-valuenow={r.units_done} aria-valuemin={0} aria-valuemax={budget}>
                      <span style={{ width: `${Math.min(100, (100 * r.units_done) / budget)}%` }} data-live={running} />
                    </div>
                  )}
                  <div className="runlist-main">
                    <span className="runlist-meta num">
                      {r.method.replace(/_/g, " ")} · unit {r.units_done}
                      {budget > 0 ? ` of ${budget}` : ""}
                    </span>
                    {running && !r.stop_requested && (
                      <Button
                        variant="ghost"
                        onClick={() => {
                          stopRun(r.run_id).catch((e: Error) => setError(e.message));
                        }}
                        title="Finish the current unit, examine it, then stop"
                        testId="training-stop"
                      >
                        Stop
                      </Button>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      </div>

      <div className="panel span-8">
        <div className="panel-body">
          <Figure
            title={watched === null ? "Choose a run" : `${shortRun(watched)}, as it trains`}
            note="Rings: units whose checkpoint took the Driving Exam."
            objective={shown === null ? undefined : shown.includes("loss") ? "loss" : shown.includes("fitness") ? "fitness" : undefined}
            actions={
              keys.length > 1 && shown !== null ? <Select value={shown} options={keys.map((k) => ({ value: k, label: k.replace(/_/g, " ") }))} onChange={setMetric} label="Metric" /> : undefined
            }
            testId="live-figure"
          >
            <div className="live-stats">
              <Stat label="Unit" value={current?.units_done ?? "–"} unit={current?.budget_units ? `of ${current.budget_units}` : undefined} />
              <Stat label={shown === null ? "metric" : shown.replace(/_/g, " ")} value={drawn.value.length === 0 ? "–" : num(drawn.value[drawn.value.length - 1], 4)} />
              <Stat label="Checkpoints" value={watch.checkpoints.length} />
              <Stat label="Exams" value={watch.exams.length} />
            </div>
            {drawn.unit.length === 0 ? (
              <div className="empty" style={{ minHeight: 260 }} data-testid="live-waiting">
                <span className="skeleton" style={{ width: 180, height: 10 }} />
                <span className="empty-title">{watched === null ? "No run chosen" : "Waiting for the first metrics line"}</span>
                <span>The chart starts when the run writes its first line of metrics.</span>
              </div>
            ) : (
              <Plot height={260} margin={{ top: 14, right: 20, bottom: 36, left: 54 }} label="The watched run's metric so far" testId="live-chart">
                {(box) => {
                  const lo = Math.min(0, ...drawn.value);
                  const hi = Math.max(...drawn.value);
                  const top = hi + (hi - lo) * 0.08 || 1;
                  const x = linear(0, total, box.left, box.right);
                  const y = linear(lo, top, box.bottom, box.top);
                  const lastUnit = drawn.unit[drawn.unit.length - 1];
                  const lastValue = drawn.value[drawn.value.length - 1];
                  return (
                    <>
                      <YAxis box={box} scale={y} ticks={niceTicks(lo, top, 5)} />
                      <XAxis box={box} scale={x} ticks={niceTicks(0, total, 6)} label="unit" />
                      <rect x={x(lastUnit)} y={box.top} width={Math.max(0, box.right - x(lastUnit))} height={box.bottom - box.top} className="pending" />
                      <path d={linePath(drawn.unit.map(x), drawn.value.map(y))} className="line" style={{ stroke: "var(--series-1)" }} />
                      {watch.exams.map((e) => {
                        const i = drawn.unit.indexOf(e.unit);
                        return i < 0 ? null : <circle key={e.unit} cx={x(e.unit)} cy={y(drawn.value[i])} r={4.5} className="checkpoint" />;
                      })}
                      <circle cx={x(lastUnit)} cy={y(lastValue)} r={4.5} className="marker" style={{ fill: "var(--series-1)" }} data-live={live} />
                    </>
                  );
                }}
              </Plot>
            )}
          </Figure>
          <Figure title="Driving Exams so far" objective="exam">
            {watch.exams.length === 0 ? (
              <p className="figure-foot">No checkpoint of this run has been examined yet.</p>
            ) : (
              <table className="table" data-testid="exam-table">
                <thead>
                  <tr>
                    <th className="r">Unit</th>
                    <th className="r">Training: finished</th>
                    <th className="r">Held-out: finished</th>
                    <th className="r">Held-out: crashed</th>
                    <th className="r">Held-out: progress, median</th>
                  </tr>
                </thead>
                <tbody>
                  {watch.exams.map((e) => (
                    <tr key={e.unit}>
                      <td className="r num">{e.unit}</td>
                      <td className="r num">{(100 * e.training.completion_rate).toFixed(0)} %</td>
                      <td className="r num">{(100 * e.heldout.completion_rate).toFixed(0)} %</td>
                      <td className="r num">{(100 * e.heldout.crash_rate).toFixed(0)} %</td>
                      <td className="r num">{(100 * e.heldout.progress_frac_median).toFixed(0)} %</td>
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

/** The Training tool: start, watch and stop training runs. */
export function Training() {
  const source = useApp((s) => s.source);
  return source === "live" ? <LiveTraining /> : <RecordedTraining />;
}
