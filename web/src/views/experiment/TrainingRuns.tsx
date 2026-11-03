import { useEffect, useRef, useState, type FormEvent, type ReactNode } from "react";

import type { ConfigInfo, LaunchRequest } from "../../api/generated";
import { fetchConfigs, launchRun, stopRun } from "../../api/rest";
import { connectLive, type ConnectionStatus, type LiveConnection } from "../../api/ws";
import {
  applyRunsMessage,
  defaultMetric,
  initialRunsWatch,
  plottableKeys,
  runsNewestFirst,
  series,
  statusLabel,
  watchRun,
} from "../../state/runsWatch";
import { formatNumber, shortName } from "./charts";
import { LiveChart } from "./LiveChart";

/** A whole number typed into a field, or null when the field is empty (keep the file's value). */
function wholeOrNull(text: string): number | null {
  return text.trim() === "" ? null : Number(text);
}

const pct = (v: number) => `${Math.round(v * 100)}%`;

/**
 * The Training tab (task 5.8, ADR-041): start a run from a config in `configs/` with the overrides
 * `nd train` knows, watch its chart grow while it trains, read each exam card's headline as it
 * appears, and stop it. The server starts the training as a separate process and reports what it
 * writes over `/ws/runs`; this page sends commands and draws what it is told.
 */
export function TrainingRuns({ headerExtra }: { headerExtra?: ReactNode }) {
  const [configs, setConfigs] = useState<ConfigInfo[] | null>(null);
  const [configName, setConfigName] = useState("");
  const [seed, setSeed] = useState("");
  const [tag, setTag] = useState("");
  const [length, setLength] = useState("");
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [watch, setWatch] = useState(initialRunsWatch);
  const [connection, setConnection] = useState<ConnectionStatus>("connecting");
  const [metric, setMetric] = useState<string | null>(null);
  const socket = useRef<LiveConnection | null>(null);
  const watched = useRef<string | null>(null);

  useEffect(() => {
    fetchConfigs()
      .then((list) => {
        setConfigs(list.configs);
        const first = list.configs.find((c) => c.name === "random_search.yaml") ?? list.configs[0];
        if (first) setConfigName(first.name);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    const connection = connectLive(
      (status) => {
        setConnection(status);
        // After a reconnect the server knows nothing of this page: ask for the history again.
        if (status === "connected" && watched.current !== null) send(watched.current);
      },
      (message) => setWatch((s) => applyRunsMessage(s, message)),
      "/ws/runs",
    );
    socket.current = connection;
    return () => {
      socket.current = null;
      connection.stop();
    };
  }, []);

  function send(runId: string | null) {
    socket.current?.send(JSON.stringify({ type: "watch", run_id: runId }));
  }

  function show(runId: string) {
    watched.current = runId;
    setWatch((s) => watchRun(s, runId));
    setMetric(null);
    send(runId);
  }

  const config = configs?.find((c) => c.name === configName) ?? null;

  function onLaunch(event: FormEvent) {
    event.preventDefault();
    if (config === null) return;
    const n = wholeOrNull(length);
    const request: LaunchRequest = {
      config: config.name,
      seed: wholeOrNull(seed),
      tag: tag.trim() === "" ? null : tag.trim(),
      epochs: config.length_field === "epochs" ? n : null,
      units: config.length_field === "units" ? n : null,
    };
    setLaunching(true);
    setError(null);
    launchRun(request)
      .then((answer) => show(answer.run_id))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLaunching(false));
  }

  function onStop(runId: string) {
    setError(null);
    stopRun(runId).catch((e: Error) => setError(e.message));
  }

  const runs = runsNewestFirst(watch);
  const current = watch.watched !== null ? (watch.runs[watch.watched] ?? null) : null;
  const keys = plottableKeys(watch.records);
  const shown = metric !== null && keys.includes(metric) ? metric : defaultMetric(keys);
  const drawn = shown !== null ? series(watch.records, shown) : { unit: [], value: [] };

  return (
    <section className="view toylab" data-testid="training">
      <div className="view-header">
        <h2>Experiment: training</h2>
        {headerExtra}
      </div>

      <form className="toylab-controls training-form" onSubmit={onLaunch} data-testid="launch-form">
        <label className="picker">
          <span>Config </span>
          <select
            data-testid="launch-config"
            value={configName}
            onChange={(e) => setConfigName(e.target.value)}
            disabled={!configs?.length}
          >
            {(configs ?? []).map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        <label className="picker">
          <span>Seed </span>
          <input
            type="number"
            min={0}
            step={1}
            value={seed}
            placeholder={config ? String(config.seed) : ""}
            onChange={(e) => setSeed(e.target.value)}
            data-testid="launch-seed"
          />
        </label>
        <label className="picker">
          <span>Tag </span>
          <input
            type="text"
            value={tag}
            placeholder={config?.tag ?? ""}
            pattern="[A-Za-z0-9_\-]+"
            onChange={(e) => setTag(e.target.value)}
            data-testid="launch-tag"
          />
        </label>
        <label className="picker">
          <span>{config?.length_field ?? "units"} </span>
          <input
            type="number"
            min={1}
            step={1}
            value={length}
            placeholder={config?.length != null ? String(config.length) : ""}
            onChange={(e) => setLength(e.target.value)}
            data-testid="launch-length"
          />
        </label>
        <button type="submit" disabled={config === null || launching || connection !== "connected"} data-testid="launch-start">
          {launching ? "Starting…" : "Start"}
        </button>
        {connection !== "connected" && (
          <span className="camera-note" data-testid="runs-connection">
            {connection === "connecting" ? "Connecting to the server…" : "Server not reachable; retrying…"}
          </span>
        )}
      </form>
      <p className="hint">
        The server runs <code>nd train --config configs/{configName || "<config>"}</code> with these overrides as a
        separate process (empty fields keep the file's values). Runs started in a terminal appear here as well.
      </p>

      {error && (
        <p className="toylab-error" data-testid="launch-error">
          {error}
        </p>
      )}

      <div className="compare-layout">
        <ul className="runlist" data-testid="training-runs">
          {runs.length === 0 && <li className="hint">No runs are training. Choose a config and press Start.</li>}
          {runs.map((r) => {
            const canStop = r.status === "running" && r.exit_code === null && !r.stop_requested;
            return (
              <li
                key={r.run_id}
                className={r.run_id === watch.watched ? "runlist-group training-run training-run-watched" : "runlist-group training-run"}
                data-testid="training-run"
                data-run-id={r.run_id}
                data-status={r.status}
              >
                <div className="runlist-row">
                  <button type="button" className="training-run-name" onClick={() => show(r.run_id)} title={r.run_id}>
                    {shortName(r.run_id)}
                  </button>
                  <span className="runlist-meta" data-testid="training-run-units">
                    {r.units_done}
                    {r.budget_units !== null ? ` / ${r.budget_units}` : ""} units
                  </span>
                </div>
                <div className="runlist-row">
                  <span className="runlist-meta training-run-status" data-testid="training-run-status">
                    {statusLabel(r)}
                  </span>
                  {canStop && (
                    <button type="button" className="training-stop" onClick={() => onStop(r.run_id)} data-testid="training-stop">
                      Stop
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>

        <div className="compare-main">
          {watch.watched === null ? (
            <p className="hint">Start a run, or pick one on the left, to watch it train.</p>
          ) : (
            <>
              <div className="toylab-controls">
                <span className="toylab-summary" data-testid="watched-run">
                  <strong>{watch.watched}</strong>
                  {current && ` · ${statusLabel(current)}`}
                </span>
                {keys.length > 1 && (
                  <label className="picker">
                    <span>Metric </span>
                    <select data-testid="live-metric" value={shown ?? ""} onChange={(e) => setMetric(e.target.value)}>
                      {keys.map((k) => (
                        <option key={k} value={k}>
                          {k.replaceAll("_", " ")}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
              {shown === null ? (
                <p className="hint" data-testid="live-waiting">
                  Waiting for the first unit…
                </p>
              ) : (
                <LiveChart
                  unit={drawn.unit}
                  value={drawn.value}
                  metric={shown}
                  budgetUnits={current?.budget_units ?? null}
                  examUnits={watch.exams.map((e) => e.unit)}
                />
              )}
              {watch.exams.length > 0 && (
                <table className="compare-table" data-testid="exam-table">
                  <caption>Driving Exam at each examined checkpoint (training track; held-out tracks pooled)</caption>
                  <thead>
                    <tr>
                      <th scope="col">Unit</th>
                      <th scope="col">Training: completed</th>
                      <th scope="col">crashed</th>
                      <th scope="col">median progress</th>
                      <th scope="col">Held-out: completed</th>
                      <th scope="col">crashed</th>
                      <th scope="col">median progress</th>
                    </tr>
                  </thead>
                  <tbody>
                    {watch.exams.map((e) => (
                      <tr key={e.unit} data-testid="exam-row" data-unit={e.unit}>
                        <td>{e.unit}</td>
                        <td>{pct(e.training.completion_rate)}</td>
                        <td>{pct(e.training.crash_rate)}</td>
                        <td>{formatNumber(e.training.progress_frac_median)}</td>
                        <td>{pct(e.heldout.completion_rate)}</td>
                        <td>{pct(e.heldout.crash_rate)}</td>
                        <td>{formatNumber(e.heldout.progress_frac_median)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </>
          )}
        </div>
      </div>
    </section>
  );
}
