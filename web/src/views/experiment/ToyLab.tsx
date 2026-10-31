import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { RunDetail, RunMetrics, RunSummary, ToyCheckpointResponse } from "../../api/generated";
import {
  fetchCheckpoints,
  fetchRunDetail,
  fetchRunMetrics,
  fetchRuns,
  fetchToyCheckpoint,
} from "../../api/rest";
import { formatNumber, lossCurve, toyRunSummary } from "./charts";
import { LossChart } from "./LossChart";
import { PredictionPlot } from "./PredictionPlot";
import { ToyNetwork } from "./ToyNetwork";

interface RunData {
  runId: string;
  detail: RunDetail;
  metrics: RunMetrics;
  units: number[];
}

/**
 * The toy lab (task 3.8): pick a toy run, see its loss curve, and step through its checkpoints to
 * watch the predictions and the gradient glow change. Runs are made with
 * `uv run python -m neurodrive train --config configs/toy_<line|sine|xor|two_spirals>.yaml`
 * (task 4.6b). Everything shown is
 * computed by the Python server; this page only fetches and draws.
 */
export function ToyLab({ headerExtra }: { headerExtra?: ReactNode }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<RunData | null>(null);
  const [index, setIndex] = useState(0);
  const [toy, setToy] = useState<ToyCheckpointResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const cache = useRef(new Map<string, ToyCheckpointResponse>());

  function loadRuns() {
    fetchRuns()
      .then((r) => {
        const toys = r.runs.filter((s) => s.method.startsWith("toy_"));
        setRuns(toys);
        setRunId((current) => (current && toys.some((s) => s.run_id === current) ? current : (toys[0]?.run_id ?? null)));
      })
      .catch((e: Error) => setError(e.message));
  }
  useEffect(loadRuns, []);

  // A new run: drop the old one at once (so its slider cannot be moved while the new one loads),
  // fetch the new config, metrics and checkpoints, and start at the last checkpoint.
  useEffect(() => {
    setRun(null);
    if (runId === null) return;
    let live = true;
    Promise.all([fetchRunDetail(runId), fetchRunMetrics(runId), fetchCheckpoints(runId)])
      .then(([detail, metrics, list]) => {
        if (!live) return;
        setRun({ runId, detail, metrics, units: list.units });
        setIndex(Math.max(0, list.units.length - 1));
        setError(null);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [runId]);

  // The selected checkpoint's toy view (kept while the next one loads, so nothing flickers).
  const unit = run && run.units.length ? run.units[Math.min(index, run.units.length - 1)] : null;
  useEffect(() => {
    if (run === null || unit === null) return;
    const key = `${run.runId}/${unit}`;
    const hit = cache.current.get(key);
    if (hit) {
      setToy(hit);
      return;
    }
    let live = true;
    fetchToyCheckpoint(run.runId, unit)
      .then((t) => {
        cache.current.set(key, t);
        if (live) setToy(t);
      })
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [run, unit]);

  const curve = useMemo(() => (run ? lossCurve(run.metrics) : { unit: [], loss: [] }), [run]);
  const config = run ? toyRunSummary(run.detail.config as Record<string, unknown>) : undefined;
  const status = run?.detail.meta.status as string | undefined;
  const shown = toy !== null && run !== null && toy.run_id === run.runId ? toy : null;

  return (
    <section className="view toylab" data-testid="toylab">
      <div className="view-header">
        <h2>Experiment: toy lab</h2>
        {headerExtra}
      </div>

      <div className="toylab-controls">
        <label className="picker">
          <span>Run </span>
          <select
            data-testid="run-pick"
            value={runId ?? ""}
            onChange={(e) => setRunId(e.target.value)}
            disabled={!runs?.length}
          >
            {(runs ?? []).map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} ({r.status})
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={loadRuns} data-testid="runs-refresh">
          Refresh list
        </button>
      </div>

      {error && <p className="toylab-error">Could not load: {error}</p>}
      {runs !== null && runs.length === 0 && (
        <p className="hint" data-testid="toylab-empty">
          No toy runs yet. Train one with{" "}
          <code>uv run python -m neurodrive train --config configs/toy_xor.yaml</code>{" "}
          and press Refresh list.
        </p>
      )}

      {run && config && (
        <p className="toylab-summary" data-testid="toylab-summary" data-run-id={run.runId}>
          Task <strong>{config.task}</strong> · network {config.layerSizes.join("-")} (
          {config.hiddenActivation} hidden, {config.outputActivation} output) · optimizer{" "}
          {config.optimizer}, learning rate {config.learningRate} ·{" "}
          {run.metrics.unit.length} epochs · {status}
          {status === "diverged" && " (the curve stops at the last finite loss)"}
        </p>
      )}

      {run && unit !== null && (
        <div className="toylab-slider">
          <label htmlFor="checkpoint-slider">Checkpoint</label>
          <input
            id="checkpoint-slider"
            data-testid="checkpoint-slider"
            type="range"
            min={0}
            max={run.units.length - 1}
            step={1}
            value={Math.min(index, run.units.length - 1)}
            onChange={(e) => setIndex(Number(e.target.value))}
          />
          <span data-testid="checkpoint-label">
            epoch {unit}
            {shown && shown.unit === unit ? ` · loss ${formatNumber(shown.loss)}` : " · loading…"}
          </span>
        </div>
      )}

      {run && (
        <div className="toylab-grid">
          <LossChart
            curve={curve}
            checkpoints={run.units}
            selected={unit}
            onSelect={(u) => setIndex(run.units.indexOf(u))}
          />
          {shown && <PredictionPlot toy={shown} />}
          {shown && <ToyNetwork toy={shown} />}
        </div>
      )}
    </section>
  );
}
