import { useEffect, useMemo, useRef, useState } from "react";

import type { RunSummary } from "@prod/api/generated";
import { fetchAncestry, fetchEvolution, fetchExamCard, fetchGenerationStarts, fetchRunDetail } from "@prod/api/rest";
import { useLive } from "@prod/state/live";
import { defaultMetric, plottableKeys, type RunsWatchState, series } from "@prod/state/runsWatch";
import { defaultGhost } from "@prod/views/environment/population";

import type { Ancestry, GenerationStarts } from "@prod/api/generated";
import {
  configText,
  drivingRuns,
  type Loadable,
  methodLabel,
  toEvolution,
  toExamCard,
  useAnswer,
  useRunList,
  useRunsSocket,
} from "../../source/runs";
import { useWorld } from "../../source/world";
import { shortRun, useApp } from "../../state/store";
import { Button, Chip, Field, Select } from "../../ui/controls";
import { SourceWaiting } from "../../ui/SourceWaiting";
import { Figure } from "./charts";
import { ExamPanel, GaDashboard, type GaModel, MetricChart, Slot } from "./RunTimeline";

const ready = <T,>(value: T): Loadable<T> => ({ state: "ready", value });
const toEnvironment = () => {
  window.location.hash = "#/environment";
};
/** The largest of `units` that is at most `at`, or null. */
const atOrBefore = (units: readonly number[], at: number) => units.filter((u) => u <= at).reduce<number | null>((m, u) => (m === null || u > m ? u : m), null);

const STATUS_KIND: Record<string, "run" | "ok" | "bad" | "idle"> = { running: "run", completed: "ok", diverged: "bad", failed: "bad" };

/** The run picker and what kind of run it is. */
function RunHead({
  runs,
  runId,
  onPick,
  method,
  status,
  settings,
  disabled,
}: {
  runs: { run_id: string; method: string; checkpoints?: number }[];
  runId: string;
  onPick: (id: string) => void;
  method: string;
  status: string;
  settings: string;
  disabled?: boolean;
}) {
  return (
    <div className="dash-run">
      <Field label="Run">
        <Select
          value={runId}
          options={runs.map((r) => ({ value: r.run_id, label: `${shortRun(r.run_id)} · ${methodLabel(r.method)}` }))}
          onChange={onPick}
          disabled={disabled}
          label="Run"
        />
      </Field>
      <Chip kind="plain" icon={method === "ga" ? "dna" : null}>
        {methodLabel(method)}
      </Chip>
      <Chip kind={STATUS_KIND[status] ?? "idle"}>{status}</Chip>
      <span className="muted">{settings}</span>
    </div>
  );
}

// ── Recorded ────────────────────────────────────────────────────────────────────────────

/** The recorded GA run: its charts are whole, its generation panels exist for one generation. */
function RecordedTimeline() {
  const data = useApp((s) => s.data);
  const generation = useApp((s) => s.generation);
  const evoStart = useApp((s) => s.evoStart);
  const car = useApp((s) => s.car);
  const set = useApp((s) => s.set);
  const selectCar = useApp((s) => s.selectCar);
  if (data === null) return <SourceWaiting what="the recorded run" height={400} />;
  const run = data.run;
  const recorded = run.generation;
  const at = generation === recorded;
  const missing: Loadable<never> = {
    state: "missing",
    why: `Recorded for generation ${recorded} only. With the Live source this panel follows the chosen generation.`,
  };
  const model: GaModel = {
    head: (
      <RunHead
        runs={[{ run_id: run.run_id, method: "ga" }]}
        runId={run.run_id}
        onPick={() => {}}
        method="ga"
        status={run.meta.status}
        settings={configText("ga", run.config as unknown as { [k: string]: unknown })}
        disabled
      />
    ),
    all: run.evolution,
    view: evoStart === null ? run.evolution : run.evolutionStart,
    evoStart,
    startChoices: [run.start],
    setEvoStart: (k) => set({ evoStart: k }),
    raced: run.start,
    checkpoints: run.checkpoints,
    generation,
    pick: (g) => set({ generation: g }),
    starts: at ? ready(run.starts as unknown as GenerationStarts) : missing,
    ancestry: at ? ready(run.ancestry as unknown as Ancestry) : missing,
    exam: at ? ready({ unit: recorded, card: run.exam }) : missing,
    missingAction: { label: `Back to generation ${recorded}`, onClick: () => set({ generation: recorded }) },
    car,
    onCell: (row) => selectCar(row),
    onRace: toEnvironment,
    onReplay: toEnvironment,
  };
  return <GaDashboard model={model} />;
}

// ── Live: a GA run ──────────────────────────────────────────────────────────────────────

function LiveGaTimeline({ runId, head, watch }: { runId: string; head: React.ReactNode; watch: RunsWatchState }) {
  const generation = useApp((s) => s.generation);
  const evoStart = useApp((s) => s.evoStart);
  const set = useApp((s) => s.set);
  const selectCar = useApp((s) => s.selectCar);
  const loadScenario = useLive((s) => s.loadScenario);
  const send = useLive((s) => s.send);
  const raced = useLive((s) => s.scenario?.population ?? null);
  const world = useWorld();

  const all = useAnswer(`evolution/${runId}`, () => fetchEvolution(runId).then(toEvolution));
  const one = useAnswer(evoStart === null ? null : `evolution/${runId}/${evoStart}`, () => fetchEvolution(runId, evoStart).then(toEvolution));
  const n = all.state === "ready" ? all.value.unit.length : 0;

  // A newly chosen run opens at its last generation.
  const opened = useRef<string | null>(null);
  useEffect(() => {
    if (n > 0 && opened.current !== runId) {
      opened.current = runId;
      set({ generation: n, evoStart: null });
    }
  }, [n, runId, set]);
  const g = Math.max(1, Math.min(n || 1, generation));

  const starts = useAnswer<GenerationStarts>(
    n > 0 ? `starts/${runId}/${g}` : null,
    () => fetchGenerationStarts(runId, g),
    (e) => `No per-start results for generation ${g} (${e.message}). Runs trained before task 7.7 do not keep them.`,
  );
  const ancestry = useAnswer<Ancestry>(
    n > 0 ? `ancestry/${runId}/${g}` : null,
    () => fetchAncestry(runId, g),
    (e) => `No lineage for generation ${g} (${e.message}). Runs trained before task 7.8 do not keep it.`,
  );
  const examUnits = watch.exams.map((e) => e.unit);
  const examUnit = atOrBefore(examUnits, g);
  const examAnswer = useAnswer(examUnit === null ? null : `exam/${runId}/${examUnit}`, () =>
    fetchExamCard(runId, examUnit ?? 0).then((a) => ({ unit: a.unit, card: toExamCard(a.card) })),
  );
  const exam: Loadable<{ unit: number; card: ReturnType<typeof toExamCard> }> =
    examUnit !== null
      ? examAnswer
      : watch.exams.length === 0
        ? { state: "loading" }
        : { state: "missing", why: `No champion was examined up to generation ${g}. The first exam is at generation ${Math.min(...examUnits)}.` };

  if (all.state !== "ready") {
    return (
      <div className="dash" data-testid="timeline">
        <section className="dash-head span-12">{head}</section>
        <div className="panel span-12">
          <div className="panel-body">
            <Slot value={all} height={420}>
              {() => null}
            </Slot>
          </div>
        </div>
      </div>
    );
  }

  const startOf = evoStart ?? (raced?.run === runId ? raced.start : 0);
  const racingThis = raced !== null && raced.run === runId && raced.unit === g;
  const race = (start: number) => loadScenario({ policy: "population", population: { run: runId, unit: g, start, ghost_unit: defaultGhost(g) } });
  const checkpoint = atOrBefore(watch.checkpoints, g);

  const model: GaModel = {
    head,
    all: all.value,
    view: evoStart !== null && one.state === "ready" ? one.value : all.value,
    evoStart,
    startChoices: Array.from({ length: all.value.n_starts }, (_, k) => k),
    setEvoStart: (k) => set({ evoStart: k }),
    raced: startOf,
    checkpoints: watch.checkpoints,
    generation: g,
    pick: (u) => set({ generation: u }),
    starts,
    ancestry,
    exam,
    examAction:
      examUnit === null && examUnits.length > 0
        ? { label: `Go to generation ${Math.min(...examUnits)}`, onClick: () => set({ generation: Math.min(...examUnits) }) }
        : undefined,
    car: racingThis && raced.start === startOf ? (world?.selected ?? null) : null,
    // The cell's start is raced with that car selected (production's Timeline does the same).
    onCell: (row, start) => {
      if (racingThis && raced.start === start) return selectCar(row);
      race(start);
      send({ type: "select_car", index: row }); // after the load: the server answers in order
      set({ evoStart: start });
    },
    onRace: () => {
      race(startOf);
      toEnvironment();
    },
    onReplay:
      checkpoint === null
        ? null
        : () => {
            loadScenario({ policy: "checkpoint", checkpoint: { run: runId, unit: checkpoint } });
            toEnvironment();
          },
  };
  return <GaDashboard model={model} />;
}

// ── Live: any other driving run ─────────────────────────────────────────────────────────

/** Which objective a metrics column is (the badge on its chart). */
function objectiveOf(metric: string): "loss" | "fitness" | undefined {
  if (metric.includes("loss")) return "loss";
  if (metric.includes("fitness")) return "fitness";
  return undefined;
}

/**
 * A run of another method (random search, imitation, DAgger): its own score over the units, its
 * checkpoints, and the Driving Exam of the chosen one, which can be replayed on the track.
 */
function LiveRunTimeline({ runId, head, watch }: { runId: string; head: React.ReactNode; watch: RunsWatchState }) {
  const [metric, setMetric] = useState<string | null>(null);
  const [chosen, setChosen] = useState<number | null>(null);
  const loadScenario = useLive((s) => s.loadScenario);
  useEffect(() => {
    setChosen(null);
    setMetric(null);
  }, [runId]);
  const keys = plottableKeys(watch.records);
  const shown = metric !== null && keys.includes(metric) ? metric : defaultMetric(keys);
  const drawn = useMemo(() => (shown === null ? { unit: [], value: [] } : series(watch.records, shown)), [watch.records, shown]);
  const units = watch.checkpoints;
  const unit = chosen !== null && units.includes(chosen) ? chosen : (units[units.length - 1] ?? null);
  const examined = unit !== null && watch.exams.some((e) => e.unit === unit);
  const exam = useAnswer(examined ? `exam/${runId}/${unit}` : null, () => fetchExamCard(runId, unit ?? 0).then((a) => ({ unit: a.unit, card: toExamCard(a.card) })));
  const examValue: Loadable<{ unit: number; card: ReturnType<typeof toExamCard> }> =
    unit === null ? { state: "missing", why: "This run has no checkpoint yet." } : !examined ? { state: "missing", why: `Checkpoint ${unit} was not examined.` } : exam;
  const pickNear = (u: number) => {
    if (units.length === 0) return;
    setChosen(units.reduce((best, v) => (Math.abs(v - u) < Math.abs(best - u) ? v : best), units[0]));
  };

  return (
    <div className="dash" data-testid="timeline">
      <section className="dash-head span-12">{head}</section>
      <section className="dash-bar span-12" data-testid="checkpoint-bar">
        <div className="dash-gen">
          <span className="eyebrow">Checkpoint</span>
          <span className="dash-gen-value num">{unit ?? "–"}</span>
          <input
            type="range"
            min={0}
            max={Math.max(0, units.length - 1)}
            value={unit === null ? 0 : units.indexOf(unit)}
            onChange={(e) => setChosen(units[Number(e.target.value)])}
            disabled={units.length === 0}
            aria-label="Checkpoint"
            data-testid="timeline-slider"
          />
          <span className="muted">
            {units.length} checkpoints · {watch.exams.length} examined
          </span>
        </div>
        <div className="dash-actions">
          <Button
            variant="primary"
            icon="play"
            disabled={unit === null}
            onClick={() => {
              if (unit === null) return;
              loadScenario({ policy: "checkpoint", checkpoint: { run: runId, unit } });
              toEnvironment();
            }}
            title="Drive this checkpoint from the Driving Exam's 10 starts"
            testId="timeline-replay"
          >
            Replay checkpoint {unit ?? ""}
          </Button>
        </div>
      </section>
      <div className="panel span-7">
        <div className="panel-body">
          <Figure
            title={shown === null ? "Metrics" : `${shown.replace(/_/g, " ")} after each unit`}
            note="Rings: saved checkpoints. Click to choose the nearest one."
            objective={shown === null ? undefined : objectiveOf(shown)}
            actions={
              keys.length > 1 && shown !== null ? <Select value={shown} options={keys.map((k) => ({ value: k, label: k.replace(/_/g, " ") }))} onChange={setMetric} label="Metric" /> : undefined
            }
            testId="metric-figure"
          >
            {watch.records.length === 0 ? (
              <div className="skeleton" style={{ height: 260 }} />
            ) : (
              <MetricChart unit={drawn.unit} value={drawn.value} checkpoints={units} chosen={unit} onPick={pickNear} label={shown ?? ""} />
            )}
          </Figure>
        </div>
      </div>
      <div className="panel span-5">
        <div className="panel-body">
          <Figure title={unit === null ? "Driving Exam" : `Driving Exam · checkpoint ${unit}`} objective="exam" testId="exam-figure">
            <Slot value={examValue} height={260}>
              {(e) => <ExamPanel card={e.card} subject={`Checkpoint ${e.unit}`} />}
            </Slot>
          </Figure>
        </div>
      </div>
    </div>
  );
}

// ── Live: choosing the run ──────────────────────────────────────────────────────────────

function LiveTimeline() {
  const { runs, error } = useRunList();
  const picked = useApp((s) => s.runId);
  const set = useApp((s) => s.set);
  const driving = useMemo(() => (runs === null ? null : drivingRuns(runs)), [runs]);
  const runId = driving === null ? null : picked !== null && driving.some((r) => r.run_id === picked) ? picked : ((driving.find((r) => r.method === "ga") ?? driving[0])?.run_id ?? null);
  const { watch } = useRunsSocket(runId);
  const detail = useAnswer(runId === null ? null : `detail/${runId}`, () => fetchRunDetail(runId ?? ""));

  if (error !== null)
    return (
      <div className="empty" style={{ minHeight: 300 }}>
        <span className="empty-title">The run list could not be loaded</span>
        <span>{error}</span>
      </div>
    );
  if (driving === null) return <SourceWaiting what="the run folders" height={400} />;
  if (runId === null)
    return (
      <div className="empty" style={{ minHeight: 300 }} data-testid="timeline-empty">
        <span className="empty-title">No driving run yet</span>
        <span>Train one in the Training tool, or with uv run python -m neurodrive train --config configs/ga.yaml.</span>
      </div>
    );
  const summary = driving.find((r) => r.run_id === runId) as RunSummary;
  const status = watch.runs[runId]?.status ?? summary.status;
  const head = (
    <RunHead
      runs={driving}
      runId={runId}
      onPick={(id) => set({ runId: id })}
      method={summary.method}
      status={status}
      settings={detail.state === "ready" ? configText(summary.method, detail.value.config) : `${summary.checkpoints} checkpoints`}
    />
  );
  return summary.method === "ga" ? <LiveGaTimeline runId={runId} head={head} watch={watch} /> : <LiveRunTimeline runId={runId} head={head} watch={watch} />;
}

/** The Timeline tool: one run read top to bottom, from the recording or from the server's run folders. */
export function RunTimeline() {
  const source = useApp((s) => s.source);
  return source === "live" ? <LiveTimeline /> : <RecordedTimeline />;
}
