import type {
  CheckpointMessage,
  ExamDoneMessage,
  RunMetricsMessage,
  RunStatusMessage,
} from "../api/generated";

/**
 * The Training tab's picture of the runs (task 5.8, ADR-041), built only from `/ws/runs` messages:
 * every run the server has reported, and the history of the one run being watched. Each message is
 * applied by `applyRunsMessage`, a pure function, so the page and the tests share one set of rules.
 * Nothing here is computed about the runs; the records are the run's metrics lines as stored.
 */
export interface RunsWatchState {
  /** The latest status of every run reported so far, by run id. */
  runs: Record<string, RunStatusMessage>;
  /** The run whose history is shown (sent to the server as `watch`). */
  watched: string | null;
  /** The watched run's metrics lines, in order (one per unit). */
  records: Record<string, unknown>[];
  checkpoints: number[];
  /** The watched run's exam cards (headline numbers), by unit. */
  exams: ExamDoneMessage[];
}

export const initialRunsWatch: RunsWatchState = { runs: {}, watched: null, records: [], checkpoints: [], exams: [] };

/** Start watching `runId` (or nothing): the old history goes; the server sends the new one. */
export function watchRun(state: RunsWatchState, runId: string | null): RunsWatchState {
  return { ...state, watched: runId, records: [], checkpoints: [], exams: [] };
}

type RunsMessage = RunStatusMessage | RunMetricsMessage | CheckpointMessage | ExamDoneMessage;

function isRunsMessage(m: unknown): m is RunsMessage {
  if (typeof m !== "object" || m === null) return false;
  const t = (m as { type?: unknown }).type;
  return t === "run_status" || t === "metrics" || t === "checkpoint" || t === "exam_done";
}

/**
 * Apply one message. A status is kept for any run; the other three only for the watched run.
 * `metrics` with `replace` is the run's whole history again (after `watch`, or when a resumed run
 * was cut back), so the exam rows start again too: they follow as fresh `exam_done` messages.
 */
export function applyRunsMessage(state: RunsWatchState, message: unknown): RunsWatchState {
  if (!isRunsMessage(message)) return state;
  if (message.type === "run_status") return { ...state, runs: { ...state.runs, [message.run_id]: message } };
  if (message.run_id !== state.watched) return state;
  switch (message.type) {
    case "metrics":
      return message.replace
        ? { ...state, records: message.records, exams: [] }
        : { ...state, records: [...state.records, ...message.records] };
    case "checkpoint":
      return { ...state, checkpoints: message.units };
    case "exam_done": {
      const exams = state.exams.filter((e) => e.unit !== message.unit);
      return { ...state, exams: [...exams, message].sort((a, b) => a.unit - b.unit) };
    }
  }
}

/** Runs newest first (run ids start with their start time). */
export function runsNewestFirst(state: RunsWatchState): RunStatusMessage[] {
  return Object.values(state.runs).sort((a, b) => (a.run_id < b.run_id ? 1 : a.run_id > b.run_id ? -1 : 0));
}

/** What the run list says about a run's state, in words. */
export function statusLabel(run: RunStatusMessage): string {
  if (run.status === "running") {
    if (run.exit_code !== null) return `killed (exit code ${run.exit_code}); nd resume can continue it`;
    return run.stop_requested ? "stopping after this unit…" : "running";
  }
  return run.stop_reason === "stopped" ? "stopped" : run.status;
}

/** The numbers the metrics lines contain that can be drawn (number-valued in some line), in first-seen order. */
export function plottableKeys(records: readonly Record<string, unknown>[]): string[] {
  const keys: string[] = [];
  for (const r of records)
    for (const [k, v] of Object.entries(r))
      if (k !== "unit" && typeof v === "number" && !keys.includes(k)) keys.push(k);
  return keys;
}

/** The metric shown first: random search's best fitness, a toy run's loss, else the first one. */
export function defaultMetric(keys: readonly string[]): string | null {
  for (const preferred of ["best_fitness", "loss"]) if (keys.includes(preferred)) return preferred;
  return keys[0] ?? null;
}

/** One metric against the unit, skipping lines where it is not a finite number. */
export function series(records: readonly Record<string, unknown>[], key: string): { unit: number[]; value: number[] } {
  const unit: number[] = [];
  const value: number[] = [];
  for (const r of records) {
    const u = r.unit;
    const v = r[key];
    if (typeof u === "number" && typeof v === "number" && Number.isFinite(v)) {
      unit.push(u);
      value.push(v);
    }
  }
  return { unit, value };
}
