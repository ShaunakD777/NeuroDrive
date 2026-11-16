import { describe, expect, it } from "vitest";

import type { RunStatusMessage } from "../api/generated";
import {
  applyRunsMessage,
  defaultMetric,
  historyFor,
  initialRunsWatch,
  plottableKeys,
  runsNewestFirst,
  series,
  statusLabel,
  watchRun,
} from "./runsWatch";

const A = "20261102-140000-random_search-a";
const B = "20261102-150000-toy_xor-b";

function status(runId: string, extra: Partial<RunStatusMessage> = {}): RunStatusMessage {
  return {
    type: "run_status",
    run_id: runId,
    method: "random_search",
    tag: "a",
    status: "running",
    stop_reason: null,
    units_done: 0,
    budget_units: 30,
    stop_requested: false,
    launched: true,
    exit_code: null,
    ...extra,
  };
}

const metrics = (runId: string, replace: boolean, units: number[]) => ({
  type: "metrics",
  run_id: runId,
  replace,
  records: units.map((u) => ({ unit: u, best_fitness: u / 10, improved: true })),
});

const exam = (runId: string, unit: number) => ({
  type: "exam_done",
  run_id: runId,
  unit,
  training: { completion_rate: 0, crash_rate: 1, progress_frac_median: 0.1 },
  heldout: { completion_rate: 0, crash_rate: 1, progress_frac_median: 0.05 },
});

describe("applyRunsMessage", () => {
  it("keeps the latest status of every run, newest first", () => {
    let s = applyRunsMessage(initialRunsWatch, status(A));
    s = applyRunsMessage(s, status(B));
    s = applyRunsMessage(s, status(A, { units_done: 3 }));
    expect(runsNewestFirst(s).map((r) => r.run_id)).toEqual([B, A]);
    expect(s.runs[A].units_done).toBe(3);
  });

  it("builds the watched run's history: replace, then additions, in order", () => {
    let s = watchRun(initialRunsWatch, A);
    s = applyRunsMessage(s, metrics(A, true, [1, 2]));
    s = applyRunsMessage(s, metrics(A, false, [3]));
    s = applyRunsMessage(s, { type: "checkpoint", run_id: A, units: [0, 1, 2] });
    s = applyRunsMessage(s, exam(A, 2));
    s = applyRunsMessage(s, exam(A, 1));
    expect(s.records.map((r) => r.unit)).toEqual([1, 2, 3]);
    expect(s.checkpoints).toEqual([0, 1, 2]);
    expect(s.exams.map((e) => e.unit)).toEqual([1, 2]);
  });

  it("ignores another run's history and anything that is not a runs message", () => {
    let s = watchRun(initialRunsWatch, A);
    s = applyRunsMessage(s, metrics(B, true, [1]));
    s = applyRunsMessage(s, exam(B, 0));
    s = applyRunsMessage(s, { type: "hello", version: "0.1.0", api: "v1.13" });
    s = applyRunsMessage(s, "nonsense");
    expect(s.records).toEqual([]);
    expect(s.exams).toEqual([]);
  });

  it("starts the history again on replace (a resumed run was cut back), exam rows included", () => {
    let s = watchRun(initialRunsWatch, A);
    s = applyRunsMessage(s, metrics(A, true, [1, 2, 3]));
    s = applyRunsMessage(s, exam(A, 2));
    s = applyRunsMessage(s, metrics(A, true, [1]));
    expect(s.records.map((r) => r.unit)).toEqual([1]);
    expect(s.exams).toEqual([]);
  });

  it("forgets the old history when another run is watched", () => {
    let s = watchRun(initialRunsWatch, A);
    s = applyRunsMessage(s, metrics(A, true, [1]));
    s = watchRun(s, B);
    expect(s.watched).toBe(B);
    expect(s.records).toEqual([]);
  });
});

describe("historyFor (task S0: the stale-history guard of useRunsSocket)", () => {
  // Run A's history: two checkpoints and an exam at unit 10, as the server sent it.
  function watchingA() {
    let s = applyRunsMessage(initialRunsWatch, status(A));
    s = applyRunsMessage(s, status(B, { method: "toy_xor" }));
    s = watchRun(s, A);
    s = applyRunsMessage(s, metrics(A, true, [5, 10]));
    s = applyRunsMessage(s, { type: "checkpoint", run_id: A, units: [5, 10] });
    return applyRunsMessage(s, exam(A, 10));
  }

  it("hands out the stored history when it is the run asked for", () => {
    const s = watchingA();
    expect(historyFor(s, A)).toBe(s);
  });

  it("never hands out run A's units as run B's in the render before the reset", () => {
    const shown = historyFor(watchingA(), B);
    expect(shown.watched).toBe(B);
    expect(shown.records).toEqual([]);
    expect(shown.checkpoints).toEqual([]);
    expect(shown.exams).toEqual([]);
    // every run's status stays: the Training tool's run list does not flicker
    expect(Object.keys(shown.runs).sort()).toEqual([A, B].sort());
  });

  it("also when the caller stops watching, and once the reset has run", () => {
    expect(historyFor(watchingA(), null).checkpoints).toEqual([]);
    const reset = watchRun(watchingA(), B);
    expect(historyFor(reset, B)).toBe(reset);
  });
});

describe("what is drawn", () => {
  const records = [
    { unit: 1, best_fitness: 0.1, improved: true, grad_norm: [1, 2] },
    { unit: 2, best_fitness: 0.3, unit_median_fitness: 0.02, improved: false },
    { unit: 3, best_fitness: null, unit_median_fitness: 0.04 },
  ];

  it("offers the number-valued metrics, best fitness first by default", () => {
    expect(plottableKeys(records)).toEqual(["best_fitness", "unit_median_fitness"]);
    expect(defaultMetric(["unit_median_fitness", "best_fitness"])).toBe("best_fitness");
    expect(defaultMetric(["loss_before", "loss"])).toBe("loss");
    expect(defaultMetric(["x"])).toBe("x");
    expect(defaultMetric([])).toBeNull();
  });

  it("draws a metric against the unit, skipping lines without a number", () => {
    expect(series(records, "best_fitness")).toEqual({ unit: [1, 2], value: [0.1, 0.3] });
    expect(series(records, "unit_median_fitness")).toEqual({ unit: [2, 3], value: [0.02, 0.04] });
  });

  it("describes stopped, stopping and killed runs in words", () => {
    expect(statusLabel(status(A))).toBe("running");
    expect(statusLabel(status(A, { stop_requested: true }))).toBe("stopping after this unit…");
    expect(statusLabel(status(A, { exit_code: 1 }))).toContain("nd resume");
    expect(statusLabel(status(A, { status: "completed", stop_reason: "stopped" }))).toBe("stopped");
    expect(statusLabel(status(A, { status: "completed", stop_reason: "units" }))).toBe("completed");
  });
});
