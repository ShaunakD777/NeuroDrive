// The Experiment view's data: run folders, their charts and exams. With the live source every
// answer comes from the Python server, through the production app's REST helpers and its
// /ws/runs socket (web/src/api/rest.ts, ws.ts, web/src/state/runsWatch.ts). Nothing here
// computes a statistic: medians, quartiles, histograms and diversity are Python's.

import { useEffect, useRef, useState } from "react";

import type { Ancestry, EvolutionView, GenerationStarts, RunDetail, RunSummary } from "@prod/api/generated";
import { fetchRuns } from "@prod/api/rest";
import { connectLive, type ConnectionStatus, type LiveConnection } from "@prod/api/ws";
import { applyRunsMessage, initialRunsWatch, type RunsWatchState, watchRun } from "@prod/state/runsWatch";

import type { Evolution, ExamCard } from "../mock/types";

// ── Loading one answer ────────────────────────────────────────────────────────────────────

/** A value that is there, still on its way, or not available (with the reason, in words). */
export type Loadable<T> = { state: "ready"; value: T } | { state: "loading" } | { state: "missing"; why: string };

const cache = new Map<string, Promise<unknown>>();

/**
 * Fetch once per `key` (null: nothing to fetch) and remember the answer for the page's life, so
 * moving the generation cursor back and forth asks the server only once per generation.
 */
export function useAnswer<T>(key: string | null, fetcher: () => Promise<T>, explain: (error: Error) => string = (e) => e.message): Loadable<T> {
  const [found, setFound] = useState<{ key: string; value: Loadable<T> } | null>(null);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  useEffect(() => {
    if (key === null) return;
    let live = true;
    let promise = cache.get(key) as Promise<T> | undefined;
    if (promise === undefined) {
      promise = fetcherRef.current();
      cache.set(key, promise);
      promise.catch(() => cache.delete(key)); // a refusal may be temporary (a run still training)
    }
    promise.then(
      (value) => live && setFound({ key, value: { state: "ready", value } }),
      (error: Error) => live && setFound({ key, value: { state: "missing", why: explain(error) } }),
    );
    return () => {
      live = false;
    };
    // `explain` is a formatting choice, not an input
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  if (key === null) return { state: "missing", why: "nothing chosen" };
  return found !== null && found.key === key ? found.value : { state: "loading" };
}

// ── Runs ─────────────────────────────────────────────────────────────────────────────────

/** Runs a driver can replay (not toy runs, at least one checkpoint), as production's Timeline lists them. */
export function drivingRuns(runs: readonly RunSummary[]): RunSummary[] {
  return runs.filter((r) => !r.method.startsWith("toy_") && r.checkpoints > 0);
}

let listCache: { at: number; promise: Promise<RunSummary[]> } | null = null;
const LIST_FRESH_MS = 30_000;

/** The run list, asked for at most every 30 s: `GET runs` reads every run folder's metadata. */
function runList(force: boolean): Promise<RunSummary[]> {
  if (force || listCache === null || performance.now() - listCache.at > LIST_FRESH_MS) {
    const promise = fetchRuns().then((list) => list.runs);
    promise.catch(() => (listCache = null));
    listCache = { at: performance.now(), promise };
  }
  return listCache.promise;
}

/** Every run folder the server sees, newest first (`GET runs`); `reload` asks again. */
export function useRunList(): { runs: RunSummary[] | null; error: string | null; reload: () => void } {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    runList(tick > 0)
      .then((list) => live && setRuns(list))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [tick]);
  return { runs, error, reload: () => setTick((t) => t + 1) };
}

/**
 * The /ws/runs socket (task 5.8): every run's status, and the history of the watched run (its
 * metrics lines, checkpoints and exam headlines). One socket per tool on screen, as production.
 */
export function useRunsSocket(watched: string | null): { watch: RunsWatchState; status: ConnectionStatus } {
  const [watch, setWatch] = useState<RunsWatchState>(initialRunsWatch);
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const socket = useRef<LiveConnection | null>(null);
  const current = useRef<string | null>(watched);
  current.current = watched;
  useEffect(() => {
    const connection = connectLive(
      (s) => {
        setStatus(s);
        if (s === "connected" && current.current !== null) socket.current?.send(JSON.stringify({ type: "watch", run_id: current.current }));
      },
      (message) => setWatch((state) => applyRunsMessage(state, message)),
      "/ws/runs",
    );
    socket.current = connection;
    return () => {
      socket.current = null;
      connection.stop();
    };
  }, []);
  useEffect(() => {
    setWatch((state) => watchRun(state, watched));
    socket.current?.send(JSON.stringify({ type: "watch", run_id: watched }));
  }, [watched]);
  // Until the reset above has run, `watch` still holds the previous run's history: never hand that
  // out as this run's (its checkpoints and exams would be fetched from the wrong run folder).
  const mine = watch.watched === watched ? watch : { ...initialRunsWatch, runs: watch.runs, watched };
  return { watch: mine, status };
}

// ── Shapes ───────────────────────────────────────────────────────────────────────────────

/**
 * The server's evolution charts in the shape the chart components draw. Python sends null where a
 * number does not exist (diversity of a one-network generation, finishers of a run trained before
 * task 7.7); the charts draw NaN as a gap, so null becomes NaN. No value is computed.
 */
export function toEvolution(view: EvolutionView): Evolution {
  const gap = (values: (number | null)[]) => values.map((v) => (v === null ? Number.NaN : v));
  return {
    ...view,
    diversity: gap(view.diversity),
    finished_share: gap(view.finished_share),
    finishers: gap(view.finishers),
    n_starts: view.n_starts ?? 0,
    start_finish_share: view.start_finish_share ?? [],
    start_mean: view.start_mean ?? [],
  };
}

export type { Ancestry, GenerationStarts, RunDetail, RunSummary };

/** A stored exam card (`GET runs/{id}/exams/{unit}`) in the shape the exam panel draws. */
export function toExamCard(card: { [k: string]: unknown }): ExamCard {
  return card as unknown as ExamCard;
}

/** A value deep inside a run's config, or undefined (configs differ by method). */
export function configValue(config: { [k: string]: unknown }, path: string): unknown {
  let at: unknown = config;
  for (const part of path.split(".")) {
    if (typeof at !== "object" || at === null) return undefined;
    at = (at as Record<string, unknown>)[part];
  }
  return at;
}

/** The settings that tell runs of one method apart, in one line. */
export function configText(method: string, config: { [k: string]: unknown }): string {
  const v = (path: string) => configValue(config, path);
  const sizes = v("network.layer_sizes");
  const net = Array.isArray(sizes) ? sizes.join(" → ") : null;
  const parts: string[] = [];
  if (typeof v("seed") === "number") parts.push(`seed ${v("seed")}`);
  if (net) parts.push(net);
  if (method === "ga") {
    if (v("population.size") !== undefined) parts.push(`${v("population.size")} networks, ${v("population.elite")} elites`);
    if (v("selection.tournament_size") !== undefined) parts.push(`tournament of ${v("selection.tournament_size")}`);
    if (v("variation.crossover_rate") !== undefined) parts.push(`crossover ${v("variation.crossover_rate")}`);
    if (v("variation.mutation_rate") !== undefined) parts.push(`mutation ${v("variation.mutation_rate")} × σ ${v("variation.mutation_strength")}`);
    if (typeof v("fitness.formula") === "string") parts.push(`fitness ${v("fitness.formula")}`);
  } else if (method === "imitation" || method === "dagger") {
    if (v("data.dataset") !== undefined) parts.push(`data ${v("data.dataset")}`);
    if (v("optim.name") !== undefined) parts.push(`${v("optim.name")} lr ${v("optim.lr")}`);
  }
  return parts.join(" · ");
}

export const METHOD_LABEL: Record<string, string> = {
  ga: "genetic algorithm",
  random_search: "random search",
  imitation: "imitation (cloning)",
  dagger: "DAgger",
  baseline: "baseline",
};

export function methodLabel(method: string): string {
  if (method.startsWith("toy_")) return `toy · ${method.slice(4)}`;
  return METHOD_LABEL[method] ?? method.replace(/_/g, " ");
}
