import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";

import type { CompareResponse, RunSummary } from "../../api/generated";
import { fetchCompare, fetchRuns } from "../../api/rest";
import { CompareChart } from "./CompareChart";
import {
  MAX_GROUPS,
  colorSlots,
  formatNumber,
  groupRuns,
  shortName,
} from "./charts";
import { RunList } from "./RunList";

type Metric = "loss" | "val_loss";
const METRICS: { id: Metric; label: string }[] = [
  { id: "loss", label: "training loss" },
  { id: "val_loss", label: "validation loss" },
];

/**
 * "Compare runs" (task 4.7b, ADR-033): tick runs or whole sweeps, and see their learning curves
 * overlaid as median lines inside IQR bands, with a table of final values. The server groups the
 * runs and computes every statistic (`GET /api/runs/compare`); this page fetches and draws.
 */
export function CompareRuns({ headerExtra }: { headerExtra?: ReactNode }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [metric, setMetric] = useState<Metric>("loss");
  const [data, setData] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const slotMemory = useRef<Record<string, number>>({});

  const groups = useMemo(() => groupRuns(runs ?? []), [runs]);
  const groupOf = useMemo(() => {
    const map = new Map<string, string>();
    for (const g of groups)
      for (const r of g.runs) map.set(r.run_id, g.groupId);
    return map;
  }, [groups]);

  function loadRuns() {
    setError(null);
    fetchRuns()
      .then((list) => setRuns(list.runs))
      .catch((e: Error) => setError(e.message));
  }
  useEffect(loadRuns, []);

  // The groups with at least one ticked run, in list order, and their colour slots.
  const activeGroups = groups
    .filter((g) => g.runs.some((r) => selected.has(r.run_id)))
    .map((g) => g.groupId);
  const slots = colorSlots(activeGroups, slotMemory.current) ?? {};
  useEffect(() => {
    slotMemory.current = { ...slotMemory.current, ...slots };
  });

  function toggle(runIds: string[], on: boolean) {
    setNotice(null);
    const next = new Set(selected);
    for (const id of runIds) {
      if (on) next.add(id);
      else next.delete(id);
    }
    const groupCount = new Set([...next].map((id) => groupOf.get(id) ?? id))
      .size;
    if (groupCount > MAX_GROUPS) {
      setNotice(
        `At most ${MAX_GROUPS} groups can be compared at once (one colour each). Untick one first.`,
      );
      return;
    }
    setSelected(next);
  }

  // Ask the server whenever the ticked runs or the metric change; ignore answers that arrive late.
  useEffect(() => {
    const ids = [...selected];
    if (ids.length === 0) {
      setData(null);
      return;
    }
    let live = true;
    fetchCompare(ids, metric)
      .then((answer) => {
        if (!live) return;
        setData(answer);
        setError(null);
      })
      .catch((e: Error) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, [selected, metric]);

  const metricLabel = METRICS.find((m) => m.id === metric)!.label;
  const show = (v: number | null) =>
    v === null ? "∞ (diverged)" : formatNumber(v);

  return (
    <section className="view toylab" data-testid="compare">
      <div className="view-header">
        <h2>Experiment: compare runs</h2>
        {headerExtra}
      </div>

      <div className="toylab-controls">
        <div className="segmented" role="group" aria-label="Metric">
          {METRICS.map((m) => (
            <button
              key={m.id}
              type="button"
              aria-pressed={m.id === metric}
              onClick={() => setMetric(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
        <button type="button" onClick={loadRuns} data-testid="compare-refresh">
          Refresh list
        </button>
        {selected.size > 0 && (
          <button type="button" onClick={() => setSelected(new Set())}>
            Clear
          </button>
        )}
      </div>

      {error && <p className="toylab-error">Could not load: {error}</p>}
      {notice && (
        <p className="hint" data-testid="compare-notice">
          {notice}
        </p>
      )}
      {runs !== null && runs.length === 0 && (
        <p className="hint">
          No runs yet. Make a sweep with{" "}
          <code>
            uv run python -m neurodrive sweep --config configs/toy_sine.yaml
            --seeds 0-4
          </code>{" "}
          and press Refresh list.
        </p>
      )}

      <div className="compare-layout">
        <RunList
          groups={groups}
          selected={selected}
          slots={slots}
          onToggle={toggle}
        />
        <div className="compare-main">
          {data === null ? (
            <p className="hint">
              Tick runs or a whole sweep on the left to overlay their curves.
            </p>
          ) : (
            <>
              {data.groups.length > 0 && (
                <CompareChart
                  data={data}
                  slots={slots}
                  metricLabel={metricLabel}
                />
              )}
              {data.skipped.length > 0 && (
                <p className="hint" data-testid="compare-skipped">
                  {data.skipped.length} ticked run
                  {data.skipped.length === 1 ? " has" : "s have"} no{" "}
                  {metricLabel} (no validation set):{" "}
                  {data.skipped.map(shortName).join(", ")}.
                </p>
              )}
              {data.groups.length > 0 && (
                <table className="compare-table" data-testid="compare-table">
                  <caption>
                    Final {metricLabel} per group (a diverged run counts as ∞)
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Group</th>
                      <th scope="col">Runs</th>
                      <th scope="col">Median</th>
                      <th scope="col">IQR (Q1 to Q3)</th>
                      <th scope="col">Diverged</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.groups.map((g) => (
                      <tr
                        key={g.group_id}
                        data-testid="compare-row"
                        data-group-id={g.group_id}
                        style={
                          {
                            "--series": `var(--series-${(slots[g.group_id] ?? 0) + 1})`,
                          } as CSSProperties
                        }
                      >
                        <td>
                          <span className="cmp-swatch" />{" "}
                          {shortName(g.group_id)}
                        </td>
                        <td>{g.run_ids.length}</td>
                        <td data-testid="compare-final-median">
                          {show(g.final_median)}
                        </td>
                        <td>
                          {show(g.final_q1)} to {show(g.final_q3)}
                        </td>
                        <td>{g.n_diverged}</td>
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
