import { useEffect, useRef, type CSSProperties } from "react";

import { shortName, sweepSeed, type RunGroup } from "./charts";

/** A tick box whose third, "some but not all" state is set on the element (HTML has no attribute). */
function GroupBox({
  checked,
  partial,
  onChange,
  label,
}: {
  checked: boolean;
  partial: boolean;
  onChange: () => void;
  label: string;
}) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = partial;
  }, [partial]);
  return (
    <input
      ref={ref}
      type="checkbox"
      checked={checked}
      onChange={onChange}
      aria-label={label}
    />
  );
}

/**
 * The runs to compare, newest first, grouped by sweep (task 4.7b): a sweep has one tick box for
 * all its seeds and one per seed; a run outside a sweep has its own. A ticked group shows the
 * colour it has in the chart. Only shows and reports clicks; the parent owns the selection.
 */
export function RunList({
  groups,
  selected,
  slots,
  onToggle,
}: {
  groups: RunGroup[];
  selected: ReadonlySet<string>;
  slots: Record<string, number>;
  onToggle: (runIds: string[], on: boolean) => void;
}) {
  return (
    <ul className="runlist" data-testid="runlist">
      {groups.map((g) => {
        const ids = g.runs.map((r) => r.run_id);
        const ticked = ids.filter((id) => selected.has(id)).length;
        const slot = slots[g.groupId];
        const style =
          slot !== undefined
            ? ({ "--series": `var(--series-${slot + 1})` } as CSSProperties)
            : undefined;
        const name = shortName(g.groupId);
        return (
          <li
            key={g.groupId}
            className="runlist-group"
            data-testid="runlist-group"
            data-group-id={g.groupId}
            style={style}
          >
            <label className="runlist-row">
              <GroupBox
                checked={ticked === ids.length}
                partial={ticked > 0 && ticked < ids.length}
                onChange={() => onToggle(ids, ticked < ids.length)}
                label={g.sweep ? `sweep ${name}` : `run ${name}`}
              />
              <span
                className={
                  slot !== undefined && ticked > 0
                    ? "cmp-swatch"
                    : "cmp-swatch cmp-swatch-off"
                }
              />
              <span className="runlist-name" title={g.groupId}>
                {name}
              </span>
              <span className="runlist-meta">
                {g.sweep
                  ? `${ids.length} seed${ids.length === 1 ? "" : "s"}`
                  : g.runs[0].status}
              </span>
            </label>
            {g.sweep && (
              <ul className="runlist-seeds">
                {g.runs.map((r) => (
                  <li key={r.run_id}>
                    <label className="runlist-row">
                      <input
                        type="checkbox"
                        checked={selected.has(r.run_id)}
                        onChange={() =>
                          onToggle([r.run_id], !selected.has(r.run_id))
                        }
                        data-testid="runlist-run"
                        data-run-id={r.run_id}
                      />
                      <span className="runlist-name">
                        seed {sweepSeed(r) ?? "?"}
                      </span>
                      {r.status !== "completed" && (
                        <span className="runlist-meta">{r.status}</span>
                      )}
                    </label>
                  </li>
                ))}
              </ul>
            )}
          </li>
        );
      })}
    </ul>
  );
}
