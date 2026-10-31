import { useState, type ReactNode } from "react";

import { CompareRuns } from "./CompareRuns";
import { ToyLab } from "./ToyLab";

type Tab = "toylab" | "compare";
const TABS: { id: Tab; label: string }[] = [
  { id: "toylab", label: "Toy lab" },
  { id: "compare", label: "Compare runs" },
];

/**
 * The Experiment page (ADR-033): two tabs, the toy lab (one run, its checkpoints, predictions and
 * gradient glow) and "Compare runs" (several runs or sweeps, overlaid median and IQR curves). The
 * tab switch sits in the view header, next to whatever the layout puts there.
 */
export function ExperimentView({ headerExtra }: { headerExtra?: ReactNode }) {
  const [tab, setTab] = useState<Tab>("toylab");
  const tabs = (
    <div className="segmented" role="group" aria-label="Experiment tool">
      {TABS.map((t) => (
        <button
          key={t.id}
          type="button"
          aria-pressed={t.id === tab}
          onClick={() => setTab(t.id)}
          data-testid={`experiment-tab-${t.id}`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
  const extra = (
    <>
      {tabs}
      {headerExtra}
    </>
  );
  return tab === "toylab" ? (
    <ToyLab headerExtra={extra} />
  ) : (
    <CompareRuns headerExtra={extra} />
  );
}
