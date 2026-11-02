import { useState, type ReactNode } from "react";

import { CompareRuns } from "./CompareRuns";
import { Timeline } from "./Timeline";
import { ToyLab } from "./ToyLab";
import { TrainingRuns } from "./TrainingRuns";

type Tab = "toylab" | "compare" | "training" | "timeline";
const TABS: { id: Tab; label: string }[] = [
  { id: "toylab", label: "Toy lab" },
  { id: "compare", label: "Compare runs" },
  { id: "training", label: "Training" },
  { id: "timeline", label: "Timeline" },
];

/**
 * The Experiment page (ADR-033, ADR-041, ADR-042): four tabs, the toy lab (one run, its
 * checkpoints, predictions and gradient glow), "Compare runs" (several runs or sweeps, overlaid
 * median and IQR curves), "Training" (start runs, watch them train, stop them) and "Timeline"
 * (replay a driving run's checkpoints in every view). The tab switch sits in the view header,
 * next to whatever the layout puts there.
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
  if (tab === "toylab") return <ToyLab headerExtra={extra} />;
  if (tab === "compare") return <CompareRuns headerExtra={extra} />;
  if (tab === "training") return <TrainingRuns headerExtra={extra} />;
  return <Timeline headerExtra={extra} />;
}
