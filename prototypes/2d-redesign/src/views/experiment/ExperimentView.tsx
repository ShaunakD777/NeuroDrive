import { useApp, type ExperimentTab } from "../../state/store";
import { Segmented } from "../../ui/controls";
import { Compare } from "./Compare";
import { RunTimeline } from "./Timeline";
import { Training } from "./Training";

const SPEC_ONLY: Record<"toylab" | "datasets", { title: string; body: string; parts: string[] }> = {
  toylab: {
    title: "Toy lab",
    body: "One toy run (line, sine, XOR, two spirals): its loss curve, what the network predicts at a chosen checkpoint, and which weights the gradient is pulling on.",
    parts: [
      "Loss curve: the line chart of the Timeline tool, with the Loss badge and the checkpoint cursor.",
      "Prediction plot: target points as ringed dots, the network's curve as a 2 px line; XOR and spirals as a diverging map (blue, orange, neutral middle).",
      "Gradient glow: the network diagram of the Network view, lines shaded by |dL/dW| instead of by weight.",
    ],
  },
  datasets: {
    title: "Datasets",
    body: "What a demonstration dataset holds: label and observation histograms with their dense coverage, the frames on the track map, the episodes, and a second dataset beside the first.",
    parts: [
      "Histograms: the bar mark of the Timeline's histogram (capped width, rounded data end), two datasets as paired bars in series 1 and 2.",
      "Track map: the Environment view's track drawing at whole-track scale, frames as dots on the diverging pair (steer) or the one-hue ramp (throttle).",
      "Episodes: the shared table style, status as chips.",
    ],
  },
};

/** A tool that is specified but not built in the prototype: says so, and how it follows the system. */
function SpecOnly({ tab }: { tab: "toylab" | "datasets" }) {
  const spec = SPEC_ONLY[tab];
  return (
    <div className="dash">
      <div className="panel span-12">
        <div className="empty spec-only" data-testid={`spec-${tab}`}>
          <span className="eyebrow">Specified, not prototyped</span>
          <span className="empty-title">{spec.title}</span>
          <span className="spec-body">{spec.body}</span>
          <ul>
            {spec.parts.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
          <span className="muted">The existing tool keeps all its functions. DESIGN_SPEC.md §8 and PRODUCTION_MAPPING.md §3.3 say how it takes on the new chart styles.</span>
        </div>
      </div>
    </div>
  );
}

/**
 * The Experiment view: five tools, as in production (Toy lab, Compare runs, Training, Timeline,
 * Datasets). The prototype builds the three that carry the redesign's ideas.
 */
export function ExperimentView() {
  const tab = useApp((s) => s.experimentTab);
  const set = useApp((s) => s.set);
  return (
    <section className="view exp-view" data-testid="experiment-view">
      <header className="view-head">
        <div className="view-title">
          <h2>Experiment</h2>
          <span className="view-question">How did it learn to?</span>
        </div>
        <Segmented<ExperimentTab>
          label="Experiment tool"
          value={tab}
          onChange={(id) => set({ experimentTab: id })}
          options={[
            { id: "timeline", testId: "experiment-tab-timeline", label: "Timeline", icon: "chart", hint: "One run: its checkpoints, generations and exams" },
            { id: "compare", testId: "experiment-tab-compare", label: "Compare runs", icon: "compare", hint: "Several runs or sweeps on one objective, and every method on the exam" },
            { id: "training", testId: "experiment-tab-training", label: "Training", icon: "rocket", hint: "Start, watch and stop training runs" },
            { id: "toylab", testId: "experiment-tab-toylab", label: "Toy lab", icon: "flask", hint: "Backpropagation on toy tasks" },
            { id: "datasets", testId: "experiment-tab-datasets", label: "Datasets", icon: "data", hint: "Demonstration datasets" },
          ]}
        />
      </header>
      <div className="view-body exp-body">
        {tab === "timeline" && <RunTimeline />}
        {tab === "compare" && <Compare />}
        {tab === "training" && <Training />}
        {(tab === "toylab" || tab === "datasets") && <SpecOnly tab={tab} />}
      </div>
    </section>
  );
}
