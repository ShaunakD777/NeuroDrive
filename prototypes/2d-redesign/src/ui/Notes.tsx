import { useEffect, useState } from "react";

import { useApp, type ExperimentTab, type ViewId } from "../state/store";

// Design notes: the annotated-mockup layer. With "Notes" on, numbered markers sit on the parts of
// the running prototype they describe, and the drawer lists what each one changes and how hard
// it is to bring into production (1 = current architecture, 2 = frontend refactoring, 3 = future
// enhancement; PRODUCTION_MAPPING.md has the detail). This file is prototype-only.

export interface Note {
  /** CSS selector of the element the marker sits on. */
  target: string;
  /** Where on that element, as fractions of its width and height (default: its top-left corner). */
  at?: [number, number];
  title: string;
  text: string;
  category: 1 | 2 | 3;
}

const ENVIRONMENT: Note[] = [
  {
    target: "[data-testid=stage]",
    at: [0.6, 0.34],
    title: "A track that looks like a track",
    text: "Asphalt with grain and a rubbered line, kerbs on the inside and exit of bends, sand outside tight ones, a mown verge and striped grass. All of it is worked out in the browser from the three lines the server already sends. Decoration stays outside the white edge lines, which are exactly where rays stop and cars crash.",
    category: 1,
  },
  {
    target: "[data-testid=stage]",
    at: [0.36, 0.52],
    title: "Cars that read at every zoom",
    text: "A race-car outline with wheels, wings and a shadow when close; a dart when the whole track is shown. Body colour is the GA's rank band, a gold ring is an elite, a dashed see-through car is the ghost, a faded car with a badge has stopped. The selected car has a reticle and a name tag and is always drawn on top.",
    category: 1,
  },
  {
    target: "[data-testid=btn-play]",
    title: "Controls on the stage, grouped by use",
    text: "Play, step, reset and speed at the top left; rays, trail, camera and zoom at the top right. The scenario (track, driver, generation, start, ghost) moves to the header, because it is set once and the transport is used all the time.",
    category: 2,
  },
  {
    target: "[data-testid=minimap]",
    title: "Minimap",
    text: "While the camera follows a car, the whole circuit stays in view with every car as a dot. It replaces switching to Whole track and back just to see where the car is.",
    category: 1,
  },
  {
    target: "[data-testid=telemetry]",
    title: "Telemetry strip instead of a table over the track",
    text: "The old HUD was a twelve-row table over the upper left of the track; in population mode it hid a third of it. The same numbers now sit in one strip under the action: who, speed, steer and throttle, lap, rays, fitness, step. It sheds groups as the stage narrows.",
    category: 2,
  },
  {
    target: "[data-testid=trailstrip]",
    title: "Look back on what you are scrubbing",
    text: "The look-back slider is drawn on the car's steer and throttle history, so a hard brake or a steering reversal can be seen before it is looked at. The data is the trail the server already sends.",
    category: 1,
  },
  {
    target: "[data-testid=crumbs]",
    at: [0, 0.35],
    title: "The shared selection, always visible",
    text: "Run, generation, start, car, neuron, moment: the chain every view reads. Each part jumps to the view where it is chosen. Today the selection is only visible indirectly, by what each view happens to show.",
    category: 1,
  },
  {
    target: "[data-testid=drawer]",
    title: "Drawer: network or standings",
    text: "Beside the compact network, a Standings tab lists every car in road order with its rank band, status, progress and fitness. A row selects that car everywhere.",
    category: 1,
  },
];

const NETWORK: Note[] = [
  {
    target: "[data-testid=network-diagram]",
    at: [0.5, 0.2],
    title: "Curves, not a mesh",
    text: "Weights are S-curves that leave and arrive horizontally, so 400 lines no longer form a moiré. Weak weights recede; with an output selected, the strongest routes back to the sensors stay lit and the rest fade.",
    category: 1,
  },
  {
    target: "[data-testid=network-diagram]",
    at: [0.02, 0.5],
    title: "Inputs and outputs you can read without hovering",
    text: "Each input has its name, value and a small bar; each output has its value, a signed bar and its meaning in words. Bias, weighted sum and activation appear in the tooltip and the read-out line.",
    category: 1,
  },
  {
    target: ".net-modes",
    title: "One switch for what the lines mean",
    text: "Weights (what was learned), Flow (what each connection carries now), Mutation (what changed since the parent), Change (what changed since the last checkpoint). Today the last two are separate checkboxes in banners. Flow needs Python to send weight × value for every layer, not only the output layer.",
    category: 3,
  },
  {
    target: "[data-testid=lineage-banner]",
    title: "Lineage in one sentence",
    text: "Where this network came from, with the numbers that matter: parents, their ranks, genes from each, genes mutated.",
    category: 1,
  },
  {
    target: "[data-testid=output-inspector]",
    title: "The inspector beside the diagram",
    text: "Today the inspector is below the diagram and out of sight until scrolled to. As a rail, a decision and its explanation are read together. The two outputs are cards at the top: the way in.",
    category: 2,
  },
  {
    target: "[data-testid=inspector-why]",
    at: [0.62, 0.72],
    title: "The weighted sum and the activation, drawn",
    text: "Contributions add up to z; the curve shows where z lands on tanh. Near the middle the output follows z; far out it barely moves. Hovering a row lights that neuron in the diagram.",
    category: 1,
  },
  {
    target: "[data-testid=whatif-panel]",
    title: "What-if, with its limits stated",
    text: "Paused only, about the present only. When it cannot be used it says why and offers the one click that fixes it.",
    category: 1,
  },
];

const TIMELINE: Note[] = [
  {
    target: "[data-testid=generation-bar]",
    title: "One generation for the whole page",
    text: "A single cursor, set here or on any chart, drives every panel and stays in reach while the page scrolls. The two actions that leave the page sit beside it.",
    category: 2,
  },
  {
    target: "[data-testid=fitness-figure]",
    at: [0.3, 0.3],
    title: "Lines, the middle half, and checkpoints",
    text: "The band is the first to third quartile the server already computes. Rings mark saved checkpoints. The distribution map below shares the generation axis, so a column in one is the same generation in the other.",
    category: 1,
  },
  {
    target: ".objective",
    title: "Every chart says what it measures",
    text: "Loss, fitness and reward are different objectives. A badge on every chart names the objective and which way is better, so two charts are never read as one scale.",
    category: 1,
  },
  {
    target: "[data-testid=exam-figure]",
    title: "The Driving Exam is set apart",
    text: "The only filled badge. Training track and held-out tracks are two blocks, because the gap between them is the finding: this champion finishes every training start and a quarter of the held-out ones.",
    category: 1,
  },
  {
    target: "[data-testid=outcomes-figure]",
    title: "Outcomes: status colours with words",
    text: "Finished, crashed, stalled, out of time: the same four colours as the chips and the track badges, always with a legend. A cell selects that car in every view.",
    category: 1,
  },
  {
    target: "[data-testid=ancestry-figure]",
    title: "Ancestry in the selection colour",
    text: "The line of descent is the thing being followed, so it wears the selection colour; the generation's best is a quiet dashed reference.",
    category: 1,
  },
];

const COMPARE: Note[] = [
  {
    target: "[data-testid=exam-figure]",
    title: "Where methods meet",
    text: "One row per method, a dot per seed, the median as a tick, the middle half as a bar. Built from exam cards that already exist in run folders; the endpoint that gathers them is roadmap task 8.1.",
    category: 3,
  },
  {
    target: "[data-testid=runlist]",
    title: "Pick a comparison, not 270 checkboxes",
    text: "Runs are grouped by study and method, with the colour key beside the name. Today the list is one long column of every run in the folder.",
    category: 2,
  },
  {
    target: "[data-testid=loss-figure]",
    title: "Median and middle half, labelled at the line",
    text: "Names at the line ends instead of a legend to match by colour; the table repeats every number.",
    category: 1,
  },
];

const TRAINING: Note[] = [
  {
    target: "[data-testid=launch-form]",
    title: "Launch form as a form",
    text: "Labels above fields, one primary button. The run list shows progress as a bar, not only as text.",
    category: 1,
  },
  {
    target: "[data-testid=live-figure]",
    title: "A run that is visibly alive",
    text: "The latest point pulses, the part of the axis not reached yet is shaded, exams appear as rings and as table rows when they finish.",
    category: 1,
  },
];

export function notesFor(view: ViewId, tab: ExperimentTab): Note[] {
  if (view === "environment") return ENVIRONMENT;
  if (view === "network") return NETWORK;
  if (tab === "compare") return COMPARE;
  if (tab === "training") return TRAINING;
  if (tab === "timeline") return TIMELINE;
  return [];
}

const CATEGORY = ["", "Current architecture", "Frontend refactoring", "Future enhancement"];

/** The numbered markers, placed over whatever each note points at. */
export function NoteMarkers() {
  const on = useApp((s) => s.notes);
  const view = useApp((s) => s.view);
  const tab = useApp((s) => s.experimentTab);
  const [spots, setSpots] = useState<({ x: number; y: number } | null)[]>([]);
  const notes = notesFor(view, tab);

  useEffect(() => {
    if (!on) return;
    const place = () => {
      setSpots(
        notes.map((note) => {
          const el = document.querySelector(note.target);
          if (el === null) return null;
          const r = el.getBoundingClientRect();
          if (r.width === 0 || r.bottom < 0 || r.top > window.innerHeight) return null;
          const [fx, fy] = note.at ?? [0, 0];
          return { x: r.left + r.width * fx, y: Math.max(14, r.top + r.height * fy) };
        }),
      );
    };
    place();
    const timer = window.setInterval(place, 250);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [on, view, tab]);

  if (!on) return null;
  return (
    <div className="note-layer" aria-hidden="true">
      {notes.map((note, k) => {
        const spot = spots[k];
        return spot ? (
          <span key={k} className="note-marker" data-category={note.category} style={{ left: spot.x, top: spot.y }}>
            {k + 1}
          </span>
        ) : null;
      })}
    </div>
  );
}

/** The list of notes for the current view, shown in the drawer while notes are on. */
export function NoteList() {
  const view = useApp((s) => s.view);
  const tab = useApp((s) => s.experimentTab);
  const notes = notesFor(view, tab);
  return (
    <div className="note-list" data-testid="note-list">
      {notes.length === 0 && <p className="muted">No notes for this tool.</p>}
      {notes.map((note, k) => (
        <article key={k} className="note">
          <span className="note-marker note-marker-inline" data-category={note.category}>
            {k + 1}
          </span>
          <div>
            <h4>{note.title}</h4>
            <p>{note.text}</p>
            <span className="note-category" data-category={note.category}>
              Category {note.category} · {CATEGORY[note.category]}
            </span>
          </div>
        </article>
      ))}
    </div>
  );
}
