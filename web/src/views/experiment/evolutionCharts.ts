// Pure helpers for the evolution charts (task 7.7, ADR-058). No maths about the population here:
// the statistics, bands and counts come from the Python server (experiment/evolution_view.py);
// this only maps them to shades, labels and positions.

import type { EvolutionView } from "../../api/generated";

/**
 * Upper limits of the heat map's shades 1 to 4, as a share of the generation (shade 5 is above
 * the last). With N = 50: 1–2 networks, 3–5, 6–10, 11–20, 21 or more.
 */
export const SHADE_LIMITS = [0.05, 0.1, 0.2, 0.4] as const;

/** The shade of a band holding `share` of the generation: 0 (empty, not drawn), else 1 (fewest) to 5 (most). */
export function shadeOf(share: number): number {
  if (!(share > 0)) return 0;
  const i = SHADE_LIMITS.findIndex((limit) => share <= limit + 1e-12);
  return i < 0 ? 5 : i + 1;
}

/** The CSS colour of a shade: the rank ramp's tokens, whose step 1 is the strongest in both themes. */
export function shadeColour(shade: number): string {
  return `var(--rank-${6 - shade})`;
}

/** The legend's words for every shade, for a generation of `n` networks. */
export function shadeLegend(n: number): { shade: number; label: string }[] {
  const limits = [0, ...SHADE_LIMITS];
  return limits.map((lo, i) => {
    const shade = i + 1;
    const first = Math.floor(lo * n + 1e-9) + 1;
    if (shade === 5) return { shade, label: `more than ${Math.round(lo * 100)} % (${first}+)` };
    const last = Math.floor(SHADE_LIMITS[i] * n + 1e-9);
    const count = first >= last ? `${last}` : `${first}–${last}`;
    return { shade, label: `up to ${Math.round(SHADE_LIMITS[i] * 100)} % (${count})` };
  });
}

/** A band's range in words, e.g. "0.10–0.15". */
export function bandLabel(edges: readonly number[], k: number): string {
  return `${edges[k].toFixed(2)}–${edges[k + 1].toFixed(2)}`;
}

/**
 * The generation the panel shows: the timeline slider's checkpoint unit (unit 0, the first
 * generation before it was scored, is generation 1), kept inside the generations there are.
 * Null while there are none.
 */
export function panelGeneration(sliderUnit: number | null, units: readonly number[]): number | null {
  if (units.length === 0) return null;
  const last = units[units.length - 1];
  if (sliderUnit === null) return last;
  return Math.min(Math.max(units[0], sliderUnit, 1), last);
}

export type StatKey = "best" | "median" | "mean" | "worst";

/** The fitness chart's four lines, in the categorical palette's fixed order (slot 0 = --series-1). */
export const FITNESS_LINES: { key: StatKey; label: string; slot: number }[] = [
  { key: "best", label: "best", slot: 0 },
  { key: "median", label: "median", slot: 1 },
  { key: "mean", label: "mean", slot: 2 },
  { key: "worst", label: "worst", slot: 3 },
];

/** Labels at the ends of lines, moved apart so no two are closer than `gap` px (order kept by y). */
export function spreadLabels<T extends { y: number }>(labels: readonly T[], gap: number, top: number, bottom: number): T[] {
  const sorted = [...labels].sort((a, b) => a.y - b.y).map((l) => ({ ...l }));
  for (let i = 0; i < sorted.length; i++) {
    sorted[i].y = Math.max(sorted[i].y, i === 0 ? top : sorted[i - 1].y + gap);
  }
  // pushed past the bottom: move the whole stack back up, keeping the gaps
  const over = sorted.length ? sorted[sorted.length - 1].y - bottom : 0;
  if (over > 0) {
    for (let i = sorted.length - 1; i >= 0; i--) {
      sorted[i].y = i === sorted.length - 1 ? bottom : Math.min(sorted[i].y - over, sorted[i + 1].y - gap);
    }
  }
  return sorted;
}

/** The bands holding networks in generation index `g`: [band, count] pairs, lowest band first. */
export function filledBands(view: EvolutionView, g: number): [number, number][] {
  const row = view.counts[g] ?? [];
  const out: [number, number][] = [];
  row.forEach((count, k) => {
    if (count > 0) out.push([k, count]);
  });
  return out;
}

/** The highest band edge any generation reaches (the top of the heat map), at least 0.5. */
export function topEdge(view: EvolutionView): number {
  let k = 0;
  for (const row of view.counts) row.forEach((count, b) => count > 0 && (k = Math.max(k, b + 1)));
  const top = view.edges[Math.min(k, view.edges.length - 1)] ?? 2;
  return Math.max(top, 0.5);
}

// ── one start at a time (task 7.7 follow-up, ADR-059) ───────────────────────────────────────

/** Upper limits of the starts map's shades 1 to 4, as the share of a generation that finishes a start. */
export const FINISH_LIMITS = [0.2, 0.4, 0.6, 0.8] as const;

/** The shade of a start finished by `share` of the generation: 0 (nobody), else 1 (fewest) to 5 (more than 80 %). */
export function finishShade(share: number): number {
  if (!(share > 0)) return 0;
  const i = FINISH_LIMITS.findIndex((limit) => share <= limit + 1e-12);
  return i < 0 ? 5 : i + 1;
}

/** The starts map legend's words for shades 0 to 5. */
export function finishLegend(): { shade: number; label: string }[] {
  const limits = [0, ...FINISH_LIMITS];
  return [
    { shade: 0, label: "none" },
    ...limits.map((lo, i) => ({
      shade: i + 1,
      label: i === limits.length - 1 ? `more than ${lo * 100} %` : `up to ${FINISH_LIMITS[i] * 100} %`,
    })),
  ];
}

export type Outcome = "finished" | "crashed" | "stalled" | "timeout";

/**
 * How a drive can end, in words and colour. The colours are the reserved status colours, so a
 * cell is never read by colour alone: the legend names them and the hover and table say the word.
 */
export const OUTCOMES: { key: Outcome; label: string; colour: string }[] = [
  { key: "finished", label: "finished", colour: "var(--ok)" },
  { key: "crashed", label: "crashed", colour: "var(--bad)" },
  { key: "stalled", label: "stalled", colour: "var(--warn)" },
  { key: "timeout", label: "out of time", colour: "var(--chart-mid)" },
];

/** The words and colour of an outcome name from the server (an unknown name is shown as is, in grey). */
export function outcomeOf(name: string): { key: string; label: string; colour: string } {
  return OUTCOMES.find((o) => o.key === name) ?? { key: name, label: name, colour: "var(--chart-mid)" };
}

/** Where start k is, in words: "start 3: 40 % of the lap, 1.5 m left" (k is 0-based, shown from 1). */
export function startLabel(k: number, fraction: number | undefined, lateral: number | undefined): string {
  if (fraction === undefined || lateral === undefined) return `start ${k + 1}`;
  const side = lateral === 0 ? "on the centre line" : `${Math.abs(lateral)} m ${lateral > 0 ? "left" : "right"}`;
  return `start ${k + 1}: ${Math.round(fraction * 100)} % of the lap, ${side}`;
}

/** The fitness symbol the charts use: F (the mean over every start) or f on start k (shown from 1). */
export function fitnessName(start: number | null): string {
  return start === null ? "fitness F" : `fitness f on start ${start + 1}`;
}
