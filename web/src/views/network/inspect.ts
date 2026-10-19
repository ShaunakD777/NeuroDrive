// Pure helpers for the output inspector, the what-if panel and the ray colours. They only sort,
// label and format numbers the server computed (contributions, sensitivities, what-if answers);
// no network maths happens here (ADR-012).

import type {
  ExplanationMessage,
  Explanation,
  FrameMessage,
  GhostPaths,
  OutputReceipt,
} from "../../api/generated";
import { describeSteer } from "./curve";
import { describeThrottle, INPUT_LABELS, num, OUTPUT_LABELS } from "./networkLayout";

/** Words for output `m`'s value: steer 0, throttle 1. */
export function describeOutput(m: number, value: number): string {
  return m === 0 ? describeSteer(value) : describeThrottle(value);
}

/** "Steer = −0.93 → hard left". */
export function outputHeadline(m: number, value: number): string {
  return `${OUTPUT_LABELS[m] ?? `Output ${m + 1}`} = ${num(value)} → ${describeOutput(m, value)}`;
}

export interface ReceiptRow {
  /** Index of the hidden neuron (0-based). */
  index: number;
  label: string;
  contribution: number;
}

/** The receipt's rows, largest |contribution| first; ties keep the neuron order. */
export function rankedReceipt(receipt: OutputReceipt, hiddenLayer: number): ReceiptRow[] {
  return receipt.contributions
    .map((c, index) => ({ index, label: `Hidden ${hiddenLayer}, neuron ${index + 1}`, contribution: c }))
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution) || a.index - b.index);
}

/** Split ranked rows into the `top` shown one by one and the sum of the rest. */
export function topAndRest(rows: ReceiptRow[], top: number): { shown: ReceiptRow[]; rest: number; restCount: number } {
  const shown = rows.slice(0, top);
  const others = rows.slice(top);
  return { shown, rest: others.reduce((s, r) => s + r.contribution, 0), restCount: others.length };
}

/** The largest |value| in a list (at least `floor`, so an all-zero list does not divide by 0). */
export function maxAbs(values: number[], floor = 1e-6): number {
  return values.reduce((m, v) => Math.max(m, Math.abs(v)), floor);
}

/**
 * The sentence that reads a sensitivity: "If Ray +60° rose by 0.1, steer would change by about
 * −0.036 (to the left)". `s` is output `m`'s sensitivity to input `k`.
 */
export function nudgeSentence(m: number, k: number, s: number): string {
  const output = (OUTPUT_LABELS[m] ?? `output ${m + 1}`).toLowerCase();
  const change = 0.1 * s;
  const direction =
    Math.abs(change) < 0.0005
      ? "it would barely move"
      : m === 0
        ? change < 0
          ? "towards the left"
          : "towards the right"
        : change < 0
          ? "towards braking"
          : "towards accelerating";
  return `If ${INPUT_LABELS[k] ?? `input ${k + 1}`} rose by 0.1, ${output} would change by about ${num(change, 3)} (${direction}).`;
}

/** The sensitivities of output `m` to the five rays, for colouring them (null when unavailable). */
export function raySensitivity(explain: Explanation | null | undefined, m: number | null): number[] | null {
  if (explain === null || explain === undefined || m === null) return null;
  const row = explain.sensitivity[m];
  return row === undefined ? null : row.slice(0, 5);
}

/** Line width for a ray coloured by sensitivity: 1.5 px for none, up to 7 px for the strongest. */
export function rayWidth(s: number, strongest: number): number {
  return 1.5 + 5.5 * Math.min(1, Math.abs(s) / Math.max(strongest, 1e-6));
}

/** The range a what-if slider may take: rays and speed 0..1, heading error −1..1. */
export function sliderRange(k: number): { min: number; max: number } {
  return k === 6 ? { min: -1, max: 1 } : { min: 0, max: 1 };
}

/**
 * The what-if ghost to draw, or null: only while some input is overridden and the server's answer
 * is about the car and the step on screen (an answer for another moment is never shown).
 */
export function activeGhost(
  live: { whatIfResult: ExplanationMessage | null; frame: FrameMessage | null },
  overrides: Record<number, number>,
): GhostPaths | null {
  const { whatIfResult: result, frame } = live;
  if (result === null || frame === null || Object.keys(overrides).length === 0) return null;
  if (result.step !== frame.step || result.car !== frame.selected?.index) return null;
  return result.ghost;
}

/** Distance in metres between where the two held-action paths end (display only). */
export function ghostGap(ghost: GhostPaths): number {
  return Math.hypot(ghost.what_if_end[0] - ghost.real_end[0], ghost.what_if_end[1] - ghost.real_end[1]);
}
