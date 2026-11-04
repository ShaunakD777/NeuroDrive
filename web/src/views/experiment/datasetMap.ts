// Display helpers for the dataset inspector (task 6.2): colours for the track map's points,
// finding the point under the mouse, and (6.2 follow-up) placing paired bars and the dense line. Pure functions, so they can be unit-tested. Every number
// about the dataset comes from the server; this only decides how to draw it.

import { divergingMix } from "./charts";

/** "#rrggbb" (or "#rgb") to [r, g, b] in 0..255; anything else gives null. */
export function parseHex(colour: string): [number, number, number] | null {
  const c = colour.trim();
  const long = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(c);
  if (long) return [parseInt(long[1], 16), parseInt(long[2], 16), parseInt(long[3], 16)];
  const short = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(c);
  if (short) return [parseInt(short[1] + short[1], 16), parseInt(short[2] + short[2], 16), parseInt(short[3] + short[3], 16)];
  return null;
}

export interface MapPalette {
  pos: [number, number, number];
  neg: [number, number, number];
  mid: [number, number, number];
}

/**
 * The canvas colour of a label value on the diverging scale lo..hi: the neutral midpoint at the
 * middle, mixed towards `pos` above it and `neg` below it in proportion to the distance (the
 * same rule as the toy lab's maps, `divergingMix`). Canvas cannot read CSS variables or
 * color-mix, so the mix is done here, channel by channel. Example: steer 0.5 on −1..1 is 50 % of
 * the way from the midpoint to `pos`.
 */
export function pointColour(value: number, lo: number, hi: number, palette: MapPalette): string {
  const { pole, percent } = divergingMix(value, lo, hi);
  const end = pole === "pos" ? palette.pos : pole === "neg" ? palette.neg : palette.mid;
  const t = percent / 100;
  const ch = (i: number) => Math.round(palette.mid[i] + (end[i] - palette.mid[i]) * t);
  return `rgb(${ch(0)}, ${ch(1)}, ${ch(2)})`;
}

/**
 * The half-width m of the map's symmetric colour scale −m..+m: the largest |value| sent, rounded
 * up to 2 decimals (1 when every value is 0). The expert steers within ±0.29, so on a fixed ±1
 * scale every point would look nearly neutral; on ±m the hardest steering shown is full colour
 * and 0 stays the neutral midpoint. Example: [0.1, −0.287, 0.2] gives m = 0.29.
 */
export function colourScale(values: readonly number[]): number {
  let m = 0;
  for (const v of values) m = Math.max(m, Math.abs(v));
  return m === 0 ? 1 : Math.ceil(m * 100 - 1e-9) / 100;
}

/**
 * One colour scale for two datasets shown together (6.2 follow-up, ADR-046): `colourScale` over
 * the values of both, so the same colour means the same label in A and in B. Example: A's steers
 * reach 0.29 and B's 0.6, so both are coloured on ±0.6.
 */
export function sharedColourScale(...lists: readonly (readonly number[])[]): number {
  return Math.max(...lists.map((values) => (values.length === 0 ? 0 : colourScale(values))), 0) || 1;
}

/**
 * The left and right screen x of bar `slot` (0-based) of `slots` bars side by side in the bin
 * from x0 to x1: 1 px gap at each bin edge (2 px between bins, as with one bar) and 1 px between
 * the bars. Example: the bin 100..141 with 2 slots gives 101..120 and 121..140.
 */
export function barSpan(x0: number, x1: number, slot: number, slots: number): [number, number] {
  const inner = x1 - x0 - 2 - (slots - 1);
  const width = Math.max(inner / slots, 1);
  const left = x0 + 1 + slot * (width + 1);
  return [left, left + width];
}

/**
 * The stretches of the axis where the dataset is dense (6.2 follow-up, ADR-046): every bin whose
 * count reaches `threshold` (tau·F, from the server), clipped to [min, max], with neighbouring
 * stretches joined. The server's dense coverage d is their summed length / (hi − lo); this only
 * decides where to draw the thick line. Example: edges 0, 0.25, 0.5, 0.75, 1, counts 3, 2, 0, 1,
 * threshold 1.5, min 0.1, max 0.9 give one stretch, 0.1..0.5.
 */
export function denseStretches(
  edges: readonly number[],
  counts: readonly number[],
  threshold: number,
  min: number | null,
  max: number | null,
): [number, number][] {
  if (min === null || max === null) return [];
  const out: [number, number][] = [];
  counts.forEach((c, b) => {
    if (c === 0 || c < threshold) return;
    const a = Math.max(edges[b], min);
    const z = Math.min(edges[b + 1], max);
    if (z <= a) return;
    const last = out[out.length - 1];
    if (last && Math.abs(last[1] - a) < 1e-12) last[1] = z;
    else out.push([a, z]);
  });
  return out;
}

/**
 * The index of the screen point nearest to (px, py), or null when none lies within `radius`
 * pixels. `xs` and `ys` are the points' screen coordinates.
 */
export function nearestPoint(xs: readonly number[], ys: readonly number[], px: number, py: number, radius = 8): number | null {
  let best: number | null = null;
  let bestD = radius * radius;
  for (let i = 0; i < xs.length; i++) {
    const d = (xs[i] - px) ** 2 + (ys[i] - py) ** 2;
    if (d <= bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

/** A share 0..1 as a percentage for reading: "3.4 %", "0.03 %" for tiny non-zero shares, "0 %". */
export function formatShare(share: number): string {
  if (share === 0) return "0 %";
  const p = share * 100;
  if (p < 0.1) return `${p.toPrecision(1)} %`;
  return `${p < 10 ? p.toFixed(1) : Math.round(p)} %`;
}
