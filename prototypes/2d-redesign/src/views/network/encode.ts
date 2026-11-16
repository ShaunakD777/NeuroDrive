// How numbers become marks in the network diagram. Pure functions: no React, no network maths
// (production: web/src/views/network/networkLayout.ts and diff.ts hold the same kind of helper).

import type { Mutation, Network } from "../../mock/types";
import type { LineMode } from "../../state/store";

export interface Point {
  x: number;
  y: number;
}

export const FRAME = { width: 860, height: 510, left: 200, right: 686, top: 60, bottom: 42 };
export const NODE_R = 10;
export const OUTPUT_R = 13;

/**
 * Node positions, one array per layer. Columns are spread evenly between the left and right
 * margins. Rows share one spacing (set by the widest layer) so the hidden layers line up; the
 * output layer gets a wider spacing because each output carries a label, a value and a bar.
 */
export function layoutNodes(sizes: number[]): Point[][] {
  const columns = sizes.length;
  const widest = Math.max(...sizes);
  const usable = FRAME.height - FRAME.top - FRAME.bottom;
  const gap = Math.min(30, widest > 1 ? usable / (widest - 1) : 0);
  const middle = FRAME.top + usable / 2;
  return sizes.map((size, layer) => {
    const x = FRAME.left + ((FRAME.right - FRAME.left) * layer) / Math.max(1, columns - 1);
    const g = layer === columns - 1 ? Math.max(gap, 104) : gap;
    const top = middle - (g * (size - 1)) / 2;
    return Array.from({ length: size }, (_, i) => ({ x, y: top + g * i }));
  });
}

/** A smooth S-curve between two nodes: it leaves and arrives horizontally, so fans stay readable. */
export function edgePath(a: Point, b: Point, ra: number, rb: number): string {
  const x1 = a.x + ra;
  const x2 = b.x - rb;
  const mid = (x1 + x2) / 2;
  return `M${x1} ${a.y}C${mid} ${a.y} ${mid} ${b.y} ${x2} ${b.y}`;
}

export interface Stroke {
  /** pos = blue, neg = orange, other = grey dashed (second parent), none = barely there. */
  tone: "pos" | "neg" | "other" | "none";
  width: number;
  opacity: number;
}

/** Weights at or beyond this size are drawn at full thickness (as production: WEIGHT_FULL). */
const WEIGHT_FULL = 1;

export function weightStroke(w: number): Stroke {
  const size = Math.min(1, Math.abs(w) / WEIGHT_FULL);
  return { tone: w < 0 ? "neg" : "pos", width: 0.35 + 3.4 * size, opacity: 0.1 + 0.75 * size };
}

/** Flow: what the connection carries now (weight × source value), relative to its layer's largest. */
export function flowStroke(c: number, layerMax: number): Stroke {
  const size = Math.min(1, Math.abs(c) / Math.max(layerMax, 1e-9));
  return { tone: c < 0 ? "neg" : "pos", width: 0.3 + 4.4 * size ** 1.3, opacity: 0.04 + 0.92 * size ** 1.5 };
}

/** A mutation: the nudge of a mutated weight, grey for genes of the second parent (task 7.8). */
export function mutationStroke(change: number, mutated: boolean, fromOther: boolean): Stroke {
  if (mutated) return { tone: change < 0 ? "neg" : "pos", width: 1.2 + 3.6 * Math.min(1, Math.abs(change)), opacity: 1 };
  if (fromOther) return { tone: "other", width: 0.8, opacity: 0.5 };
  return { tone: "none", width: 0.5, opacity: 0.05 };
}

export interface Edge {
  layer: number;
  from: number;
  to: number;
  weight: number;
}

/** Every connection, weakest first so the strong ones are drawn on top. */
export function collectEdges(network: Network): Edge[] {
  const edges: Edge[] = [];
  network.weights.forEach((matrix, layer) =>
    matrix.forEach((row, to) => row.forEach((weight, from) => edges.push({ layer, from, to, weight }))),
  );
  return edges.sort((a, b) => Math.abs(a.weight) - Math.abs(b.weight));
}

export function strokeFor(
  mode: LineMode,
  e: Edge,
  flow: number[][][] | null,
  flowMax: number[],
  mutation: Mutation | null,
): Stroke {
  if (mode === "flow" && flow !== null) return flowStroke(flow[e.layer][e.to][e.from], flowMax[e.layer]);
  if (mode === "mutation" && mutation !== null) {
    return mutationStroke(
      mutation.weight_change[e.layer][e.to][e.from],
      mutation.weight_mutated[e.layer][e.to][e.from],
      mutation.weight_from_other[e.layer][e.to][e.from],
    );
  }
  return weightStroke(e.weight);
}

/**
 * The strongest routes from the inputs to one output, as a set of edge keys: the `fan[0]`
 * largest contributions into the output, then for each of those neurons its `fan[1]` largest,
 * and so on back to the inputs. This is what "why" looks like as a picture.
 */
export function strongPaths(flow: number[][][], output: number, fan = [4, 2, 2]): Set<string> {
  const keys = new Set<string>();
  let targets = [output];
  for (let layer = flow.length - 1, depth = 0; layer >= 0; layer--, depth++) {
    const next = new Set<number>();
    for (const to of targets) {
      const row = flow[layer][to];
      const order = row.map((c, from) => ({ from, size: Math.abs(c) })).sort((a, b) => b.size - a.size);
      for (const { from } of order.slice(0, fan[depth] ?? 1)) {
        keys.add(`${layer}-${from}-${to}`);
        next.add(from);
      }
    }
    targets = [...next];
  }
  return keys;
}

export function nodeName(layer: number, index: number, layers: number, inputs: string[], outputs: string[]): string {
  if (layer === 0) return inputs[index] ?? `Input ${index + 1}`;
  if (layer === layers - 1) return outputs[index] ?? `Output ${index + 1}`;
  return `Hidden ${layer} · ${index + 1}`;
}
