import { type ReactNode, type RefObject, useEffect, useRef, useState } from "react";

// The chart kit: the few pieces every chart in the Experiment view is built from, so they all
// share one look (DESIGN_SPEC.md §8). Charts are plain SVG drawn at the size they are shown, so
// text is never scaled or blurred. Production keeps its own helpers (views/experiment/charts.ts);
// these components wrap the same ideas.

export type Scale = (value: number) => number;

export function linear(d0: number, d1: number, r0: number, r1: number): Scale {
  const span = d1 - d0 || 1;
  return (v) => r0 + ((v - d0) / span) * (r1 - r0);
}

export function logScale(d0: number, d1: number, r0: number, r1: number): Scale {
  const a = Math.log10(d0);
  const span = Math.log10(d1) - a || 1;
  return (v) => r0 + ((Math.log10(Math.max(v, d0)) - a) / span) * (r1 - r0);
}

/** About `count` round tick values covering lo..hi. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  const raw = (hi - lo) / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw) ?? power * 10;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

/** Powers of ten from lo to hi. */
export function logTicks(lo: number, hi: number): number[] {
  const ticks: number[] = [];
  for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) ticks.push(10 ** e);
  return ticks;
}

export function formatPower(v: number): string {
  const e = Math.round(Math.log10(v));
  if (e >= -2 && e <= 3) return String(Number(v.toPrecision(1)));
  return `1e${e}`;
}

export function linePath(xs: number[], ys: number[]): string {
  let d = "";
  let pen = false;
  for (let i = 0; i < xs.length; i++) {
    if (!Number.isFinite(ys[i])) {
      pen = false;
      continue;
    }
    d += `${pen ? "L" : "M"}${xs[i].toFixed(1)} ${ys[i].toFixed(1)}`;
    pen = true;
  }
  return d;
}

export function bandPath(xs: number[], lo: number[], hi: number[]): string {
  if (xs.length === 0) return "";
  let d = `M${xs[0].toFixed(1)} ${hi[0].toFixed(1)}`;
  for (let i = 1; i < xs.length; i++) d += `L${xs[i].toFixed(1)} ${hi[i].toFixed(1)}`;
  for (let i = xs.length - 1; i >= 0; i--) d += `L${xs[i].toFixed(1)} ${lo[i].toFixed(1)}`;
  return `${d}Z`;
}

/** The width of an element, kept up to date. Charts draw at this width. */
export function useWidth<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (el === null) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

export interface Box {
  width: number;
  height: number;
  left: number;
  right: number;
  top: number;
  bottom: number;
}

/** An SVG that fills its container's width at a fixed height and hands its box to `children`. */
export function Plot({
  height,
  margin,
  children,
  label,
  testId,
}: {
  height: number;
  margin: { top: number; right: number; bottom: number; left: number };
  children: (box: Box) => ReactNode;
  label: string;
  testId?: string;
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const box: Box = { width, height, left: margin.left, right: width - margin.right, top: margin.top, bottom: height - margin.bottom };
  return (
    <div ref={ref} className="plot" style={{ height }}>
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label={label} data-testid={testId}>
          {children(box)}
        </svg>
      )}
    </div>
  );
}

/** Horizontal gridlines with their values at the left: recessive, hairline, solid. */
export function YAxis({ box, scale, ticks, format = String, label }: { box: Box; scale: Scale; ticks: number[]; format?: (v: number) => string; label?: string }) {
  return (
    <g className="axis">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={box.left} x2={box.right} y1={scale(t)} y2={scale(t)} className="grid" />
          <text x={box.left - 8} y={scale(t) + 3.5} textAnchor="end" className="tick">
            {format(t)}
          </text>
        </g>
      ))}
      {label && (
        <text x={12} y={(box.top + box.bottom) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(box.top + box.bottom) / 2})`} className="axis-label">
          {label}
        </text>
      )}
    </g>
  );
}

export function XAxis({ box, scale, ticks, format = String, label }: { box: Box; scale: Scale; ticks: number[]; format?: (v: number) => string; label?: string }) {
  return (
    <g className="axis">
      <line x1={box.left} x2={box.right} y1={box.bottom} y2={box.bottom} className="baseline" />
      {ticks.map((t) => (
        <text key={t} x={scale(t)} y={box.bottom + 15} textAnchor="middle" className="tick">
          {format(t)}
        </text>
      ))}
      {label && (
        <text x={(box.left + box.right) / 2} y={box.bottom + 30} textAnchor="middle" className="axis-label">
          {label}
        </text>
      )}
    </g>
  );
}

export interface TipState {
  x: number;
  y: number;
  content: ReactNode;
}

/** The floating read-out that follows the pointer (an HTML box: sharp at any chart size). */
export function Tip({ tip }: { tip: TipState | null }) {
  if (tip === null) return null;
  const left = Math.min(tip.x + 14, window.innerWidth - 260);
  return (
    <div className="float-tip" style={{ left, top: tip.y + 16 }} data-testid="chart-tip">
      {tip.content}
    </div>
  );
}

/** One row of a tooltip: a colour key, a name and a number. */
export function TipRow({ color, label, value }: { color?: string; label: ReactNode; value: ReactNode }) {
  return (
    <div className="row">
      <span>
        {color && <i className="tip-key" style={{ background: color }} />}
        {label}
      </span>
      <span className="num">{value}</span>
    </div>
  );
}

const OBJECTIVES = {
  loss: { name: "Loss", arrow: "↓", note: "lower is better", tip: "A training objective of gradient methods. Comparable only between runs of the same task and loss." },
  fitness: { name: "Fitness", arrow: "↑", note: "higher is better", tip: "One score for a whole drive. Comparable only between runs with the same fitness function." },
  exam: { name: "Driving Exam", arrow: "↑", note: "the common yardstick", tip: "The same tracks, starts and limits for every network, however it was trained. The only measure that compares methods." },
  diversity: { name: "Diversity", arrow: "", note: "not an objective", tip: "A description of the population, not an objective." },
} as const;

/**
 * Says which kind of number a chart shows. Loss, fitness and reward are different objectives
 * (PROJECT_CONCEPT §3): the badge is on every chart so two of them are never read as one scale.
 */
export function Objective({ kind }: { kind: keyof typeof OBJECTIVES }) {
  const o = OBJECTIVES[kind];
  return (
    <span className="objective" data-kind={kind} data-tip={o.tip}>
      <b>{o.name}</b>
      {o.arrow && <i>{o.arrow}</i>}
      {o.note}
    </span>
  );
}

/** A chart with its heading: what it shows, which objective, and controls on the right. */
export function Figure({
  title,
  objective,
  note,
  actions,
  children,
  footer,
  testId,
}: {
  title: ReactNode;
  objective?: keyof typeof OBJECTIVES;
  note?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  testId?: string;
}) {
  return (
    <figure className="figure" data-testid={testId}>
      <figcaption className="figure-head">
        <div className="figure-titles">
          <span className="figure-title">{title}</span>
          {note && <span className="figure-note">{note}</span>}
        </div>
        {objective && <Objective kind={objective} />}
        {actions}
      </figcaption>
      {children}
      {footer && <div className="figure-foot">{footer}</div>}
    </figure>
  );
}
