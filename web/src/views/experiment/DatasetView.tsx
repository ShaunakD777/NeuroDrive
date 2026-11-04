import { type CSSProperties, useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";

import type { DatasetDetail, DatasetHistogram, DatasetMap, DatasetSummary, TrackGeometry } from "../../api/generated";
import { fetchDataset, fetchDatasets, fetchTrack } from "../../api/rest";
import { drawScene, readTheme } from "../environment/draw";
import { type Camera, fitCamera, panBy, scaleLimits, type Size, wheelFactor, worldToScreen, zoomAt } from "../environment/geometry";
import { ChartHover } from "./ChartHover";
import { formatNumber, linearScale, niceTicks } from "./charts";
import { barSpan, denseStretches, formatShare, type MapPalette, nearestPoint, parseHex, pointColour, sharedColourScale } from "./datasetMap";

/** What each quantity means, in words, for the axis titles. */
const MEANING: Record<string, string> = {
  steer: "steer label (−1 hard left … +1 hard right)",
  throttle: "throttle label (−1 brake … +1 full throttle)",
  ray_far_left: "far-left ray (0 wall … 1 clear)",
  ray_left: "left ray (0 wall … 1 clear)",
  ray_centre: "centre ray (0 wall … 1 clear)",
  ray_right: "right ray (0 wall … 1 clear)",
  ray_far_right: "far-right ray (0 wall … 1 clear)",
  speed: "speed (0 standing … 1 top)",
  heading_error: "heading error (0 = along the road)",
};

interface Episode {
  index: number;
  start_kind?: string;
  start_s0_m?: number;
  start_lateral_m?: number;
  status: string;
  frames: number;
}

/** The two datasets of the comparator (6.2 follow-up, ADR-046) and their series colours. */
type Side = "a" | "b";
const SERIES: Record<Side, CSSProperties> = {
  a: { "--series": "var(--series-1)" } as CSSProperties,
  b: { "--series": "var(--series-2)" } as CSSProperties,
};

function Swatch({ side }: { side: Side }) {
  return <span className="cmp-swatch" style={SERIES[side]} aria-label={side === "a" ? "A" : "B"} />;
}

/** One dataset's histogram of a quantity, with that dataset's frame count. */
interface Hist {
  h: DatasetHistogram;
  frames: number;
}

/**
 * One quantity's histogram (task 6.2): a bar per bin over the quantity's whole contract range, so
 * empty bins show what the dataset never contains. Under the axis, a thin line marks the range
 * the values reach (min to max) and a thick line the stretches where the dataset is dense: bins
 * holding at least τ = 1 % of the frames (6.2 follow-up, ADR-046); the dashed line is that 1 %
 * bar height. With a second dataset `b`, each bin holds two bars, A's and B's, each the share of
 * its own dataset's frames, and each dataset gets its own reach and dense line. Every number
 * comes from the server; hovering a bin shows its counts and shares.
 */
function HistogramChart({ a, b = null, big = false }: { a: Hist; b?: Hist | null; big?: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const { h } = a;
  const sides: [Side, Hist][] = b ? [["a", a], ["b", b]] : [["a", a]];
  const W = big ? 420 : 300;
  const H = big ? 196 : 156;
  const M = { left: 44, right: 10, top: 10, bottom: 44 };
  const share = (s: Hist, c: number) => (s.frames > 0 ? c / s.frames : 0);
  const tau = a.frames > 0 ? h.dense_threshold / a.frames : 0;
  // The share axis runs from 0 to the first round tick at or above the tallest bar (or τ).
  const tallest = Math.max(...sides.map(([, s]) => Math.max(...s.h.counts, 1) / Math.max(s.frames, 1)), tau);
  const rough = niceTicks(0, tallest, 3);
  const step = rough.length > 1 ? rough[1] - rough[0] : tallest;
  const yMax = Math.max(Math.ceil(tallest / step - 1e-9) * step, 1e-9);
  const yTicks = Array.from({ length: Math.round(yMax / step) + 1 }, (_, i) => Number((i * step).toPrecision(6)));
  const x = linearScale(h.lo, h.hi, M.left, W - M.right);
  const y = linearScale(0, yMax, H - M.bottom, M.top);
  const base = H - M.bottom;
  const label = MEANING[h.name] ?? h.name;
  const stretches = sides.map(([, s]) => denseStretches(s.h.edges, s.h.counts, s.h.dense_threshold, s.h.min, s.h.max));
  const drawnDense = stretches[0].reduce((sum, [lo, hi]) => sum + hi - lo, 0) / (h.hi - h.lo);

  return (
    <figure
      className="toy-figure ds-hist"
      data-testid="dataset-histogram"
      data-name={h.name}
      data-dense-drawn={drawnDense}
      data-bars-b={b ? b.h.counts.filter((c) => c > 0).length : 0}
    >
      <figcaption className="toy-figure-title">
        {h.name.replaceAll("_", " ")}
        <span className="ds-covered" data-testid="dataset-covered" data-covered={h.covered} data-dense={h.dense} data-dense-b={b?.h.dense}>
          {" "}
          ·{" "}
          {b ? (
            <>
              densely covers <Swatch side="a" /> {formatShare(h.dense)} · <Swatch side="b" /> {formatShare(b.h.dense)} of its range (reaches{" "}
              {formatShare(h.covered)} · {formatShare(b.h.covered)})
            </>
          ) : (
            <>
              densely covers {formatShare(h.dense)} of its range (reaches {formatShare(h.covered)}
              {h.min !== null && h.max !== null && `: ${formatNumber(h.min)} to ${formatNumber(h.max)}`})
            </>
          )}
        </span>
      </figcaption>
      <div className="cmp-plot">
        <svg className="toy-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Histogram of ${label}`}>
          {yTicks.map((t) => (
            <g key={t}>
              <line className="toy-grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
              <text className="toy-tick" x={M.left - 5} y={y(t) + 4} textAnchor="end">
                {formatShare(t)}
              </text>
            </g>
          ))}
          {[h.lo, (h.lo + h.hi) / 2, h.hi].map((t) => (
            <text key={`x${t}`} className="toy-tick" x={x(t)} y={base + 28} textAnchor="middle">
              {formatNumber(t)}
            </text>
          ))}
          <text className="toy-axis-title" x={(M.left + W - M.right) / 2} y={H - 3} textAnchor="middle">
            {label}
          </text>
          {sides.map(([side, s], i) => (
            <g key={side} className={b ? `ds-side ds-side-${side}` : "ds-side"}>
              {s.h.counts.map((c, bin) => {
                if (c === 0) return null;
                const [x0, x1] = barSpan(x(h.edges[bin]), x(h.edges[bin + 1]), i, sides.length);
                const yTop = Math.min(y(share(s, c)), base - 1); // a non-empty bin is always visible
                return (
                  <rect
                    key={bin}
                    className={bin === hover ? "ds-bar ds-bar-hover" : "ds-bar"}
                    data-bin={bin}
                    x={x0}
                    y={yTop}
                    width={x1 - x0}
                    height={base - yTop}
                    rx={1.5}
                  />
                );
              })}
              {s.h.min !== null && s.h.max !== null && (
                <line
                  className="ds-range"
                  data-testid="dataset-range"
                  x1={x(s.h.min)}
                  x2={Math.max(x(s.h.max), x(s.h.min) + 2)}
                  y1={base + 7 + 7 * i}
                  y2={base + 7 + 7 * i}
                />
              )}
              {stretches[i].map(([lo, hi]) => (
                <line
                  key={lo}
                  className="ds-dense"
                  data-testid="dataset-dense"
                  x1={x(lo)}
                  x2={Math.max(x(hi), x(lo) + 2)}
                  y1={base + 7 + 7 * i}
                  y2={base + 7 + 7 * i}
                />
              ))}
            </g>
          ))}
          {tau > 0 && <line className="ds-threshold" data-testid="dataset-threshold" x1={M.left} x2={W - M.right} y1={y(tau)} y2={y(tau)} />}
          {h.counts.map((_, bin) => (
            <rect
              key={`hit${bin}`}
              className="toy-hit"
              x={x(h.edges[bin])}
              y={M.top}
              width={x(h.edges[bin + 1]) - x(h.edges[bin])}
              height={base - M.top + 8}
              onMouseEnter={() => setHover(bin)}
              onMouseLeave={() => setHover(null)}
            />
          ))}
        </svg>
        {hover !== null && (
          <ChartHover x={x(h.edges[hover + 1])} y={M.top} width={W} height={H} testId="dataset-hist-hover">
            <div className="cmp-hover-title">
              {formatNumber(h.edges[hover])} to {formatNumber(h.edges[hover + 1])}
            </div>
            {sides.map(([side, s]) => (
              <div key={side}>
                {b && <Swatch side={side} />} {s.h.counts[hover]} frames ({formatShare(share(s, s.h.counts[hover]))})
                {s.h.counts[hover] > 0 && s.h.counts[hover] >= s.h.dense_threshold ? ", dense" : ""}
              </div>
            ))}
          </ChartHover>
        )}
      </div>
    </figure>
  );
}

const MAP_HEIGHT = 440;

/** What the map's points are coloured by; "dataset" only while comparing on one track. */
type ColourBy = "steer" | "throttle" | "dataset";

interface MapColours extends MapPalette {
  a: string;
  b: string;
}

/**
 * The map's colours from the current theme (light or dark): the diverging chart pair, and as the
 * neutral midpoint the muted text grey, which stays visible on the white (or dark) road where the
 * charts' pale midpoint would vanish; for colouring by dataset, the series pair of the comparator.
 */
function readPalette(el: Element): MapColours {
  const style = getComputedStyle(el);
  const get = (name: string, fallback: [number, number, number]) => parseHex(style.getPropertyValue(name)) ?? fallback;
  const css = (name: string, fallback: string) => style.getPropertyValue(name).trim() || fallback;
  return {
    pos: get("--chart-pos", [37, 99, 235]),
    neg: get("--chart-neg", [180, 83, 9]),
    mid: get("--text-muted", [91, 100, 114]),
    a: css("--series-1", "#2a78d6"),
    b: css("--series-2", "#eb6834"),
  };
}

/**
 * The track map (task 6.2, ADR-045): the road, and on it every k-th frame of the dataset at the
 * position it was recorded, coloured by its label on a symmetric scale −m..+m (`colourScale`):
 * blue = steer right / throttle, orange = steer left / brake, grey = 0. With a second dataset on
 * the same track (6.2 follow-up, ADR-046), its points are drawn too: coloured by dataset (A and
 * B in their series colours), or by a label on one scale for both, B as rings. The wheel zooms
 * around the mouse, dragging moves the map, and hovering a point shows that frame's labels.
 * Drawn on a canvas, since a dataset has thousands of points.
 */
function TrackMap({
  geometry,
  map,
  mapB,
  colourBy,
}: {
  geometry: TrackGeometry | null;
  map: DatasetMap;
  mapB: DatasetMap | null;
  colourBy: ColourBy;
}) {
  const box = useRef<HTMLDivElement | null>(null);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const [width, setWidth] = useState(600);
  const [camera, setCamera] = useState<Camera | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const size: Size = useMemo(() => ({ w: width, h: MAP_HEIGHT }), [width]);
  const label = colourBy === "throttle" ? "throttle" : "steer";
  const values = map[label];
  const valuesB = mapB ? mapB[label] : null;
  const scale = useMemo(() => sharedColourScale(values, valuesB ?? []), [values, valuesB]);
  const nA = map.x.length;

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const observer = new ResizeObserver(() => setWidth(Math.max(200, Math.round(el.clientWidth))));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const fit = useMemo(() => (geometry ? fitCamera(geometry.left.concat(geometry.right), size) : null), [geometry, size]);
  useEffect(() => setCamera(null), [geometry, width]); // a new track or size shows the whole track again
  const cam = camera ?? fit;

  // A's points first, then B's: index i < nA is A's row i, otherwise B's row i − nA.
  const screen = useMemo(() => {
    if (!cam) return { xs: [] as number[], ys: [] as number[] };
    const xs: number[] = [];
    const ys: number[] = [];
    for (const m of mapB ? [map, mapB] : [map]) {
      for (let i = 0; i < m.x.length; i++) {
        const [px, py] = worldToScreen(cam, size, m.x[i], m.y[i]);
        xs.push(px);
        ys.push(py);
      }
    }
    return { xs, ys };
  }, [cam, size, map, mapB]);

  useEffect(() => {
    const el = canvas.current;
    if (!el || !cam) return;
    const dpr = window.devicePixelRatio || 1;
    el.width = Math.round(size.w * dpr);
    el.height = Math.round(size.h * dpr);
    const ctx = el.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawScene(ctx, readTheme(el), {
      size,
      camera: cam,
      geometry,
      scenario: null,
      frame: null,
      selected: 0,
      compact: false,
      highlight: [],
      raySensitivity: null,
      ghost: null,
      trail: null,
      moment: null,
    });
    const palette = readPalette(el);
    const r = Math.max(2, Math.min(4, cam.scale * 0.4));
    ctx.lineWidth = 1.5;
    for (let i = 0; i < screen.xs.length; i++) {
      const isB = i >= nA;
      const v = isB ? valuesB![i - nA] : values[i];
      const colour = colourBy === "dataset" ? (isB ? palette.b : palette.a) : pointColour(v, -scale, scale, palette);
      const radius = i === hover ? r + 2 : r;
      ctx.beginPath();
      if (isB && colourBy !== "dataset") {
        ctx.strokeStyle = colour; // B as rings, so both datasets stay apart on one colour scale
        ctx.arc(screen.xs[i], screen.ys[i], radius, 0, 2 * Math.PI);
        ctx.stroke();
      } else {
        ctx.fillStyle = colour;
        ctx.arc(screen.xs[i], screen.ys[i], radius, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }, [cam, size, geometry, values, valuesB, nA, colourBy, scale, screen, hover]);

  // The wheel zooms around the cursor. Registered by hand, as in the Environment view, because
  // React's wheel listener cannot stop the page from scrolling.
  const latest = useRef({ cam, fit, size });
  latest.current = { cam, fit, size };
  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      const { cam: c, fit: f, size: s } = latest.current;
      if (!c || !f) return;
      event.preventDefault();
      const rect = el.getBoundingClientRect();
      const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY; // lines → pixels
      setCamera(zoomAt(c, s, event.clientX - rect.left, event.clientY - rect.top, wheelFactor(delta), scaleLimits(f.scale)));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  function local(event: MouseEvent<HTMLCanvasElement>): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }
  function onMove(event: MouseEvent<HTMLCanvasElement>) {
    const [px, py] = local(event);
    if (drag.current && cam) {
      setCamera(panBy(cam, px - drag.current.x, py - drag.current.y));
      drag.current = { x: px, y: py };
      return;
    }
    setHover(nearestPoint(screen.xs, screen.ys, px, py));
  }

  const h = hover !== null && hover < screen.xs.length ? hover : null;
  const hoverMap = h !== null && h >= nA ? mapB! : map;
  const hoverRow = h !== null && h >= nA ? h - nA : h;
  const every = (m: DatasetMap) => `every ${m.stride === 1 ? "" : `${m.stride}th `}frame (${m.x.length} points)`;
  return (
    <figure
      className="toy-figure ds-map"
      data-testid="dataset-map"
      data-points={nA}
      data-points-b={mapB ? mapB.x.length : 0}
      data-colour={colourBy}
    >
      <figcaption className="toy-figure-title">
        Where the frames were recorded: {mapB ? <>A&apos;s {every(map)} and B&apos;s {every(mapB)}</> : every(map)}, coloured by{" "}
        {colourBy === "dataset" ? "dataset" : `the ${colourBy} label`}
      </figcaption>
      <div className="cmp-plot" ref={box}>
        <canvas
          ref={canvas}
          className="ds-canvas"
          style={{ width: "100%", height: MAP_HEIGHT }}
          onMouseDown={(e) => {
            const [px, py] = local(e);
            drag.current = { x: px, y: py };
          }}
          onMouseUp={() => (drag.current = null)}
          onMouseMove={onMove}
          onMouseLeave={() => {
            drag.current = null;
            setHover(null);
          }}
        />
        {h !== null && hoverRow !== null && (
          <ChartHover x={screen.xs[h]} y={screen.ys[h]} width={size.w} height={size.h} testId="dataset-map-hover">
            <div className="cmp-hover-title">
              {mapB && (
                <>
                  <Swatch side={h >= nA ? "b" : "a"} /> {h >= nA ? "B" : "A"},{" "}
                </>
              )}
              episode {hoverMap.episode[hoverRow]}
            </div>
            <div>steer {formatNumber(hoverMap.steer[hoverRow])}</div>
            <div>throttle {formatNumber(hoverMap.throttle[hoverRow])}</div>
          </ChartHover>
        )}
      </div>
      <div className="toy-legend" data-testid="dataset-map-legend" data-scale={scale}>
        {colourBy === "dataset" ? (
          <>
            <span className="toy-legend-item">
              <Swatch side="a" /> A
            </span>
            <span className="toy-legend-item">
              <Swatch side="b" /> B
            </span>
          </>
        ) : (
          <>
            <span className="toy-legend-item">
              {colourBy === "steer" ? "steer left" : "brake"} −{formatNumber(scale)}
              <span className="ds-ramp" />
              {colourBy === "steer" ? "steer right" : "full throttle"} +{formatNumber(scale)}
            </span>
            <span className="toy-legend-item ds-legend-note">
              grey = 0; the largest |{colourBy}| in {mapB ? "either dataset" : "the dataset"} is full colour
              {mapB && "; dots = A, rings = B"}
            </span>
          </>
        )}
        {camera !== null && (
          <button type="button" className="ds-reset" onClick={() => setCamera(null)} data-testid="dataset-map-reset">
            Show the whole track
          </button>
        )}
      </div>
    </figure>
  );
}

function episodesOf(detail: DatasetDetail): Episode[] {
  return (detail.meta.episodes as Episode[] | undefined) ?? [];
}

/** The header line of one dataset: teacher, track, frames, episodes, the share steering straight. */
function SummaryLine({ detail, side, testId }: { detail: DatasetDetail; side: Side | null; testId: string }) {
  const frames = Number(detail.meta.demo_frames);
  const episodes = episodesOf(detail);
  const finished = episodes.filter((e) => e.status === "finished").length;
  const teacher = String((detail.meta.teacher as { kind?: string } | undefined)?.kind ?? "?");
  return (
    <p className="toylab-summary" data-testid={testId} data-frames={frames}>
      {side && (
        <>
          <Swatch side={side} /> <strong>{side.toUpperCase()}</strong> <code>{detail.dataset_id}</code>:{" "}
        </>
      )}
      Teacher <strong>{teacher}</strong> on <strong>{String(detail.meta.track)}</strong>: {frames} frames from {episodes.length} episodes (
      {finished} finished). {formatShare(Number(detail.meta.straight_share))} of the frames steer nearly straight (|steer| &lt; 0.1).
    </p>
  );
}

/** A dataset's episodes: where each drive started, how it ended and how many frames it gave. */
function EpisodesTable({ detail, caption, testId }: { detail: DatasetDetail; caption: string; testId: string }) {
  return (
    <table className="compare-table" data-testid={testId}>
      <caption>{caption}</caption>
      <thead>
        <tr>
          <th scope="col">Episode</th>
          <th scope="col">Start</th>
          <th scope="col">Start (m along, m left)</th>
          <th scope="col">Ends</th>
          <th scope="col">Frames</th>
        </tr>
      </thead>
      <tbody>
        {episodesOf(detail).map((e) => (
          <tr key={e.index} data-testid={`${testId}-row`}>
            <td>{e.index}</td>
            <td>{e.start_kind ?? "?"}</td>
            <td>
              {e.start_s0_m === undefined ? "?" : formatNumber(e.start_s0_m)}, {e.start_lateral_m === undefined ? "?" : formatNumber(e.start_lateral_m)}
            </td>
            <td>{e.status}</td>
            <td>{e.frames}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Fetches one dataset's detail whenever `id` changes; null while loading or when `id` is null. */
function useDataset(id: string | null, onError: (message: string) => void): DatasetDetail | null {
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  useEffect(() => {
    if (id === null) return;
    let live = true;
    fetchDataset(id)
      .then((d) => live && setDetail(d))
      .catch((e: Error) => live && onError(e.message));
    return () => {
      live = false;
    };
  }, [id, onError]);
  return detail !== null && detail.dataset_id === id ? detail : null;
}

/**
 * The dataset inspector (task 6.2, ADR-045): pick a demonstration dataset from `data/demos/` and
 * see what the teacher did (label histograms), what the student will get to see (a histogram of
 * every observation over its whole range, with the share of the range covered densely and
 * reached) and where on the track the frames were recorded. "Compare with" (6.2 follow-up,
 * ADR-046) adds a second dataset B to every chart. The page only draws: every number comes from
 * the server.
 */
export function DatasetView({ headerExtra }: { headerExtra?: ReactNode }) {
  const [list, setList] = useState<DatasetSummary[] | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [compareId, setCompareId] = useState<string | null>(null);
  const [geometry, setGeometry] = useState<TrackGeometry | null>(null);
  const [colourBy, setColourBy] = useState<ColourBy>("steer");
  const [error, setError] = useState<string | null>(null);

  function loadList() {
    fetchDatasets()
      .then((answer) => {
        const known = (id: string | null) => id !== null && answer.datasets.some((d) => d.dataset_id === id);
        setList(answer.datasets);
        setDatasetId((current) => (known(current) ? current : (answer.datasets[0]?.dataset_id ?? null)));
        setCompareId((current) => (known(current) ? current : null));
      })
      .catch((e: Error) => setError(e.message));
  }
  useEffect(loadList, []);
  useEffect(() => setError(null), [datasetId, compareId]);

  const shown = useDataset(datasetId, setError);
  const shownB = useDataset(compareId, setError);
  const trackName = shown ? String(shown.meta.track) : null;
  const sameTrack = shownB !== null && String(shownB.meta.track) === trackName;
  useEffect(() => {
    if (trackName === null) return;
    let live = true;
    fetchTrack(trackName)
      .then((g) => live && setGeometry(g))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [trackName]);
  // Comparing on one track colours the map by dataset; without a second dataset that is not offered.
  useEffect(() => setColourBy((c) => (sameTrack ? "dataset" : c === "dataset" ? "steer" : c)), [sameTrack]);

  const frames = shown ? Number(shown.meta.demo_frames) : 0;
  const framesB = shownB ? Number(shownB.meta.demo_frames) : 0;
  const colourChoices: ColourBy[] = sameTrack ? ["dataset", "steer", "throttle"] : ["steer", "throttle"];

  return (
    <section className="view toylab" data-testid="datasets">
      <div className="view-header">
        <h2>Experiment: demonstration datasets</h2>
        {headerExtra}
      </div>

      <div className="toylab-controls">
        <label className="picker">
          <span>Dataset </span>
          <select
            data-testid="dataset-select"
            value={datasetId ?? ""}
            onChange={(e) => {
              if (e.target.value === compareId) setCompareId(null);
              setDatasetId(e.target.value);
            }}
            disabled={!list?.length}
          >
            {(list ?? []).map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.dataset_id} ({d.demo_frames} frames)
              </option>
            ))}
          </select>
        </label>
        <label className="picker">
          <span>Compare with </span>
          <select
            data-testid="dataset-compare"
            value={compareId ?? ""}
            onChange={(e) => setCompareId(e.target.value === "" ? null : e.target.value)}
            disabled={(list?.length ?? 0) < 2}
          >
            <option value="">nothing</option>
            {(list ?? [])
              .filter((d) => d.dataset_id !== datasetId)
              .map((d) => (
                <option key={d.dataset_id} value={d.dataset_id}>
                  {d.dataset_id} ({d.demo_frames} frames)
                </option>
              ))}
          </select>
        </label>
        <button type="button" onClick={loadList} data-testid="dataset-refresh">
          Refresh list
        </button>
        <div className="segmented" role="group" aria-label="Colour the map by">
          {colourChoices.map((c) => (
            <button key={c} type="button" aria-pressed={c === colourBy} onClick={() => setColourBy(c)} data-testid={`dataset-colour-${c}`}>
              Colour by {c}
            </button>
          ))}
        </div>
      </div>

      {error && <p className="toylab-error">Could not load: {error}</p>}
      {list !== null && list.length === 0 && (
        <p className="hint" data-testid="dataset-empty">
          No datasets yet. Record one with <code>uv run python -m neurodrive demo-record --teacher expert</code> and press Refresh list.
        </p>
      )}

      {shown && (
        <>
          <SummaryLine detail={shown} side={shownB ? "a" : null} testId="dataset-summary" />
          {shownB && <SummaryLine detail={shownB} side="b" testId="dataset-summary-b" />}
          {shownB && (
            <p className="hint">
              Each bar is the share of its own dataset&apos;s frames, so datasets of different sizes compare fairly.
            </p>
          )}

          <h3 className="ds-heading">What the teacher did: the labels</h3>
          <div className="ds-grid ds-grid-big">
            {shown.labels.map((h, i) => (
              <HistogramChart key={h.name} a={{ h, frames }} b={shownB ? { h: shownB.labels[i], frames: framesB } : null} big />
            ))}
          </div>

          <h3 className="ds-heading">What the student will see: the observations</h3>
          <p className="hint">
            Each histogram spans the input&apos;s whole range. Under the axis, the thin line is the range the values reach and the thick
            line where they are dense: bins holding at least 1 % of the frames (the dashed line), enough for training to take them
            into account. Outside the thick line a network trained on the dataset has seen few or no examples, so it has to guess there.
          </p>
          <div className="ds-grid">
            {shown.observations.map((h, i) => (
              <HistogramChart key={h.name} a={{ h, frames }} b={shownB ? { h: shownB.observations[i], frames: framesB } : null} />
            ))}
          </div>

          <h3 className="ds-heading">Where the frames were recorded</h3>
          {shownB && !sameTrack && (
            <p className="hint" data-testid="dataset-map-other-track">
              B was recorded on {String(shownB.meta.track)}, so the map shows A only.
            </p>
          )}
          <TrackMap geometry={geometry} map={shown.map} mapB={sameTrack && shownB ? shownB.map : null} colourBy={colourBy} />

          <EpisodesTable
            detail={shown}
            caption={`${shownB ? "A: the" : "The"} episodes: where each drive started, how it ended and how many frames it gave.`}
            testId="dataset-episodes"
          />
          {shownB && <EpisodesTable detail={shownB} caption="B: the episodes." testId="dataset-episodes-b" />}
        </>
      )}
    </section>
  );
}
