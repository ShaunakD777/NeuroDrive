import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactNode } from "react";

import type { DatasetDetail, DatasetHistogram, DatasetMap, DatasetSummary, TrackGeometry } from "../../api/generated";
import { fetchDataset, fetchDatasets, fetchTrack } from "../../api/rest";
import { drawScene, readTheme } from "../environment/draw";
import { type Camera, fitCamera, panBy, scaleLimits, type Size, wheelFactor, worldToScreen, zoomAt } from "../environment/geometry";
import { ChartHover } from "./ChartHover";
import { formatNumber, linearScale, niceTicks } from "./charts";
import { colourScale, formatShare, type MapPalette, nearestPoint, parseHex, pointColour } from "./datasetMap";

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

/**
 * One quantity's histogram (task 6.2): a bar per bin over the quantity's whole contract range, so
 * empty bins show what the dataset never contains. The thin line under the axis marks the range
 * the values actually reach (min to max). The bins and counts come from the server; hovering a
 * bar shows its bin, count and share of the frames.
 */
function HistogramChart({ h, frames, big = false }: { h: DatasetHistogram; frames: number; big?: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = big ? 420 : 300;
  const H = big ? 190 : 150;
  const M = { left: 44, right: 10, top: 10, bottom: 38 };
  // The share axis runs from 0 to the first round tick at or above the tallest bar.
  const tallest = Math.max(...h.counts, 1) / Math.max(frames, 1);
  const rough = niceTicks(0, tallest, 3);
  const step = rough.length > 1 ? rough[1] - rough[0] : tallest;
  const yMax = Math.max(Math.ceil(tallest / step - 1e-9) * step, 1e-9);
  const yTicks = Array.from({ length: Math.round(yMax / step) + 1 }, (_, i) => Number((i * step).toPrecision(6)));
  const x = linearScale(h.lo, h.hi, M.left, W - M.right);
  const y = linearScale(0, yMax, H - M.bottom, M.top);
  const share = (c: number) => (frames > 0 ? c / frames : 0);
  const base = H - M.bottom;
  const label = MEANING[h.name] ?? h.name;

  return (
    <figure className="toy-figure ds-hist" data-testid="dataset-histogram" data-name={h.name}>
      <figcaption className="toy-figure-title">
        {h.name.replaceAll("_", " ")}
        <span className="ds-covered" data-testid="dataset-covered" data-covered={h.covered}>
          {" "}
          · covers {formatShare(h.covered)} of its range
          {h.min !== null && h.max !== null && ` (${formatNumber(h.min)} to ${formatNumber(h.max)})`}
        </span>
      </figcaption>
      <div className="cmp-plot">
        <svg className="toy-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Histogram of ${label} over ${frames} frames`}>
          {yTicks.map((t) => (
            <g key={t}>
              <line className="toy-grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
              <text className="toy-tick" x={M.left - 5} y={y(t) + 4} textAnchor="end">
                {formatShare(t)}
              </text>
            </g>
          ))}
          {[h.lo, (h.lo + h.hi) / 2, h.hi].map((t) => (
            <text key={`x${t}`} className="toy-tick" x={x(t)} y={base + 22} textAnchor="middle">
              {formatNumber(t)}
            </text>
          ))}
          <text className="toy-axis-title" x={(M.left + W - M.right) / 2} y={H - 3} textAnchor="middle">
            {label}
          </text>
          {h.counts.map((c, b) => {
            if (c === 0) return null;
            const x0 = x(h.edges[b]) + 1; // a 2 px gap between neighbouring bars
            const x1 = x(h.edges[b + 1]) - 1;
            const yTop = Math.min(y(share(c)), base - 1); // a non-empty bin is always visible
            return (
              <rect
                key={b}
                className={b === hover ? "ds-bar ds-bar-hover" : "ds-bar"}
                x={x0}
                y={yTop}
                width={Math.max(x1 - x0, 1)}
                height={base - yTop}
                rx={1.5}
              />
            );
          })}
          {h.min !== null && h.max !== null && (
            <line className="ds-range" data-testid="dataset-range" x1={x(h.min)} x2={Math.max(x(h.max), x(h.min) + 2)} y1={base + 6} y2={base + 6} />
          )}
          {h.counts.map((_, b) => (
            <rect
              key={`hit${b}`}
              className="toy-hit"
              x={x(h.edges[b])}
              y={M.top}
              width={x(h.edges[b + 1]) - x(h.edges[b])}
              height={base - M.top + 8}
              onMouseEnter={() => setHover(b)}
              onMouseLeave={() => setHover(null)}
            />
          ))}
        </svg>
        {hover !== null && (
          <ChartHover x={x(h.edges[hover + 1])} y={M.top} width={W} height={H} testId="dataset-hist-hover">
            <div className="cmp-hover-title">
              {formatNumber(h.edges[hover])} to {formatNumber(h.edges[hover + 1])}
            </div>
            <div>
              {h.counts[hover]} frames ({formatShare(share(h.counts[hover]))})
            </div>
          </ChartHover>
        )}
      </div>
    </figure>
  );
}

const MAP_HEIGHT = 440;

/**
 * The map's colours from the current theme (light or dark): the diverging chart pair, and as the
 * neutral midpoint the muted text grey, which stays visible on the white (or dark) road where the
 * charts' pale midpoint would vanish.
 */
function readPalette(el: Element): MapPalette {
  const style = getComputedStyle(el);
  const get = (name: string, fallback: [number, number, number]) => parseHex(style.getPropertyValue(name)) ?? fallback;
  return { pos: get("--chart-pos", [37, 99, 235]), neg: get("--chart-neg", [180, 83, 9]), mid: get("--text-muted", [91, 100, 114]) };
}

/**
 * The track map (task 6.2, ADR-045): the road, and on it every k-th frame of the dataset at the
 * position it was recorded, coloured by its label on a symmetric scale −m..+m (`colourScale`):
 * blue = steer right / throttle, orange = steer left / brake, grey = 0. The wheel zooms around
 * the mouse, dragging moves the map, and hovering a point shows that frame's labels. Drawn on a
 * canvas, since a dataset has thousands of points.
 */
function TrackMap({ geometry, map, colourBy }: { geometry: TrackGeometry | null; map: DatasetMap; colourBy: "steer" | "throttle" }) {
  const box = useRef<HTMLDivElement | null>(null);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const [width, setWidth] = useState(600);
  const [camera, setCamera] = useState<Camera | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const size: Size = useMemo(() => ({ w: width, h: MAP_HEIGHT }), [width]);
  const values = colourBy === "steer" ? map.steer : map.throttle;
  const scale = useMemo(() => colourScale(values), [values]);

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

  const screen = useMemo(() => {
    if (!cam) return { xs: [] as number[], ys: [] as number[] };
    const xs: number[] = [];
    const ys: number[] = [];
    for (let i = 0; i < map.x.length; i++) {
      const [px, py] = worldToScreen(cam, size, map.x[i], map.y[i]);
      xs.push(px);
      ys.push(py);
    }
    return { xs, ys };
  }, [cam, size, map]);

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
    for (let i = 0; i < screen.xs.length; i++) {
      ctx.fillStyle = pointColour(values[i], -scale, scale, palette);
      ctx.beginPath();
      ctx.arc(screen.xs[i], screen.ys[i], i === hover ? r + 2 : r, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [cam, size, geometry, values, scale, screen, hover]);

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

  const h = hover !== null && hover < map.x.length ? hover : null;
  return (
    <figure className="toy-figure ds-map" data-testid="dataset-map" data-points={map.x.length} data-colour={colourBy}>
      <figcaption className="toy-figure-title">
        Where the frames were recorded: every {map.stride === 1 ? "" : `${map.stride}th `}frame ({map.x.length} points),
        coloured by its {colourBy} label
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
        {h !== null && (
          <ChartHover x={screen.xs[h]} y={screen.ys[h]} width={size.w} height={size.h} testId="dataset-map-hover">
            <div className="cmp-hover-title">episode {map.episode[h]}</div>
            <div>steer {formatNumber(map.steer[h])}</div>
            <div>throttle {formatNumber(map.throttle[h])}</div>
          </ChartHover>
        )}
      </div>
      <div className="toy-legend" data-testid="dataset-map-legend" data-scale={scale}>
        <span className="toy-legend-item">
          {colourBy === "steer" ? "steer left" : "brake"} −{formatNumber(scale)}
          <span className="ds-ramp" />
          {colourBy === "steer" ? "steer right" : "full throttle"} +{formatNumber(scale)}
        </span>
        <span className="toy-legend-item ds-legend-note">grey = 0; the largest |{colourBy}| in the dataset is full colour</span>
        {camera !== null && (
          <button type="button" className="ds-reset" onClick={() => setCamera(null)} data-testid="dataset-map-reset">
            Show the whole track
          </button>
        )}
      </div>
    </figure>
  );
}

/**
 * The dataset inspector (task 6.2, ADR-045): pick a demonstration dataset from `data/demos/` and
 * see what the teacher did (label histograms), what the student will get to see (a histogram of
 * every observation over its whole range, with the share of the range covered) and where on the
 * track the frames were recorded. The page only draws: every number comes from the server.
 */
export function DatasetView({ headerExtra }: { headerExtra?: ReactNode }) {
  const [list, setList] = useState<DatasetSummary[] | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [geometry, setGeometry] = useState<TrackGeometry | null>(null);
  const [colourBy, setColourBy] = useState<"steer" | "throttle">("steer");
  const [error, setError] = useState<string | null>(null);

  function loadList() {
    fetchDatasets()
      .then((answer) => {
        setList(answer.datasets);
        setDatasetId((current) =>
          current && answer.datasets.some((d) => d.dataset_id === current) ? current : (answer.datasets[0]?.dataset_id ?? null),
        );
      })
      .catch((e: Error) => setError(e.message));
  }
  useEffect(loadList, []);

  useEffect(() => {
    if (datasetId === null) return;
    let live = true;
    setError(null);
    fetchDataset(datasetId)
      .then((d) => live && setDetail(d))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [datasetId]);

  const shown = detail !== null && detail.dataset_id === datasetId ? detail : null;
  const trackName = shown ? String(shown.meta.track) : null;
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

  const frames = shown ? Number(shown.meta.demo_frames) : 0;
  const episodes = (shown?.meta.episodes as Episode[] | undefined) ?? [];
  const finished = episodes.filter((e) => e.status === "finished").length;
  const teacher = shown ? String((shown.meta.teacher as { kind?: string } | undefined)?.kind ?? "?") : "";

  return (
    <section className="view toylab" data-testid="datasets">
      <div className="view-header">
        <h2>Experiment: demonstration datasets</h2>
        {headerExtra}
      </div>

      <div className="toylab-controls">
        <label className="picker">
          <span>Dataset </span>
          <select data-testid="dataset-select" value={datasetId ?? ""} onChange={(e) => setDatasetId(e.target.value)} disabled={!list?.length}>
            {(list ?? []).map((d) => (
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
          {(["steer", "throttle"] as const).map((c) => (
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
          <p className="toylab-summary" data-testid="dataset-summary" data-frames={frames}>
            Teacher <strong>{teacher}</strong> on <strong>{trackName}</strong>: {frames} frames from {episodes.length} episodes ({finished}{" "}
            finished). {formatShare(Number(shown.meta.straight_share))} of the frames steer nearly straight (|steer| &lt; 0.1).
          </p>

          <h3 className="ds-heading">What the teacher did: the labels</h3>
          <div className="ds-grid ds-grid-big">
            {shown.labels.map((h) => (
              <HistogramChart key={h.name} h={h} frames={frames} big />
            ))}
          </div>

          <h3 className="ds-heading">What the student will see: the observations</h3>
          <p className="hint">
            Each histogram spans the input's whole range. Empty stretches are situations the dataset holds no example of, so a network
            trained on it has to guess there.
          </p>
          <div className="ds-grid">
            {shown.observations.map((h) => (
              <HistogramChart key={h.name} h={h} frames={frames} />
            ))}
          </div>

          <h3 className="ds-heading">Where the frames were recorded</h3>
          <TrackMap geometry={geometry} map={shown.map} colourBy={colourBy} />

          <table className="compare-table" data-testid="dataset-episodes">
            <caption>The episodes: where each drive started, how it ended and how many frames it gave.</caption>
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
              {episodes.map((e) => (
                <tr key={e.index} data-testid="dataset-episode-row">
                  <td>{e.index}</td>
                  <td>{e.start_kind ?? "?"}</td>
                  <td>
                    {e.start_s0_m === undefined ? "?" : formatNumber(e.start_s0_m)},{" "}
                    {e.start_lateral_m === undefined ? "?" : formatNumber(e.start_lateral_m)}
                  </td>
                  <td>{e.status}</td>
                  <td>{e.frames}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
