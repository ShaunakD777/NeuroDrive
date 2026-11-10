import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import type { RunSummary, StartShift } from "../../api/generated";
import { fetchExamCard, fetchRuns } from "../../api/rest";
import { connectLive, type LiveConnection } from "../../api/ws";
import { useLive } from "../../state/live";
import { applyRunsMessage, defaultMetric, initialRunsWatch, plottableKeys, series, watchRun } from "../../state/runsWatch";
import { useSelection } from "../../state/selection";
import { formatNumber, shortName } from "./charts";
import { LiveChart } from "./LiveChart";

/** One car's result on one exam track, as the card stores it (only the fields shown here). */
interface Episode {
  start_index: number;
  start_s0_m: number;
  start_lateral_m: number;
  status: string;
  progress_frac: number;
  time_s: number;
  laps: number;
}

interface CardTrack {
  track: string;
  split: string;
  episodes: Episode[];
}

/** Runs whose checkpoints drive: everything but toy runs, with at least one checkpoint. */
function drivingRuns(runs: readonly RunSummary[]): RunSummary[] {
  return runs.filter((r) => !r.method.startsWith("toy_") && r.checkpoints > 0);
}

/** The shift for the server (radians), or null when both numbers are 0 (the exam's own starts). */
export function startShift(lateralM: number, headingDeg: number): StartShift | null {
  if (lateralM === 0 && headingDeg === 0) return null;
  return { lateral_m: lateralM, heading_rad: (headingDeg * Math.PI) / 180 };
}

function sameShift(a: StartShift | null, b: StartShift | null): boolean {
  return a === b || (a !== null && b !== null && a.lateral_m === b.lateral_m && a.heading_rad === b.heading_rad);
}

/**
 * The run timeline (task 5.9, ADR-042): pick a driving run and a track, then move along the run's
 * checkpoints. Every move replays that checkpoint's exam drive in the live session (the
 * Environment view shows the 10 cars, the Network view the checkpoint's network and what changed),
 * puts the cursor on the run's chart, and lists what the exam card says each car did on that
 * track. The page only sends commands and draws: the drive, the card and the change come from
 * the server. The run's history arrives over /ws/runs, so a run still training grows here too.
 */
export function Timeline({ headerExtra }: { headerExtra?: ReactNode }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [track, setTrack] = useState("train_loop_01");
  const [index, setIndex] = useState<number | null>(null);
  const [card, setCard] = useState<{ key: string; tracks: CardTrack[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [watch, setWatch] = useState(initialRunsWatch);
  // Perturbed starts (task 6.5, ADR-048): every exam start moved sideways and turned.
  const [shiftM, setShiftM] = useState(0);
  const [shiftDeg, setShiftDeg] = useState(0);
  const shift = startShift(shiftM, shiftDeg);
  const socket = useRef<LiveConnection | null>(null);
  const watched = useRef<string | null>(null);

  const loadScenario = useLive((s) => s.loadScenario);
  const send = useLive((s) => s.send);
  const replaying = useLive((s) => s.scenario?.checkpoint ?? null);
  const replayedShift = useLive((s) => s.scenario?.start_shift ?? null);
  const policy = useLive((s) => s.scenario?.policy ?? null);
  const selectedCar = useLive((s) => s.frame?.selected?.index ?? null);
  const selectCar = useSelection((s) => s.selectCar);

  function loadRuns() {
    fetchRuns()
      .then((list) => {
        const driving = drivingRuns(list.runs);
        setRuns(driving);
        setRunId((current) => {
          if (current && driving.some((r) => r.run_id === current)) return current;
          return (driving.find((r) => r.method === "random_search") ?? driving[0])?.run_id ?? null;
        });
      })
      .catch((e: Error) => setError(e.message));
  }
  useEffect(loadRuns, []);

  // The run's metrics, checkpoints and exam headlines over /ws/runs (task 5.8).
  useEffect(() => {
    const connection = connectLive(
      (status) => {
        if (status === "connected" && watched.current !== null)
          socket.current?.send(JSON.stringify({ type: "watch", run_id: watched.current }));
      },
      (message) => setWatch((s) => applyRunsMessage(s, message)),
      "/ws/runs",
    );
    socket.current = connection;
    return () => {
      socket.current = null;
      connection.stop();
    };
  }, []);
  useEffect(() => {
    watched.current = runId;
    setWatch((s) => watchRun(s, runId));
    setIndex(null);
    socket.current?.send(JSON.stringify({ type: "watch", run_id: runId }));
  }, [runId]);

  const units = watch.checkpoints;
  const at = units.length === 0 ? null : Math.min(index ?? units.length - 1, units.length - 1);
  const unit = at === null ? null : units[at];

  // The card of the checkpoint shown (the per-car results of the replayed drive).
  const hasCard = unit !== null && watch.exams.some((e) => e.unit === unit);
  useEffect(() => {
    if (runId === null || unit === null || !hasCard) return;
    let live = true;
    const key = `${runId}/${unit}`;
    fetchExamCard(runId, unit)
      .then((answer) => live && setCard({ key, tracks: answer.card.tracks as CardTrack[] }))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, [runId, unit, hasCard]);
  const shownCard = card !== null && card.key === `${runId}/${unit}` ? card : null;
  const examTracks = shownCard?.tracks.map((t) => t.track) ?? [track];
  const part = shownCard?.tracks.find((t) => t.track === track) ?? null;

  /** Replay checkpoint `u` on `onTrack`; keeps playing if the live drive was playing. */
  function replay(u: number, onTrack = track) {
    if (runId === null) return;
    const wasPlaying = useLive.getState().frame?.playing ?? false;
    loadScenario({ track: onTrack, policy: "checkpoint", checkpoint: { run: runId, unit: u }, start_shift: shift });
    if (wasPlaying) send({ type: "play" }); // after the load (the server answers in order)
  }

  function moveTo(i: number) {
    setIndex(i);
    replay(units[i]);
  }

  const keys = plottableKeys(watch.records);
  const metric = defaultMetric(keys);
  const drawn = useMemo(() => (metric ? series(watch.records, metric) : { unit: [], value: [] }), [watch.records, metric]);
  const isReplayed =
    replaying !== null && replaying.run === runId && replaying.unit === unit && sameShift(replayedShift, shift);
  const expertShown = policy === "expert" && replayedShift !== null && sameShift(replayedShift, shift);
  const run = runs?.find((r) => r.run_id === runId) ?? null;

  return (
    <section className="view toylab" data-testid="timeline">
      <div className="view-header">
        <h2>Experiment: run timeline</h2>
        {headerExtra}
      </div>

      <div className="toylab-controls">
        <label className="picker">
          <span>Run </span>
          <select
            data-testid="timeline-run"
            value={runId ?? ""}
            onChange={(e) => setRunId(e.target.value)}
            disabled={!runs?.length}
          >
            {(runs ?? []).map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} ({r.checkpoints} checkpoints)
              </option>
            ))}
          </select>
        </label>
        <label className="picker">
          <span>Track </span>
          <select
            data-testid="timeline-track"
            value={track}
            onChange={(e) => {
              setTrack(e.target.value);
              if (unit !== null) replay(unit, e.target.value);
            }}
          >
            {examTracks.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <button type="button" onClick={loadRuns} data-testid="timeline-refresh">
          Refresh list
        </button>
      </div>

      <div className="toylab-controls" data-testid="timeline-shift">
        <label className="picker">
          <span>Start shift sideways (m, left +) </span>
          <input
            type="number"
            data-testid="timeline-shift-lateral"
            min={-5}
            max={5}
            step={0.5}
            value={shiftM}
            onChange={(e) => setShiftM(Number.isFinite(e.target.valueAsNumber) ? e.target.valueAsNumber : 0)}
          />
        </label>
        <label className="picker">
          <span>heading (°, left +) </span>
          <input
            type="number"
            data-testid="timeline-shift-heading"
            min={-90}
            max={90}
            step={5}
            value={shiftDeg}
            onChange={(e) => setShiftDeg(Number.isFinite(e.target.valueAsNumber) ? e.target.valueAsNumber : 0)}
          />
        </label>
        <button
          type="button"
          data-testid="timeline-shift-expert"
          disabled={shift === null || expertShown}
          onClick={() => loadScenario({ track, policy: "expert", start_shift: shift })}
        >
          {expertShown ? "The expert drives these starts" : "Watch the expert from these starts"}
        </button>
      </div>

      {error && <p className="toylab-error">Could not load: {error}</p>}
      {runs !== null && runs.length === 0 && (
        <p className="hint" data-testid="timeline-empty">
          No driving runs yet. Start one in the Training tab, or run{" "}
          <code>uv run python -m neurodrive train --config configs/random_search.yaml</code> and press Refresh list.
        </p>
      )}

      {unit !== null && at !== null && (
        <div className="toylab-slider">
          <label htmlFor="timeline-slider">Checkpoint</label>
          <input
            id="timeline-slider"
            data-testid="timeline-slider"
            type="range"
            min={0}
            max={units.length - 1}
            step={1}
            value={at}
            onChange={(e) => moveTo(Number(e.target.value))}
          />
          <span data-testid="timeline-label" data-unit={unit}>
            unit {unit} ({at + 1} of {units.length})
          </span>
          <button type="button" onClick={() => replay(unit)} data-testid="timeline-replay" disabled={isReplayed}>
            {isReplayed ? "Replaying in the Environment view" : "Replay this checkpoint"}
          </button>
        </div>
      )}
      {unit !== null && (
        <p className="hint">
          {isReplayed
            ? `The Environment view (page or drawer) drives this checkpoint's network from the exam's 10 starts${shift === null ? "" : `, each moved ${shiftM} m sideways and turned ${shiftDeg}°`}; press Play there. The Network view shows the network and what changed since the previous checkpoint.`
            : "Move the slider (or press Replay) to load this checkpoint into the Environment and Network views."}
        </p>
      )}

      {metric !== null && unit !== null && (
        <LiveChart
          unit={drawn.unit}
          value={drawn.value}
          metric={metric}
          budgetUnits={null}
          examUnits={units}
          cursor={unit}
          onPick={(u) => {
            const i = units.indexOf(u);
            if (i >= 0) moveTo(i);
          }}
        />
      )}

      {part !== null && unit !== null && shift !== null && (
        <p className="hint" data-testid="timeline-card-shifted">
          The exam card describes the exam's own starts, not shifted ones, so it is not shown. Set the shift to 0 m and 0°
          to see it; the Environment view's HUD shows how each shifted car ends.
        </p>
      )}
      {part !== null && unit !== null && shift === null && (
        <table className="compare-table" data-testid="timeline-card">
          <caption>
            Exam card of unit {unit} on {part.track} ({part.split}): what each car of the replay does. Click a row to
            select that car.
          </caption>
          <thead>
            <tr>
              <th scope="col">Car</th>
              <th scope="col">Start (m along, m left)</th>
              <th scope="col">Ends</th>
              <th scope="col">Progress</th>
              <th scope="col">After</th>
            </tr>
          </thead>
          <tbody>
            {part.episodes.map((e) => (
              <tr
                key={e.start_index}
                data-testid="timeline-card-row"
                data-status={e.status}
                className={isReplayed && selectedCar === e.start_index ? "timeline-row-selected" : undefined}
                onClick={() => isReplayed && selectCar(e.start_index)}
              >
                <td>{e.start_index + 1}</td>
                <td>
                  {formatNumber(e.start_s0_m)}, {formatNumber(e.start_lateral_m)}
                </td>
                <td>{e.status}</td>
                <td>{Math.round(e.progress_frac * 100)}% of 2 laps</td>
                <td>{formatNumber(e.time_s)} s</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {unit !== null && !hasCard && (
        <p className="hint">Unit {unit} has no exam card, so there are no per-car results to show.</p>
      )}
      {run && <p className="hint">Run {shortName(run.run_id)} · {run.status}</p>}
    </section>
  );
}
