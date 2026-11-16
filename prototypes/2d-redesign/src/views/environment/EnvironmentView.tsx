import { useEffect, useState } from "react";

import type { RunSummary } from "@prod/api/generated";
import { fetchRuns } from "@prod/api/rest";
import { type ScenarioRequest, useLive } from "@prod/state/live";
import { defaultGhost } from "@prod/views/environment/population";
import { useDrivingKeys } from "@prod/views/environment/useDrivingKeys";

import { POLICY_LABEL, type Policy, type World } from "../../source/model";
import { useWorld } from "../../source/world";
import { shortRun, STATUS, useApp } from "../../state/store";
import { Button, Chip, Field, Segmented, Select } from "../../ui/controls";
import { SourceWaiting } from "../../ui/SourceWaiting";
import { Minimap, RaceLegend, TrailStrip } from "./Extras";
import { Stage } from "./Stage";
import { Telemetry } from "./Telemetry";

const SPEEDS = [0.5, 1, 2, 4, 8];
const DEFAULT_SEED = 0;

const zoom = (factor: number) => window.dispatchEvent(new CustomEvent("nd-zoom", { detail: factor }));

/** Play, step, reset and the speed: the controls used most, floating at the top left of the stage. */
function Transport({ world }: { world: World }) {
  const { play, pause, stepOnce, reset, setSpeed } = useApp.getState();
  const { playing, speed, ended } = world;
  return (
    <div className="hudbar" role="toolbar" aria-label="Simulation">
      <Button
        variant="hud"
        icon={playing ? "pause" : "play"}
        onClick={playing ? pause : play}
        title={playing ? "Pause (Space)" : ended ? "The episode has ended: Reset to drive again" : "Play (Space)"}
        disabled={ended && !playing && world.source === "live"}
        testId="btn-play"
      >
        {playing ? "Pause" : "Play"}
      </Button>
      <Button variant="hud" icon="step" onClick={stepOnce} disabled={playing || ended} title="One step forward" testId="btn-step" />
      <Button variant="hud" icon="reset" onClick={reset} title="Back to the start" testId="btn-reset" />
      <span className="hudbar-sep" />
      <Segmented
        variant="hud"
        label="Speed"
        value={String(speed)}
        onChange={(id) => setSpeed(Number(id))}
        options={SPEEDS.map((s) => ({ id: String(s), label: `${s}×` }))}
      />
    </div>
  );
}

/** Camera and layer switches, floating at the top right of the stage. */
function CameraBar() {
  const mode = useApp((s) => s.cameraMode);
  const showRays = useApp((s) => s.showRays);
  const showTrail = useApp((s) => s.showTrail);
  const set = useApp((s) => s.set);
  return (
    <div className="hudbar" role="toolbar" aria-label="Camera">
      <Button variant="hud" icon="rays" pressed={showRays} onClick={() => set({ showRays: !showRays })} title="Sensor rays of the selected car" testId="toggle-rays" />
      <Button variant="hud" icon="trail" pressed={showTrail} onClick={() => set({ showTrail: !showTrail })} title="Decision trail of the selected car" testId="toggle-trail" />
      <span className="hudbar-sep" />
      <Segmented
        variant="hud"
        label="Camera"
        value={mode === "free" ? "none" : mode}
        onChange={(id) => set({ cameraMode: id as "follow" | "fit" })}
        options={[
          { id: "follow", label: "Follow", icon: "follow", hint: "Follow the selected car (double-click the track)", testId: "camera-follow" },
          { id: "fit", label: "Whole track", icon: "fit", hint: "Show the whole track", testId: "camera-fit" },
        ]}
      />
      {mode === "free" && <span className="hudbar-note">Free view</span>}
      <Button variant="hud" icon="minus" onClick={() => zoom(0.8)} title="Zoom out" testId="zoom-out" />
      <Button variant="hud" icon="plus" onClick={() => zoom(1.25)} title="Zoom in" testId="zoom-in" />
    </div>
  );
}

/**
 * A whole number typed in a small box, sent when the user presses Enter or leaves the box (as
 * production's IntField: no request per keystroke, out-of-range values clamped).
 */
function IntField({ value, min, max, onCommit, disabled, testId, wide }: { value: number; min: number; max: number; onCommit: (n: number) => void; disabled?: boolean; testId?: string; wide?: boolean }) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const n = Math.round(Number(text));
    if (!Number.isFinite(n)) return setText(String(value));
    const clamped = Math.max(min, Math.min(max, n));
    setText(String(clamped));
    if (clamped !== value) onCommit(clamped);
  };
  return (
    <input
      className={`input${wide ? " input-wide" : ""}`}
      inputMode="numeric"
      value={text}
      disabled={disabled}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          commit();
          (e.target as HTMLInputElement).blur(); // so the arrow keys drive the car again
        }
      }}
      data-testid={testId}
    />
  );
}

/**
 * The selected car, by name: with 100 cars a click on the track is not the only way to find one
 * (production: `select-car`). In a GA race the cars are listed best rank first.
 */
function CarSelect({ world }: { world: World }) {
  const selectCar = useApp((s) => s.selectCar);
  if (world.cars.length <= 1) return null;
  const info = world.scenario.population;
  const order = world.cars.map((_, i) => i);
  if (info !== null) order.sort((a, b) => (info.ranks[a] ?? info.size) - (info.ranks[b] ?? info.size));
  const label = (i: number) =>
    info === null ? `Car ${i + 1}` : i === info.ghost_car ? `Ghost · gen ${info.ghost_unit}` : `Car ${i + 1} · rank ${info.ranks[i] + 1}${info.elite.includes(i) ? " · elite" : ""}`;
  return (
    <Field label="Car">
      <Select value={world.selected} options={order.map((i) => ({ value: i, label: label(i) }))} onChange={selectCar} label="Selected car" testId="select-car" />
    </Field>
  );
}

/** The scenario of the recorded race: shown, but fixed (one recorded generation). */
function RecordedScenarioBar({ world }: { world: World }) {
  const info = world.scenario.population;
  const fixed = "Fixed with the recorded source: it replays one race. Switch the top bar to Live for every driver and track.";
  if (info === null) return null;
  return (
    <div className="scenario">
      <Field label="Track" hint={fixed}>
        <Select value={world.track.name} options={[{ value: world.track.name, label: world.track.name }]} onChange={() => {}} disabled />
      </Field>
      <Field label="Driver" hint={fixed}>
        <Select value="population" options={[{ value: "population", label: `GA generation · ${shortRun(info.run)}` }]} onChange={() => {}} disabled />
      </Field>
      <Field label="Generation" hint={fixed}>
        <input className="input" value={info.unit} readOnly disabled />
        <span className="muted">of {info.last_unit}</span>
      </Field>
      <Field label="Start" hint={fixed}>
        <input className="input" value={info.start + 1} readOnly disabled />
        <span className="muted">of {info.n_starts}</span>
      </Field>
      <Field label="Ghost gen" hint={fixed}>
        <input className="input" value={info.ghost_unit ?? 0} readOnly disabled />
      </Field>
      <CarSelect world={world} />
    </div>
  );
}

/** GA runs with at least one saved generation, newest first (the run list is newest first). */
function useGaRuns(): RunSummary[] {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  useEffect(() => {
    let live = true;
    fetchRuns()
      .then((list) => live && setRuns(list.runs.filter((r) => r.method === "ga" && r.checkpoints > 0)))
      .catch(() => live && setRuns([]));
    return () => {
      live = false;
    };
  }, []);
  return runs;
}

/**
 * The live scenario: every choice production's Controls offers, in one row. Each change is a
 * `load_scenario` built by production's own `loadScenario`, which keeps the other settings.
 */
function LiveScenarioBar({ world }: { world: World }) {
  const tracks = useLive((s) => s.tracks);
  const recording = useLive((s) => s.recording !== null);
  const loadScenario = useLive((s) => s.loadScenario);
  const gaRuns = useGaRuns();
  const { scenario } = world;
  const race = scenario.population;
  const replay = scenario.checkpoint;
  const policy = scenario.policy;

  const policies: { value: Policy; label: string }[] = (["keyboard", "constant", "neuron", "random_networks", "expert"] as Policy[]).map((p) => ({
    value: p,
    label: POLICY_LABEL[p],
  }));
  if (gaRuns.length > 0 || race !== null) policies.push({ value: "population", label: race ? `GA generation · ${shortRun(race.run)}` : "GA generation (a run of yours)" });
  if (replay !== null) policies.push({ value: "checkpoint", label: `Checkpoint · ${shortRun(replay.run)} unit ${replay.unit}` });

  const choosePolicy = (next: Policy) => {
    if (next === "population" && race === null) {
      const run = gaRuns[0];
      if (run === undefined) return;
      const unit = Math.max(1, run.last_unit ?? 1);
      loadScenario({ policy: "population", population: { run: run.run_id, unit, start: 0, ghost_unit: defaultGhost(unit) } });
      return;
    }
    loadScenario({ policy: next as ScenarioRequest["policy"] });
  };
  const raceWith = (changes: Partial<{ run: string; unit: number; start: number; ghost_unit: number | null }>) => {
    if (race === null) return;
    const next = { run: race.run, unit: race.unit, start: race.start, ghost_unit: race.ghost_unit, ...changes };
    loadScenario({ policy: "population", population: next });
  };

  return (
    <div className="scenario" data-testid="scenario-bar">
      <Field label="Track" hint={race !== null ? "A GA generation races on the run's own fitness track" : undefined}>
        <Select
          value={scenario.track}
          options={tracks.map((t) => ({ value: t.name, label: t.split === "heldout" ? `${t.name} · held out` : t.name }))}
          onChange={(name) => loadScenario({ track: name })}
          disabled={race !== null || recording}
          label="Track"
          testId="select-track"
        />
      </Field>
      <Field label="Driver">
        <Select value={policy} options={policies} onChange={choosePolicy} disabled={recording} label="Driver" testId="select-policy" />
      </Field>
      <CarSelect world={world} />

      {race !== null ? (
        <>
          <Field label="Run">
            <Select
              value={race.run}
              options={(gaRuns.some((r) => r.run_id === race.run) ? gaRuns : [{ run_id: race.run } as RunSummary, ...gaRuns]).map((r) => ({
                value: r.run_id,
                label: shortRun(r.run_id),
              }))}
              onChange={(run) => {
                const unit = Math.max(1, gaRuns.find((r) => r.run_id === run)?.last_unit ?? 1);
                raceWith({ run, unit, start: 0, ghost_unit: defaultGhost(unit) });
              }}
              label="Run"
            />
          </Field>
          <Field label="Generation">
            <IntField value={race.unit} min={1} max={race.last_unit} onCommit={(unit) => raceWith({ unit, ghost_unit: race.ghost_unit === null ? null : defaultGhost(unit) })} testId="race-generation" />
            <span className="muted">of {race.last_unit}</span>
          </Field>
          <Field label="Start">
            <IntField value={race.start + 1} min={1} max={race.n_starts} onCommit={(n) => raceWith({ start: n - 1 })} testId="race-start" />
            <span className="muted">of {race.n_starts}</span>
          </Field>
          <Field label="Ghost gen" hint="An earlier generation's champion, see-through, racing alongside; 0 for none">
            <IntField
              value={race.ghost_unit ?? 0}
              min={0}
              max={Math.max(0, race.unit - 1)}
              onCommit={(n) => raceWith({ ghost_unit: n === 0 ? null : n })}
              testId="race-ghost"
            />
          </Field>
        </>
      ) : replay !== null ? (
        <span className="muted">The exam&apos;s 10 starts; pick another checkpoint in Experiment → Timeline.</span>
      ) : (
        <>
          {policy !== "keyboard" && policy !== "neuron" && (
            <Field label="Cars">
              <IntField value={scenario.n_cars} min={1} max={100} onCommit={(n) => loadScenario({ n_cars: n })} testId="input-cars" />
            </Field>
          )}
          <label className="scenario-check">
            <input
              type="checkbox"
              checked={scenario.seed !== null}
              onChange={(e) => loadScenario({ seed: e.target.checked ? DEFAULT_SEED : null })}
              data-testid="check-random-start"
            />
            Random start
          </label>
          {scenario.seed !== null && (
            <Field label="Seed">
              <IntField value={scenario.seed} min={0} max={999999} onCommit={(n) => loadScenario({ seed: n })} testId="input-seed" wide />
            </Field>
          )}
        </>
      )}
    </div>
  );
}

/** Drive the car with the arrow keys or W A S D (production's own key handling). */
function KeyboardDriving() {
  useDrivingKeys();
  return null;
}

/**
 * Recording keyboard demonstrations (task 6.1b): a second floating bar under the transport,
 * shown for the keyboard driver only (PRODUCTION_MAPPING.md §3.2).
 */
function RecordBar() {
  const recording = useLive((s) => s.recording);
  const saved = useLive((s) => s.savedDataset);
  const send = useLive((s) => s.send);
  const [tag, setTag] = useState("");
  const cleanTag = tag.trim().replace(/[^A-Za-z0-9_-]+/g, "-");
  return (
    <div className="hudbar recbar" role="toolbar" aria-label="Record demonstrations">
      {recording === null ? (
        <>
          <Button variant="hud" icon="dot" onClick={() => send({ type: "record_start" })} title="Record your driving as a demonstration dataset" testId="btn-record">
            Record
          </Button>
          {saved !== null && (
            <span className="recbar-note" data-testid="record-saved">
              Saved {saved.dataset_id} · {saved.frames} frames
            </span>
          )}
        </>
      ) : (
        <>
          <span className="rec-badge" data-testid="rec-badge">
            REC <span className="num">{recording.frames}</span> frames · {recording.episodes} drive{recording.episodes === 1 ? "" : "s"}
          </span>
          <input className="hud-input" placeholder="tag" value={tag} onChange={(e) => setTag(e.target.value)} aria-label="Dataset tag" data-testid="record-tag" />
          <Button
            variant="hud"
            icon="check"
            disabled={cleanTag === "" || recording.frames === 0}
            onClick={() => send({ type: "record_stop", save: true, tag: cleanTag })}
            title={cleanTag === "" ? "Give the dataset a tag first" : `Save as …-keyboard-${cleanTag}`}
            testId="btn-record-save"
          >
            Save
          </Button>
          <Button variant="hud" icon="cross" onClick={() => send({ type: "record_stop", save: false, tag: "" })} title="Forget this recording" testId="btn-record-discard">
            Discard
          </Button>
        </>
      )}
    </div>
  );
}

/** Why the selected car stopped, in words (production: the status banner); or the server's error. */
function Banners({ world }: { world: World }) {
  const error = useLive((s) => s.error);
  const pose = world.shown?.pose;
  const status = pose === undefined ? null : STATUS[pose.status];
  const reason =
    status === null || status.key === "running"
      ? null
      : {
          crashed: "left the road",
          finished: `completed ${world.scenario.target_laps === 1 ? "its lap" : `all ${world.scenario.target_laps} laps`}`,
          stalled: "made no progress for 3 s",
          timeout: `ran out of time before finishing ${world.scenario.target_laps} laps`,
        }[status.key];
  return (
    <>
      {world.source === "live" && error !== null && (
        <div className="endbanner" data-kind="bad" role="alert" data-testid="server-error">
          <Chip kind="bad">Server</Chip>
          <span>{error}</span>
        </div>
      )}
      {status !== null && reason !== null && (
        <div className="endbanner" data-kind={status.kind} role="status" data-testid="status-banner">
          <Chip kind={status.kind}>{status.label}</Chip>
          <span>
            {world.scenario.policy === "keyboard" ? "You" : `Car ${world.selected + 1}`} {reason}.
          </span>
        </div>
      )}
    </>
  );
}

/**
 * The Environment view. Full version: scenario bar, the stage with its floating controls,
 * telemetry and legend, and the look-back timeline. Compact version (a drawer): the stage alone,
 * following the selected car, with a one-line read-out.
 */
export function EnvironmentView({ compact = false }: { compact?: boolean }) {
  const world = useWorld();
  const keyboard = world?.source === "live" && world.scenario.policy === "keyboard";

  if (compact) {
    const pose = world?.shown?.pose ?? null;
    const status = STATUS[pose?.status ?? 0];
    return (
      <div className="env-compact" data-testid="environment-compact">
        {world !== null ? <Stage compact /> : <SourceWaiting what="the track" />}
        {pose !== null && (
          <div className="env-compact-hud">
            <span className="num">{(pose.v * 3.6).toFixed(0)} km/h</span>
            <Chip kind={status.kind}>{status.label}</Chip>
          </div>
        )}
      </div>
    );
  }

  return (
    <section className="view env-view" data-testid="environment-view">
      {keyboard && <KeyboardDriving />}
      <header className="view-head">
        <div className="view-title">
          <h2>Environment</h2>
          <span className="view-question">What does the car do?</span>
        </div>
        {world !== null && (world.source === "live" ? <LiveScenarioBar world={world} /> : <RecordedScenarioBar world={world} />)}
      </header>
      <div className="env-body">
        <div className="stage">
          {world !== null ? <Stage compact={false} /> : <SourceWaiting what="a track and its first frame" />}
          {world !== null && (
            <>
              <div className="stage-top">
                <div className="stage-top-left">
                  <Transport world={world} />
                  {keyboard && <RecordBar />}
                </div>
                <div className="stage-top-right">
                  <CameraBar />
                  <Minimap />
                </div>
              </div>
              <div className="stage-bottom">
                <Banners world={world} />
                <RaceLegend />
                <Telemetry />
              </div>
            </>
          )}
        </div>
        {world !== null && <TrailStrip />}
      </div>
    </section>
  );
}
