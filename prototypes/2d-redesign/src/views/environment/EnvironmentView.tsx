import { STATUS, useApp, usePose } from "../../state/store";
import { Button, Chip, Field, Segmented, Select } from "../../ui/controls";
import { Minimap, RaceLegend, TrailStrip } from "./Extras";
import { Stage } from "./Stage";
import { Telemetry } from "./Telemetry";

const SPEEDS = [0.5, 1, 2, 4, 8];

const zoom = (factor: number) => window.dispatchEvent(new CustomEvent("nd-zoom", { detail: factor }));

/** Play, step, reset and the speed: the controls used most, floating at the top left of the stage. */
function Transport() {
  const playing = useApp((s) => s.playing);
  const speed = useApp((s) => s.speed);
  const ended = useApp((s) => s.data !== null && s.t >= s.data.race.frames.step.length - 1);
  const { play, pause, stepOnce, reset, set } = useApp.getState();
  return (
    <div className="hudbar" role="toolbar" aria-label="Simulation">
      <Button
        variant="hud"
        icon={playing ? "pause" : "play"}
        onClick={playing ? pause : play}
        title={playing ? "Pause (Space)" : ended ? "Play again from the start" : "Play (Space)"}
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
        onChange={(id) => set({ speed: Number(id) })}
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
          { id: "follow", label: "Follow", icon: "follow", hint: "Follow the selected car (double-click the track)" },
          { id: "fit", label: "Whole track", icon: "fit", hint: "Show the whole track" },
        ]}
      />
      {mode === "free" && <span className="hudbar-note">Free view</span>}
      <Button variant="hud" icon="minus" onClick={() => zoom(0.8)} title="Zoom out" testId="zoom-out" />
      <Button variant="hud" icon="plus" onClick={() => zoom(1.25)} title="Zoom in" testId="zoom-in" />
    </div>
  );
}

/** The scenario: what is on the track. Fixed in the prototype (one recorded race). */
function ScenarioBar() {
  const data = useApp((s) => s.data);
  if (data === null) return null;
  const info = data.race.scenario.population;
  const fixed = "Fixed in the prototype: it replays one recorded race";
  return (
    <div className="scenario">
      <Field label="Track" hint={fixed}>
        <Select value={data.track.name} options={[{ value: data.track.name, label: data.track.name }]} onChange={() => {}} disabled />
      </Field>
      <Field label="Driver" hint={fixed}>
        <Select
          value="population"
          options={[{ value: "population", label: `GA generation · ${info.run.replace(/^\d{8}-\d{6}-/, "")}` }]}
          onChange={() => {}}
          disabled
        />
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
    </div>
  );
}

/** Why the selected car stopped, in words (production: the status banner). */
function EndBanner() {
  const pose = usePose();
  const car = useApp((s) => s.car);
  if (pose === null || pose.status === 0) return null;
  const status = STATUS[pose.status];
  const reason = {
    crashed: "left the road",
    finished: "completed both laps",
    stalled: "made no progress for 3 s",
    timeout: "ran out of time before finishing 2 laps",
  }[status.key as "crashed"];
  return (
    <div className="endbanner" data-kind={status.kind} role="status" data-testid="status-banner">
      <Chip kind={status.kind}>{status.label}</Chip>
      <span>
        Car {car + 1} {reason}.
      </span>
    </div>
  );
}

/**
 * The Environment view. Full version: scenario bar, the stage with its floating controls,
 * telemetry and legend, and the look-back timeline. Compact version (a drawer): the stage alone,
 * following the selected car, with a one-line read-out.
 */
export function EnvironmentView({ compact = false }: { compact?: boolean }) {
  const pose = usePose();
  const ready = useApp((s) => s.data !== null);

  if (compact) {
    const status = STATUS[pose?.status ?? 0];
    return (
      <div className="env-compact" data-testid="environment-compact">
        {ready ? <Stage compact /> : <div className="skeleton" style={{ height: "100%" }} />}
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
      <header className="view-head">
        <div className="view-title">
          <h2>Environment</h2>
          <span className="view-question">What does the car do?</span>
        </div>
        <ScenarioBar />
      </header>
      <div className="env-body">
        <div className="stage">
          {ready ? <Stage compact={false} /> : <div className="skeleton" style={{ position: "absolute", inset: 0 }} />}
          <div className="stage-top">
            <Transport />
            <div className="stage-top-right">
              <CameraBar />
              <Minimap />
            </div>
          </div>
          <div className="stage-bottom">
            <EndBanner />
            <RaceLegend />
            <Telemetry />
          </div>
        </div>
        <TrailStrip />
      </div>
    </section>
  );
}
