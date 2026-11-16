// The live adapter: the world snapshot built from the production app's own stores, which the
// production API client fills from the Python server (web/src/api/useLiveConnection.ts). Nothing
// here computes anything about driving or networks: it picks fields out of the server's messages,
// sends the server's commands, and fetches two REST answers (lineage and checkpoint change) the
// way production's NetworkView does.
//
// Imports from `@prod` are read-only uses of web/src/. They are the reason the prototype obeys
// the same linking rules as the app (ARCHITECTURE.md §8.2): who follows whose selected car, when
// a what-if or a look back is dropped, how a reconnect restores the scenario.

import { create } from "zustand";

import type { FrameMessage } from "@prod/api/generated";
import { fetchCheckpointDiff, fetchCheckpoints, fetchMutation } from "@prod/api/rest";
import { type Command, type LiveState, useLive } from "@prod/state/live";
import { activeMoment, usePlayback } from "@prod/state/playback";
import { useSelection } from "@prod/state/selection";

import { attachBridge, useApp } from "../state/store";
import type { CarDetail, CarPose, Origin, World } from "./model";

const NO_ORIGIN: Origin = { mutation: null, diff: null, diffFrom: null, note: null };

// ── Lineage and checkpoint change of the selected network (REST) ─────────────────────────

/**
 * The selected network's origin: in a GA race, what made it differ from its parents (task 7.8);
 * in a checkpoint replay, what changed since the previous checkpoint (task 5.9). Production
 * fetches the same two answers inside NetworkView.tsx (`useMutation`, `useCheckpointChange`).
 */
export const useLiveOrigin = create<{ key: string; origin: Origin }>(() => ({ key: "", origin: NO_ORIGIN }));

function originKey(live: LiveState): string {
  const scenario = live.scenario;
  const car = live.frame?.selected?.index ?? null;
  if (scenario?.population) {
    const p = scenario.population;
    return `pop/${p.run}/${p.unit}/${car}`;
  }
  if (scenario?.checkpoint) return `ckpt/${scenario.checkpoint.run}/${scenario.checkpoint.unit}`;
  return "";
}

function refreshOrigin(live: LiveState): void {
  const key = originKey(live);
  if (key === useLiveOrigin.getState().key) return;
  useLiveOrigin.setState({ key, origin: NO_ORIGIN });
  const done = (origin: Origin) => {
    if (useLiveOrigin.getState().key === key) useLiveOrigin.setState({ origin });
  };
  const scenario = live.scenario;
  const car = live.frame?.selected?.index ?? null;
  if (scenario?.population && car !== null) {
    const p = scenario.population;
    if (car >= p.size) return done({ ...NO_ORIGIN, note: `the champion of generation ${p.ghost_unit}, racing as a ghost` });
    if (p.unit < 2) return done({ ...NO_ORIGIN, note: "a random network of the first generation; it has no parent" });
    fetchMutation(p.run, p.unit, car).then(
      (mutation) => done({ ...NO_ORIGIN, mutation }),
      (error: Error) => done({ ...NO_ORIGIN, note: `no lineage: ${error.message}` }),
    );
    return;
  }
  if (scenario?.checkpoint) {
    const { run, unit } = scenario.checkpoint;
    fetchCheckpoints(run)
      .then(async (list) => {
        const earlier = list.units.filter((u) => u < unit);
        const from = earlier.length === 0 ? null : Math.max(...earlier);
        const diff = from === null ? null : await fetchCheckpointDiff(run, unit, from);
        done({ ...NO_ORIGIN, diff, diffFrom: from, note: from === null ? "the run's first checkpoint" : null });
      })
      .catch((error: Error) => done({ ...NO_ORIGIN, note: error.message }));
  }
}

// ── The snapshot ─────────────────────────────────────────────────────────────────────────

const posesCache = new WeakMap<FrameMessage["cars"], CarPose[]>();

/** The frame's columns as one pose per car. */
function posesOf(cars: FrameMessage["cars"]): CarPose[] {
  let poses = posesCache.get(cars);
  if (poses === undefined) {
    poses = cars.x.map((x, i) => ({
      x,
      y: cars.y[i],
      psi: cars.psi[i],
      v: cars.v[i],
      progress: cars.progress[i],
      steer: cars.action[i]?.[0] ?? 0,
      throttle: cars.action[i]?.[1] ?? 0,
      status: cars.status[i],
    }));
    posesCache.set(cars, poses);
  }
  return poses;
}

export interface LiveInputs {
  live: LiveState;
  viewStep: number | null;
  overrides: Record<number, number>;
  origin: Origin;
}

export function buildLive({ live, viewStep, overrides, origin }: LiveInputs): World | null {
  const { scenario, geometry, frame } = live;
  if (scenario === null || geometry === null || frame === null || geometry.name !== scenario.track) return null;
  const cars = posesOf(frame.cars);
  const selectedNow = frame.selected;
  const selected = selectedNow?.index ?? 0;
  const moment = activeMoment(live, viewStep);

  let shown: CarDetail | null = null;
  if (moment !== null) {
    shown = {
      index: moment.car,
      step: moment.step,
      pose: {
        x: moment.x,
        y: moment.y,
        psi: moment.psi,
        v: moment.v,
        progress: moment.progress,
        steer: moment.action[0] ?? 0,
        throttle: moment.action[1] ?? 0,
        status: moment.status,
      },
      rays: moment.rays,
      observation: moment.observation,
      action: moment.action,
      trace: moment.trace,
      explain: moment.explain,
      neuron: null,
    };
  } else if (selectedNow) {
    shown = {
      index: selectedNow.index,
      step: frame.step,
      pose: cars[selectedNow.index],
      rays: selectedNow.rays,
      observation: selectedNow.observation,
      action: frame.cars.action[selectedNow.index] ?? [0, 0],
      trace: selectedNow.trace,
      explain: selectedNow.explain,
      neuron: selectedNow.neuron,
    };
  }

  // A what-if answer counts only for the car and step on screen, and only while inputs are changed.
  const result = live.whatIfResult;
  const asking = Object.keys(overrides).length > 0;
  const whatIf =
    asking && result !== null && result.car === selected && result.step === frame.step
      ? {
          base: result.base_outputs,
          tried: result.trace.activations[result.trace.activations.length - 1] ?? [],
          predicted: result.predicted_outputs,
          ghost: {
            real: result.ghost.real,
            whatIf: result.ghost.what_if,
            end: [result.ghost.what_if_end[0], result.ghost.what_if_end[1], result.ghost.what_if_end[2]] as [number, number, number],
          },
        }
      : null;

  const trail = live.trail !== null && live.trail.car === selected ? live.trail : null;
  const network = live.network !== null && live.network.car === selected ? live.network : null;

  return {
    source: "live",
    track: geometry,
    scenario,
    step: frame.step,
    time: frame.time,
    playing: frame.playing,
    speed: frame.speed,
    ended: live.episodeEnd !== null,
    cars,
    selected,
    lookStep: moment === null ? null : moment.step,
    shown,
    trail,
    whatIf,
    network,
    origin,
    flow: null,
    endFitness: live.episodeEnd?.fitness ?? null,
  };
}

// ── Commands and the selection mirror ───────────────────────────────────────────────────

/**
 * Hand the prototype's actions to the production stores and keep the prototype's selection
 * fields a copy of `useSelection`. Returns a function that undoes it.
 */
export function startLive(): () => void {
  const send = (command: Command) => useLive.getState().send(command);
  attachBridge({
    selectCar: (index) => useSelection.getState().selectCar(index),
    toggleNeuron: (ref, layers) => useSelection.getState().toggleNeuron(ref, layers),
    toggleConnection: (ref) => useSelection.getState().toggleConnection(ref),
    setWhatIf: (k, value) => {
      const next = { ...useSelection.getState().whatIf, [k]: value };
      useSelection.getState().setWhatIf(k, value);
      // Only the changed inputs travel; the server keeps the car's exact values for the rest.
      send({ type: "what_if", overrides: next });
    },
    clearWhatIf: () => useSelection.getState().clearWhatIf(),
    play: () => send({ type: "play" }),
    pause: () => send({ type: "pause" }),
    stepOnce: () => send({ type: "step" }),
    reset: () => send({ type: "reset" }),
    setSpeed: (speed) => send({ type: "set_speed", speed }),
    lookAt: (step) => (step === null ? usePlayback.getState().backToLive() : usePlayback.getState().view(step)),
  });

  const mirror = () => {
    const s = useSelection.getState();
    useApp.setState({ car: s.carId ?? 0, neuron: s.neuron, connection: s.connection, output: s.output, whatIf: s.whatIf });
  };
  mirror();
  const stopSelection = useSelection.subscribe(mirror);
  const stopLive = useLive.subscribe((live, previous) => {
    if (live.status !== previous.status) useApp.setState({ liveStatus: live.status });
    refreshOrigin(live);
  });
  useApp.setState({ liveStatus: useLive.getState().status });

  return () => {
    stopSelection();
    stopLive();
    attachBridge(null);
  };
}
