// The shapes of the recorded files in public/recorded/. They mirror the production messages
// (web/src/api/generated.ts) field for field, so a component written against them maps directly
// onto the real `useLive` store and REST helpers. Only the containers (Race, CarRecord, RunData)
// are prototype inventions: production streams frames, the prototype has them all on disk.

export interface Track {
  name: string;
  split: string;
  width: number;
  length: number;
  start_s0: number;
  centerline: number[][];
  left: number[][];
  right: number[][];
}

/** `ScenarioMessage.population` in production. */
export interface PopulationInfo {
  run: string;
  unit: number;
  last_unit: number;
  start: number;
  n_starts: number;
  start_s0_m: number;
  start_lateral_m: number;
  size: number;
  fitness: number[];
  ranks: number[];
  finished: number[];
  crashed: number[];
  elite: number[];
  champion: number;
  ghost_unit: number | null;
  ghost_car: number | null;
  ghost_fitness: number | null;
}

export interface Scenario {
  track: string;
  policy: string;
  n_cars: number;
  population: PopulationInfo;
  dt: number;
  time_limit: number;
  target_laps: number;
  v_max: number;
  wheelbase: number;
  car_half_width: number;
  ray_angles: number[];
  ray_range: number;
  trail_seconds: number;
}

/** Every recorded frame, as columns: `x[f][car]`. Status: 0 running, 1 crashed, 2 finished, 3 stalled, 4 timeout. */
export interface RaceFrames {
  step: number[];
  time: number[];
  x: number[][];
  y: number[][];
  psi: number[][];
  v: number[][];
  progress: number[][];
  steer: number[][];
  throttle: number[][];
  status: number[][];
}

export interface Race {
  scenario: Scenario;
  stepsPerFrame: number;
  frames: RaceFrames;
  end: { status: string[]; progress: number[]; laps: number[]; fitness: number[] | null; time: number; step: number };
  featured: number[];
}

/** `NetworkMessage` in production. `weights[layer][to][from]`. */
export interface Network {
  car: number;
  layer_sizes: number[];
  activations: string[];
  weights: number[][][];
  biases: number[][];
}

export interface Trace {
  activations: number[][];
  weighted_sums: number[][];
}

export interface Receipt {
  output: number;
  inputs: number[];
  contributions: number[];
  bias: number;
  z: number;
  value: number;
}

export interface Explain {
  receipts: Receipt[];
  sensitivity: number[][];
}

export interface CarFrame {
  step: number;
  action: number[];
  rays: number[];
  observation: number[];
  trace: Trace;
  explain: Explain;
}

/** `MutationView` in production (GET .../generations/{g}/rows/{r}/mutation). */
export interface Mutation {
  origin: "random" | "elite" | "child";
  parents: number[];
  parents_fitness: number[];
  parents_rank: number[];
  crossed: boolean | null;
  main_parent: number;
  other_parent: number | null;
  fitness: number;
  rank: number;
  n_genes: number;
  n_mutated: number;
  n_from_other: number;
  max_abs_delta: number;
  weight_change: number[][][];
  weight_mutated: boolean[][][];
  weight_from_other: boolean[][][];
  bias_mutated: boolean[][];
}

export interface CarRecord {
  car: number;
  network: Network;
  mutation: Mutation | null;
  frames: CarFrame[];
}

/** `EvolutionView` in production (GET runs/{id}/evolution). */
export interface Evolution {
  population: number;
  elite: number;
  unit: number[];
  best_so_far: number[];
  best: number[];
  q3: number[];
  median: number[];
  mean: number[];
  q1: number[];
  worst: number[];
  zero: number[];
  diversity: number[];
  finished_share: number[];
  finishers: number[];
  champion: number[];
  band_width: number;
  edges: number[];
  counts: number[][];
  start: number | null;
  n_starts: number;
  start_finish_share: number[][];
  start_mean: number[][];
}

/** `GenerationStarts` in production (GET runs/{id}/generations/{g}/starts). */
export interface GenerationStarts {
  unit: number;
  population: number;
  n_starts: number;
  elite: number[];
  champion: number;
  start_fraction: number[];
  start_lateral_m: number[];
  fitness: number[];
  order: number[];
  start_fitness: number[][];
  status: string[][];
  progress: number[][];
  finish_time: (number | null)[][];
  finish_share: number[];
  start_mean: number[];
}

export interface AncestryStep {
  unit: number;
  row: number;
  fitness: number;
  rank: number;
  origin: "random" | "elite" | "child";
  parents: number[];
  crossed: boolean | null;
  main_parent: number;
  from_main: number;
  from_other: number;
  mutated: number;
  distance: number;
}

export interface ExamAggregate {
  episodes: number;
  completion_rate: number;
  crash_rate: number;
  progress_frac_median: number;
  progress_frac_q1: number;
  progress_frac_q3: number;
  median_best_lap_s?: number | null;
  median_smoothness: number;
}

export interface ExamEpisode {
  start_index: number;
  start_s0_m: number;
  start_lateral_m: number;
  status: string;
  time_s: number;
  progress_frac: number;
  laps: number;
  best_lap_s: number | null;
  mean_speed: number;
}

export interface ExamCard {
  tracks: { track: string; split: string; length_m: number; time_limit_s: number; episodes: ExamEpisode[]; aggregate: ExamAggregate }[];
  training: ExamAggregate;
  heldout: ExamAggregate;
  env_steps: number;
}

export interface CheckpointDiff {
  unit: number;
  from_unit: number;
  max_abs_delta: number;
  weight_change: number[][][];
}

export interface RunData {
  run_id: string;
  config: {
    method: string;
    seed: number;
    tag: string;
    network: { layer_sizes: number[]; hidden_activation: string; output_activation: string };
    population: { size: number; elite: number };
    selection: { method: string; tournament_size: number };
    variation: { crossover_rate: number; mutation_rate: number; mutation_strength: number };
    fitness: { track: string };
  };
  meta: { status: string; started_at: string; ended_at: string; git_commit: string };
  checkpoints: number[];
  generation: number;
  start: number;
  evolution: Evolution;
  evolutionStart: Evolution;
  starts: GenerationStarts;
  ancestry: { unit: number; row: number; n_genes: number; steps: AncestryStep[] };
  exam: ExamCard;
  examFinal: ExamCard;
  diff: CheckpointDiff | null;
}

/** `CompareResponse` in production (GET runs/compare). */
export interface CompareGroup {
  group_id: string;
  run_ids: string[];
  unit: number[];
  median: (number | null)[];
  q1: (number | null)[];
  q3: (number | null)[];
  final_median: number | null;
  final_q1: number | null;
  final_q3: number | null;
  n_diverged: number;
}

export interface CompareRun {
  run_id: string;
  group_id: string;
  seed: number | null;
  unit: number[];
  values: (number | null)[];
  final: number | null;
}

export interface CompareResponse {
  metric: string;
  groups: CompareGroup[];
  runs: CompareRun[];
}

export interface ExamGroup {
  key: string;
  label: string;
  seeds: number;
  cards: { run: string; training: ExamAggregate; heldout: ExamAggregate }[];
}

export interface Recorded {
  track: Track;
  race: Race;
  cars: Map<number, CarRecord>;
  run: RunData;
  compare: { optimizers: CompareResponse; imitation: CompareResponse };
  exams: ExamGroup[];
}
