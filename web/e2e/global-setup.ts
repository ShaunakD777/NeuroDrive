import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// Before the browser tests: make sure the toy lab has runs to show (task 3.8). Two small toy runs
// tagged "e2e" are trained with `nd train` (task 4.6b) into the repository's runs/ folder
// (git-ignored) unless completed ones are already there; training is seeded, so an existing run
// holds exactly the same numbers.

const repoRoot = fileURLToPath(new URL("../../", import.meta.url));
const runsDir = join(repoRoot, "runs");

export const E2E_RUNS = [
  { task: "sine", args: ["--epochs", "300"] },
  { task: "xor", args: ["--epochs", "1000"] },
];

function hasCompletedRun(task: string): boolean {
  if (!existsSync(runsDir)) return false;
  return readdirSync(runsDir)
    .filter((name) => name.endsWith(`-toy_${task}-e2e`))
    .some((name) => {
      const meta = join(runsDir, name, "meta.json");
      return existsSync(meta) && JSON.parse(readFileSync(meta, "utf8")).status === "completed";
    });
}

// Two 5-seed sweeps for the "Compare runs" tab (task 4.7b): sine for 100 epochs (tag "e2esweep"),
// and the validation example config for 50 epochs (tag "e2eval"), whose runs have a validation loss.
export const E2E_SWEEPS = [
  { config: "configs/toy_sine.yaml", tag: "e2esweep", seeds: 5, epochs: 100 },
  { config: "configs/toy_sine_val.yaml", tag: "e2eval", seeds: 5, epochs: 50 },
];

function hasCompletedSweep(tag: string, seeds: number): boolean {
  if (!existsSync(runsDir)) return false;
  const counts = new Map<string, number>();
  for (const name of readdirSync(runsDir)) {
    if (!new RegExp(`-toy_sine-${tag}-s\\d+$`).test(name)) continue;
    const meta = join(runsDir, name, "meta.json");
    if (!existsSync(meta)) continue;
    const m = JSON.parse(readFileSync(meta, "utf8"));
    if (m.status === "completed" && m.sweep) counts.set(m.sweep, (counts.get(m.sweep) ?? 0) + 1);
  }
  return [...counts.values()].some((n) => n === seeds);
}

// A small random-search run for the run timeline (task 5.9): 3 units, exams on, so it has several
// examined checkpoints (0, 1, 2, 3 and any unit with a new best).
export const E2E_SEARCH_TAG = "e2e";

function hasCompletedSearch(): boolean {
  if (!existsSync(runsDir)) return false;
  return readdirSync(runsDir)
    .filter((name) => name.endsWith(`-random_search-${E2E_SEARCH_TAG}`))
    .some((name) => {
      const meta = join(runsDir, name, "meta.json");
      return existsSync(meta) && JSON.parse(readFileSync(meta, "utf8")).status === "completed";
    });
}

// An expert demonstration dataset for the dataset inspector (task 6.2): the exam's 10 starts on
// the training loop, no random starts (tag "e2e"), recorded into data/demos/ (git-ignored) with
// `nd demo-record` unless one is there. The expert is deterministic, so it is always the same.
// A second one for the comparator (6.2 follow-up): the same 10 starts plus 2 seeded random starts
// (tag "e2ecmp"), so the two datasets share the track and most of their bins.
export const E2E_DATASET_TAG = "e2e";
export const E2E_COMPARE_TAG = "e2ecmp";
const demosDir = join(repoRoot, "data", "demos");

function hasDataset(tag: string): boolean {
  if (!existsSync(demosDir)) return false;
  return readdirSync(demosDir).some((name) => name.endsWith(`-expert-${tag}`) && existsSync(join(demosDir, name, "meta.json")));
}

export default function globalSetup(): void {
  for (const [tag, randomStarts] of [
    [E2E_DATASET_TAG, "0"],
    [E2E_COMPARE_TAG, "2"],
  ]) {
    if (hasDataset(tag)) continue;
    execFileSync(
      "uv",
      ["run", "python", "-m", "neurodrive", "demo-record", "--teacher", "expert", "--random-starts", randomStarts, "--seed", "1", "--tag", tag],
      { cwd: repoRoot, stdio: "inherit" },
    );
  }
  if (!hasCompletedSearch()) {
    execFileSync(
      "uv",
      ["run", "python", "-m", "neurodrive", "train", "--config", "configs/random_search.yaml", "--units", "3", "--tag", E2E_SEARCH_TAG],
      { cwd: repoRoot, stdio: "inherit" },
    );
  }
  for (const { task, args } of E2E_RUNS) {
    if (hasCompletedRun(task)) continue;
    execFileSync(
      "uv",
      ["run", "python", "-m", "neurodrive", "train", "--config", `configs/toy_${task}.yaml`, "--tag", "e2e", ...args],
      { cwd: repoRoot, stdio: "inherit" },
    );
  }
  for (const { config, tag, seeds, epochs } of E2E_SWEEPS) {
    if (hasCompletedSweep(tag, seeds)) continue;
    execFileSync(
      "uv",
      ["run", "python", "-m", "neurodrive", "sweep", "--config", config, "--seeds", `0-${seeds - 1}`, "--tag", tag, "--epochs", String(epochs)],
      { cwd: repoRoot, stdio: "inherit" },
    );
  }
}
