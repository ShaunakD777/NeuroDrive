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

// A 5-seed sweep for the "Compare runs" tab (task 4.7b): sine for 100 epochs, tag "e2esweep".
export const E2E_SWEEP = { tag: "e2esweep", seeds: 5, epochs: 100 };

function hasCompletedSweep(): boolean {
  if (!existsSync(runsDir)) return false;
  const counts = new Map<string, number>();
  for (const name of readdirSync(runsDir)) {
    if (!/-toy_sine-e2esweep-s\d+$/.test(name)) continue;
    const meta = join(runsDir, name, "meta.json");
    if (!existsSync(meta)) continue;
    const m = JSON.parse(readFileSync(meta, "utf8"));
    if (m.status === "completed" && m.sweep)
      counts.set(m.sweep, (counts.get(m.sweep) ?? 0) + 1);
  }
  return [...counts.values()].some((n) => n === E2E_SWEEP.seeds);
}

export default function globalSetup(): void {
  for (const { task, args } of E2E_RUNS) {
    if (hasCompletedRun(task)) continue;
    execFileSync(
      "uv",
      ["run", "python", "-m", "neurodrive", "train", "--config", `configs/toy_${task}.yaml`, "--tag", "e2e", ...args],
      { cwd: repoRoot, stdio: "inherit" },
    );
  }
  if (!hasCompletedSweep()) {
    const { tag, seeds, epochs } = E2E_SWEEP;
    execFileSync(
      "uv",
      ["run", "python", "-m", "neurodrive", "sweep", "--config", "configs/toy_sine.yaml", "--seeds", `0-${seeds - 1}`, "--tag", tag, "--epochs", String(epochs)],
      { cwd: repoRoot, stdio: "inherit" },
    );
  }
}
