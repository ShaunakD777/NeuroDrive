import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

// Before the browser tests: make sure the toy lab has runs to show (task 3.8). Two small toy runs
// tagged "e2e" are trained into the repository's runs/ folder (git-ignored) unless completed ones
// are already there; training is seeded, so an existing run holds exactly the same numbers.

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

export default function globalSetup(): void {
  for (const { task, args } of E2E_RUNS) {
    if (hasCompletedRun(task)) continue;
    execFileSync(
      "uv",
      ["run", "python", "scripts/train_toy.py", "--task", task, "--tag", "e2e", ...args],
      { cwd: repoRoot, stdio: "inherit" },
    );
  }
}
