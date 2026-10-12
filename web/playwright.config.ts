import { defineConfig, devices } from "@playwright/test";

// The smoke test needs both servers: the Python one (port 8000) and the Vite dev server (5173).
// Playwright starts whichever is not already running.
export default defineConfig({
  testDir: "e2e",
  reporter: "list",
  use: { baseURL: "http://127.0.0.1:5173" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "uv run nd serve",
      cwd: "..",
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
