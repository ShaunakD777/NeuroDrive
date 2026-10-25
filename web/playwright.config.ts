import { defineConfig, devices } from "@playwright/test";

// The smoke test needs both servers: the Python one (port 8000) and the Vite dev server (5173).
// Playwright starts whichever is not already running. The Python server is started through
// `python -m neurodrive`, because Windows Smart App Control can block the generated nd.exe
// launcher. global-setup.ts trains the small toy runs the toy-lab test opens (task 3.8).
export default defineConfig({
  testDir: "e2e",
  globalSetup: "./e2e/global-setup.ts",
  reporter: "list",
  use: { baseURL: "http://127.0.0.1:5173" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "uv run python -m neurodrive serve",
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
