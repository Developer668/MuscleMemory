import { defineConfig, devices } from "@playwright/test";

const executablePath = process.env.PLAYWRIGHT_EXECUTABLE_PATH;
const browserLaunch = executablePath
  ? { launchOptions: { executablePath } }
  : { channel: "chromium" as const };

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: "line",
  timeout: 60_000,
  expect: { timeout: 20_000 },
  use: {
    baseURL: "http://127.0.0.1:4177",
    trace: "retain-on-failure",
    ...browserLaunch,
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-chromium", use: { ...devices["Pixel 7"] } },
  ],
  webServer: [
    {
      command: "uv run python -m ops.api.serve",
      cwd: "..",
      env: {
        MM_API_BACKEND_FACTORY: "muscle_memory.runtime:create_api_backend",
        MM_API_HOST: "127.0.0.1",
        MM_API_PORT: "8000",
        MUSCLE_MEMORY_COORDINATOR_DB_PATH: "/tmp/muscle-memory-playwright-coordinator.sqlite3",
        MUSCLE_MEMORY_TELEMETRY_SPOOL: "/tmp/muscle-memory-playwright-telemetry.sqlite3",
        MUSCLE_MEMORY_FALKORDB_CACHE_PATH: "/tmp/muscle-memory-playwright-graph.jsonl",
      },
      url: "http://127.0.0.1:8000/api/v1/health",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 4177",
      url: "http://127.0.0.1:4177/",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
