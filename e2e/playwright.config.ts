import { defineConfig, devices } from "@playwright/test";

// The e2e suite expects the full stack (API + worker + seeded fixtures) to be
// reachable at E2E_BASE_URL. See docs/operations.md, "End-to-end acceptance".
export default defineConfig({
  testDir: ".",
  timeout: 60_000,
  retries: 0,
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:8000",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
