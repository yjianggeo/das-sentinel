import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, beforeEach } from "vitest";
import { renderApp } from "../test/render";

const server = setupServer();

beforeAll(() => server.listen());
beforeEach(() => {
  server.resetHandlers();
  server.use(
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
  );
});
afterAll(() => server.close());

const FAILED_JOB = {
  id: "job-failed-1",
  source_file_id: "source-1",
  processing_profile_id: "profile-1",
  pipeline_version: "pipeline-v1",
  status: "failed",
  attempt_count: 3,
  progress_numerator: 4,
  progress_denominator: 10,
  error_code: "corrupt_hdf5",
  error_message: "missing dataset",
  created_at: "2026-07-30T00:00:00Z",
  updated_at: "2026-07-30T00:01:00Z",
};

it("shows job progress and retries a failed job", async () => {
  let retried = false;
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([])),
    http.get("/api/v1/jobs", () =>
      HttpResponse.json([
        { ...FAILED_JOB, status: retried ? "queued" : "failed" },
      ]),
    ),
    http.post("/api/v1/jobs/job-failed-1/retry", () => {
      retried = true;
      return HttpResponse.json({ ...FAILED_JOB, status: "queued" });
    }),
  );
  renderApp("/jobs");

  expect(await screen.findByText("失败")).toBeVisible();
  expect(screen.getByRole("progressbar")).toHaveAttribute(
    "aria-valuenow",
    "40",
  );

  await userEvent.click(screen.getByRole("button", { name: "重试" }));

  expect(await screen.findByText("排队")).toBeVisible();
  expect(retried).toBe(true);
});

it("cancels a running job", async () => {
  let cancelled = false;
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([])),
    http.get("/api/v1/jobs", () =>
      HttpResponse.json([
        {
          ...FAILED_JOB,
          id: "job-running-1",
          status: cancelled ? "cancelled" : "running",
          error_code: null,
          error_message: null,
        },
      ]),
    ),
    http.post("/api/v1/jobs/job-running-1/cancel", () => {
      cancelled = true;
      return HttpResponse.json({ ...FAILED_JOB, status: "cancelled" });
    }),
  );
  renderApp("/jobs");

  await userEvent.click(
    await screen.findByRole("button", { name: "取消" }),
  );

  expect(await screen.findByText("已取消")).toBeVisible();
  expect(cancelled).toBe(true);
});
