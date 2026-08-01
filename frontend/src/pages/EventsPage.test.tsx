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

const SUMMARY = {
  id: "event-1",
  event_uid: "uid-1",
  site_id: "site-1",
  occurred_at: "2026-07-26T10:42:16Z",
  start_sample: 100,
  end_sample: 200,
  channel_start: 10,
  channel_stop: 40,
  machine_label: "unknown",
  latest_review: null,
};

const DETAIL = {
  ...SUMMARY,
  features: { rms: 1.5, duration_s: 2.0 },
  evidence: [
    { detector: "stalta", score: 0.82, payload: { peak_ratio: 12.3 } },
    { detector: "coherence", score: 0.71, payload: { semblance: 0.2 } },
  ],
  machine_history: [
    {
      id: 1,
      label: "unknown",
      decision_score: 0.55,
      method: "rules",
      reasons: ["单一弱候选"],
      model_version_id: null,
      created_at: "2026-07-26T10:42:20Z",
    },
  ],
  review_history: [],
  artifacts: [
    { id: "art-1", kind: "waterfall_raw", media_type: "image/png" },
    { id: "art-2", kind: "psd", media_type: "image/png" },
    { id: "art-3", kind: "stft", media_type: "image/png" },
  ],
  run: {
    id: "run-1",
    source_file_id: "source-1",
    source_fingerprint: "a".repeat(64),
    processing_profile_id: "profile-1",
    requested_backend: "cpu",
    actual_backend: "cpu",
    backend_fallback_reason: null,
    profile_version: 1,
    pipeline_version: "pipeline-v1",
    model_version_id: null,
    timings_ms: { total: 100 },
    started_at: null,
    finished_at: null,
  },
};

function useEventHandlers() {
  server.use(
    http.get("/api/v1/events", () => HttpResponse.json([SUMMARY])),
    http.get("/api/v1/events/event-1", () => HttpResponse.json(DETAIL)),
    http.get("/api/v1/events/event-1/similar", () => HttpResponse.json([])),
  );
}

it("shows event queue artifacts and independent method evidence", async () => {
  useEventHandlers();
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 10:42:16"));

  expect(await screen.findByAltText("事件瀑布图")).toBeVisible();
  expect(screen.getByRole("tab", { name: "PSD" })).toBeVisible();
  expect(screen.getByText(/STA\/LTA 多道符合/)).toBeVisible();
  expect(screen.getByText("规则判断")).toBeVisible();
});

it("switches artifact tabs", async () => {
  useEventHandlers();
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 10:42:16"));
  await userEvent.click(await screen.findByRole("tab", { name: "PSD" }));

  expect(await screen.findByAltText("事件功率谱")).toBeVisible();
});

it("submits a human review and keeps the machine label", async () => {
  let reviewed = false;
  useEventHandlers();
  server.use(
    http.post("/api/v1/events/event-1/reviews", async ({ request }) => {
      const body = (await request.json()) as Record<string, unknown>;
      expect(body).toMatchObject({ label: "earthquake", note: "可见 P/S" });
      reviewed = true;
      return HttpResponse.json(
        { id: 7, label: "earthquake", note: "可见 P/S", created_at: "2026-07-26T11:00:00Z" },
        { status: 201 },
      );
    }),
  );
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 10:42:16"));
  await userEvent.selectOptions(
    await screen.findByLabelText("人工标签"),
    "earthquake",
  );
  await userEvent.type(screen.getByLabelText("备注"), "可见 P/S");
  await userEvent.click(screen.getByRole("button", { name: "保存复核" }));

  expect(await screen.findByText("已保存")).toBeVisible();
  expect(reviewed).toBe(true);
  expect(screen.getAllByText("待定").length).toBeGreaterThan(0);
});
