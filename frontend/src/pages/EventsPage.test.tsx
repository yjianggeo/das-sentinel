import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { transferableAbortController } from "node:util";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, beforeEach } from "vitest";
import { renderApp } from "../test/render";

const server = setupServer();
const domAbortController = globalThis.AbortController;
const nativeAbortController = transferableAbortController().constructor as typeof AbortController;

beforeAll(() => {
  globalThis.AbortController = nativeAbortController;
  server.listen();
});
beforeEach(() => {
  server.resetHandlers();
  server.use(
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
  );
});
afterAll(() => {
  server.close();
  globalThis.AbortController = domAbortController;
});

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

const SUMMARY_TWO = {
  ...SUMMARY,
  id: "event-2",
  event_uid: "uid-2",
  occurred_at: "2026-07-26T11:42:16Z",
};

const STALE_SIMILAR = {
  ...SUMMARY,
  id: "event-stale-similar",
  event_uid: "uid-stale-similar",
  occurred_at: "2026-07-26T12:42:16Z",
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

const DETAIL_TWO = {
  ...DETAIL,
  ...SUMMARY_TWO,
  run: {
    ...DETAIL.run,
    id: "run-2",
  },
};

function deferredResponse(expectedDeliveries: number) {
  let resolve: () => void;
  let markStarted: () => void;
  let markDelivered: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  const started = new Promise<void>((done) => {
    let count = 0;
    markStarted = () => {
      count += 1;
      if (count === expectedDeliveries) done();
    };
  });
  const delivered = new Promise<void>((done) => {
    let count = 0;
    markDelivered = () => {
      count += 1;
      if (count === expectedDeliveries) done();
    };
  });
  return {
    promise,
    started,
    delivered,
    markStarted: () => markStarted(),
    markDelivered: () => markDelivered(),
    resolve: () => resolve(),
  };
}

function useEventHandlers() {
  server.use(
    http.get("/api/v1/events", () => HttpResponse.json([SUMMARY])),
    http.get("/api/v1/events/event-1", () => HttpResponse.json(DETAIL)),
    http.get("/api/v1/events/event-1/similar", () => HttpResponse.json([])),
  );
}

function useTwoEventHandlers(
  first: ReturnType<typeof deferredResponse>,
  second: ReturnType<typeof deferredResponse>,
) {
  server.use(
    http.get("/api/v1/events", () => HttpResponse.json([SUMMARY, SUMMARY_TWO])),
    http.get("/api/v1/events/event-1", async () => {
      first.markStarted();
      await first.promise;
      first.markDelivered();
      return HttpResponse.json(DETAIL);
    }),
    http.get("/api/v1/events/event-1/similar", async () => {
      first.markStarted();
      await first.promise;
      first.markDelivered();
      return HttpResponse.json([STALE_SIMILAR]);
    }),
    http.get("/api/v1/events/event-2", async () => {
      second.markStarted();
      await second.promise;
      second.markDelivered();
      return HttpResponse.json(DETAIL_TWO);
    }),
    http.get("/api/v1/events/event-2/similar", async () => {
      second.markStarted();
      await second.promise;
      second.markDelivered();
      return HttpResponse.json([]);
    }),
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

it("keeps the latest selected detail when responses finish in reverse order", async () => {
  const first = deferredResponse(2);
  const second = deferredResponse(2);
  useTwoEventHandlers(first, second);
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 10:42:16"));
  await first.started;
  await userEvent.click(await screen.findByText("2026-07-26 11:42:16"));
  await second.started;
  second.resolve();
  await second.delivered;

  expect(
    await screen.findByRole("heading", { name: "2026-07-26 11:42:16" }),
  ).toBeVisible();
  await act(async () => {
    first.resolve();
    await first.delivered;
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  await waitFor(() => {
    expect(
      screen.queryByRole("heading", { name: "2026-07-26 10:42:16" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "2026-07-26 11:42:16" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "2026-07-26 12:42:16" }),
    ).not.toBeInTheDocument();
  });
});

it("double click sends one review request with one UUID", async () => {
  const requests: unknown[] = [];
  useEventHandlers();
  server.use(
    http.post("/api/v1/events/event-1/reviews", async ({ request }) => {
      requests.push(await request.json());
      await new Promise((resolve) => setTimeout(resolve, 50));
      return HttpResponse.json(
        {
          id: 7,
          label: "earthquake",
          note: null,
          created_at: "2026-07-26T11:00:00Z",
        },
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
  const save = screen.getByRole("button", { name: "保存复核" });
  await Promise.all([userEvent.click(save), userEvent.click(save)]);

  expect(requests).toHaveLength(1);
  expect(requests[0]).toMatchObject({
    request_id: expect.stringMatching(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
    ),
  });
});

it("retries a failed review with the same UUID", async () => {
  const requests: Array<{ request_id: string }> = [];
  let attempts = 0;
  useEventHandlers();
  server.use(
    http.post("/api/v1/events/event-1/reviews", async ({ request }) => {
      requests.push((await request.json()) as { request_id: string });
      attempts += 1;
      if (attempts === 1) return HttpResponse.json({ detail: "retry" }, { status: 500 });
      return HttpResponse.json(
        {
          id: 8,
          label: "earthquake",
          note: null,
          created_at: "2026-07-26T11:00:00Z",
        },
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
  const save = screen.getByRole("button", { name: "保存复核" });
  await userEvent.click(save);
  await screen.findByRole("alert");
  await userEvent.click(save);

  expect(requests).toHaveLength(2);
  expect(requests[1].request_id).toBe(requests[0].request_id);
});
