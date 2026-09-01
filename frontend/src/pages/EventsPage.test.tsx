import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { transferableAbortController } from "node:util";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, beforeEach, vi } from "vitest";
import type { EventSummary } from "../api";
import ReviewForm from "../components/ReviewForm";
import { renderApp } from "../test/render";

const server = setupServer();
const domAbortController = globalThis.AbortController;
const nativeAbortController = transferableAbortController().constructor as typeof AbortController;
const originalCreateObjectUrl = Object.getOwnPropertyDescriptor(URL, "createObjectURL");
const originalRevokeObjectUrl = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");

beforeAll(() => {
  globalThis.AbortController = nativeAbortController;
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    value: vi.fn(() => "blob:event-artifact"),
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    value: vi.fn(),
  });
  server.listen();
});
beforeEach(() => {
  server.resetHandlers();
  server.use(
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
    http.get("/api/v1/artifacts/:artifactId", () =>
      new HttpResponse(new Blob(["png"], { type: "image/png" })),
    ),
    http.get("/api/v1/events/:eventId/waterfall-options", () =>
      HttpResponse.json({
        automatic_distance_start_m: 12,
        automatic_distance_stop_m: 20,
        available_distance_start_m: 8,
        available_distance_stop_m: 24,
        primary_detector: "stalta",
        primary_detector_display: "STA/LTA",
        filter_low_hz: 3.5,
        filter_high_hz: 27.25,
        distance_unit: "m",
      }),
    ),
  );
});
afterAll(() => {
  server.close();
  globalThis.AbortController = domAbortController;
  if (originalCreateObjectUrl) {
    Object.defineProperty(URL, "createObjectURL", originalCreateObjectUrl);
  } else {
    Reflect.deleteProperty(URL, "createObjectURL");
  }
  if (originalRevokeObjectUrl) {
    Object.defineProperty(URL, "revokeObjectURL", originalRevokeObjectUrl);
  } else {
    Reflect.deleteProperty(URL, "revokeObjectURL");
  }
});

const SUMMARY = {
  id: "event-1",
  event_uid: "uid-1",
  site_id: "site-1",
  site_timezone: "Asia/Shanghai",
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

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));

  expect(await screen.findByAltText("事件瀑布图")).toBeVisible();
  expect(screen.getByRole("tab", { name: "PSD" })).toBeVisible();
  expect(screen.getByText(/STA\/LTA 多道符合/)).toBeVisible();
  expect(screen.getByText("规则判断")).toBeVisible();
});

it("formats event UTC timestamps in the station timezone", async () => {
  useEventHandlers();
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));

  expect(
    await screen.findByRole("heading", { name: "2026-07-26 18:42:16" }),
  ).toBeVisible();
});

it("uses the event station timezone instead of the host timezone for detail headings", async () => {
  // Catches formatOccurredAt(detail.occurred_at) dropping the explicit station timezone.
  const pacificSummary = { ...SUMMARY, site_timezone: "America/Los_Angeles" };
  const pacificDetail = { ...DETAIL, ...pacificSummary };
  server.use(
    http.get("/api/v1/events", () => HttpResponse.json([pacificSummary])),
    http.get("/api/v1/events/event-1", () => HttpResponse.json(pacificDetail)),
    http.get("/api/v1/events/event-1/similar", () => HttpResponse.json([])),
  );
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 03:42:16"));

  expect(
    await screen.findByRole("heading", { name: "2026-07-26 03:42:16" }),
  ).toBeVisible();
});

it("formats similar event UTC timestamps in the station timezone", () => {
  const similarEvent: EventSummary = {
    id: "similar-event-1",
    event_uid: "similar-uid-1",
    site_id: "site-1",
    site_timezone: "Asia/Shanghai",
    occurred_at: "2026-07-26T11:42:16Z",
    start_sample: 100,
    end_sample: 200,
    channel_start: 10,
    channel_stop: 40,
    machine_label: "unknown",
    latest_review: null,
  };

  render(
    <ReviewForm
      eventId="event-1"
      similar={[similarEvent]}
      onReviewed={() => undefined}
      onSelectSimilar={() => undefined}
    />,
  );

  expect(
    screen.getByRole("button", { name: "2026-07-26 19:42:16" }),
  ).toBeVisible();
});

it("switches artifact tabs", async () => {
  useEventHandlers();
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));
  await userEvent.click(await screen.findByRole("tab", { name: "PSD" }));

  expect(await screen.findByAltText("事件功率谱")).toBeVisible();
});

it("returns to login when an artifact request becomes unauthorized", async () => {
  useEventHandlers();
  server.use(
    http.get("/api/v1/artifacts/art-1", () =>
      HttpResponse.json({ detail: "Unauthorized" }, { status: 401 }),
    ),
  );
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));

  expect(await screen.findByRole("button", { name: "登录" })).toBeVisible();
});

it("keeps event detail visible when artifact integrity fails", async () => {
  useEventHandlers();
  server.use(
    http.get("/api/v1/artifacts/art-1", () =>
      HttpResponse.json(
        { detail: "Artifact integrity check failed" },
        { status: 409 },
      ),
    ),
  );
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));

  expect(
    await screen.findByText("图件完整性校验失败，请联系管理员"),
  ).toBeVisible();
  expect(screen.getByRole("button", { name: "重试图件" })).toBeVisible();
  expect(screen.getByText("规则判断")).toBeVisible();
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

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));
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

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));
  await first.started;
  await userEvent.click(await screen.findByText("2026-07-26 19:42:16"));
  await second.started;
  second.resolve();
  await second.delivered;

  expect(
    await screen.findByRole("heading", { name: "2026-07-26 19:42:16" }),
  ).toBeVisible();
  await act(async () => {
    first.resolve();
    await first.delivered;
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  await waitFor(() => {
    expect(
      screen.queryByRole("heading", { name: "2026-07-26 18:42:16" }),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: "2026-07-26 19:42:16" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "2026-07-26 20:42:16" }),
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

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));
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

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));
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

it("starts a new logical review after editing a failed submission", async () => {
  const requests: Array<{
    label: string;
    note: string | null;
    request_id: string;
  }> = [];
  useEventHandlers();
  server.use(
    http.post("/api/v1/events/event-1/reviews", async ({ request }) => {
      requests.push(
        (await request.json()) as {
          label: string;
          note: string | null;
          request_id: string;
        },
      );
      if (requests.length === 1) return HttpResponse.error();
      return HttpResponse.json(
        {
          id: 9,
          label: "earthquake",
          note: "补充证据",
          created_at: "2026-07-26T11:00:00Z",
        },
        { status: 201 },
      );
    }),
  );
  renderApp("/events");

  await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));
  await userEvent.selectOptions(
    await screen.findByLabelText("人工标签"),
    "earthquake",
  );
  const save = screen.getByRole("button", { name: "保存复核" });
  await userEvent.click(save);
  await screen.findByRole("alert");

  await userEvent.type(screen.getByLabelText("备注"), "补充证据");
  expect.soft(screen.queryByRole("alert")).not.toBeInTheDocument();
  await userEvent.click(save);
  await screen.findByText("已保存");

  expect(requests).toHaveLength(2);
  expect(requests[1]).toMatchObject({
    label: "earthquake",
    note: "补充证据",
  });
  expect(requests[1].request_id).not.toBe(requests[0].request_id);
});

it("isolates an in-flight review when the selected event changes", async () => {
  const first = deferredResponse(1);
  const second = deferredResponse(1);
  const firstBodies: Array<{ request_id: string }> = [];
  const secondBodies: Array<{ request_id: string }> = [];
  let secondAttempts = 0;
  let reviewedCount = 0;
  server.use(
    http.post("/api/v1/events/event-1/reviews", async ({ request }) => {
      firstBodies.push((await request.json()) as { request_id: string });
      first.markStarted();
      await first.promise;
      first.markDelivered();
      return HttpResponse.json(
        {
          id: 9,
          label: "earthquake",
          note: null,
          created_at: "2026-07-26T11:00:00Z",
        },
        { status: 201 },
      );
    }),
    http.post("/api/v1/events/event-2/reviews", async ({ request }) => {
      secondBodies.push((await request.json()) as { request_id: string });
      secondAttempts += 1;
      if (secondAttempts === 1) {
        second.markStarted();
        await second.promise;
        second.markDelivered();
        return HttpResponse.error();
      }
      return HttpResponse.json(
        {
          id: 10,
          label: "blast",
          note: null,
          created_at: "2026-07-26T11:01:00Z",
        },
        { status: 201 },
      );
    }),
  );
  const recordReview = () => {
    reviewedCount += 1;
  };
  const { rerender } = render(
    <ReviewForm
      eventId="event-1"
      similar={[]}
      onReviewed={recordReview}
      onSelectSimilar={() => undefined}
    />,
  );

  await userEvent.selectOptions(screen.getByLabelText("人工标签"), "earthquake");
  await userEvent.click(screen.getByRole("button", { name: "保存复核" }));
  await first.started;

  rerender(
    <ReviewForm
      eventId="event-2"
      similar={[]}
      onReviewed={recordReview}
      onSelectSimilar={() => undefined}
    />,
  );
  await userEvent.selectOptions(screen.getByLabelText("人工标签"), "blast");
  await userEvent.click(screen.getByRole("button", { name: "保存复核" }));
  await second.started;

  await act(async () => {
    first.resolve();
    await first.delivered;
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  const currentSave = screen.getByRole("button", { name: "保存复核" });
  expect.soft(currentSave).toBeDisabled();
  expect.soft(screen.queryByText("已保存")).not.toBeInTheDocument();
  expect.soft(reviewedCount).toBe(0);

  await act(async () => {
    second.resolve();
    await second.delivered;
  });
  await screen.findByRole("alert");
  await userEvent.click(currentSave);
  await screen.findByText("已保存");

  expect(firstBodies).toHaveLength(1);
  expect(secondBodies).toHaveLength(2);
  expect(secondBodies[1].request_id).toBe(secondBodies[0].request_id);
  expect(reviewedCount).toBe(1);
});

for (const staleOutcome of ["success", "failure"] as const) {
  it(`keeps event B active when an unmounted event A review finishes with ${staleOutcome}`, async () => {
    const first = deferredResponse(1);
    const second = deferredResponse(1);
    const firstBodies: Array<{ request_id: string }> = [];
    const secondBodies: Array<{ request_id: string }> = [];
    let eventOneDetailRequests = 0;
    server.use(
      http.get("/api/v1/events", () =>
        HttpResponse.json([SUMMARY, SUMMARY_TWO]),
      ),
      http.get("/api/v1/events/event-1", () => {
        eventOneDetailRequests += 1;
        return HttpResponse.json(DETAIL);
      }),
      http.get("/api/v1/events/event-1/similar", () => HttpResponse.json([])),
      http.get("/api/v1/events/event-2", () => HttpResponse.json(DETAIL_TWO)),
      http.get("/api/v1/events/event-2/similar", () => HttpResponse.json([])),
      http.post("/api/v1/events/event-1/reviews", async ({ request }) => {
        firstBodies.push((await request.json()) as { request_id: string });
        first.markStarted();
        await first.promise;
        first.markDelivered();
        if (staleOutcome === "failure") return HttpResponse.error();
        return HttpResponse.json(
          {
            id: 11,
            label: "earthquake",
            note: null,
            created_at: "2026-07-26T11:00:00Z",
          },
          { status: 201 },
        );
      }),
      http.post("/api/v1/events/event-2/reviews", async ({ request }) => {
        secondBodies.push((await request.json()) as { request_id: string });
        if (secondBodies.length === 1) {
          second.markStarted();
          await second.promise;
          second.markDelivered();
          return HttpResponse.error();
        }
        return HttpResponse.json(
          {
            id: 12 + secondBodies.length,
            label: "blast",
            note: null,
            created_at: "2026-07-26T11:01:00Z",
          },
          { status: 201 },
        );
      }),
    );
    renderApp("/events");

    await userEvent.click(await screen.findByText("2026-07-26 18:42:16"));
    await userEvent.selectOptions(
      await screen.findByLabelText("人工标签"),
      "earthquake",
    );
    await userEvent.click(screen.getByRole("button", { name: "保存复核" }));
    await first.started;

    await userEvent.click(screen.getByText("2026-07-26 19:42:16"));
    expect(
      await screen.findByRole("heading", { name: "2026-07-26 19:42:16" }),
    ).toBeVisible();
    const currentLabel = screen.getByLabelText("人工标签");
    const currentNote = screen.getByLabelText("备注");
    await userEvent.selectOptions(currentLabel, "blast");
    const currentSave = screen.getByRole("button", { name: "保存复核" });
    await userEvent.click(currentSave);
    await second.started;

    await act(async () => {
      first.resolve();
      await first.delivered;
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    const formStillMounted = document.body.contains(currentSave);
    const saveWasDisabled = currentSave.hasAttribute("disabled");
    const labelWasDisabled = currentLabel.hasAttribute("disabled");
    const noteWasDisabled = currentNote.hasAttribute("disabled");
    const hadStatus = screen.queryByRole("status") !== null;
    const hadError = screen.queryByRole("alert") !== null;
    await act(async () => {
      second.resolve();
      await second.delivered;
    });

    expect(saveWasDisabled).toBe(true);
    expect(labelWasDisabled).toBe(true);
    expect(noteWasDisabled).toBe(true);
    expect(formStillMounted).toBe(true);
    expect(hadStatus).toBe(false);
    expect(hadError).toBe(false);
    expect(eventOneDetailRequests).toBe(1);

    await screen.findByRole("alert");
    await userEvent.click(currentSave);
    await screen.findByText("已保存");
    await waitFor(() => expect(secondBodies).toHaveLength(2));
    expect(secondBodies[1].request_id).toBe(secondBodies[0].request_id);

    await waitFor(() => expect(currentSave).toBeEnabled());
    await userEvent.click(currentSave);
    await waitFor(() => expect(secondBodies).toHaveLength(3));
    expect(secondBodies[2].request_id).not.toBe(secondBodies[1].request_id);
    expect(firstBodies).toHaveLength(1);
  });
}

function makeSummaries(count: number, start = 0) {
  return Array.from({ length: count }, (_, index) => ({
    ...SUMMARY,
    id: `page-event-${start + index}`,
    event_uid: `page-uid-${start + index}`,
    occurred_at: `2026-07-27T10:${String((start + index) % 60).padStart(2, "0")}:00Z`,
  }));
}

it("sends server filters and appends the next bounded page", async () => {
  const queries: URL[] = [];
  server.use(
    http.get("/api/v1/events", ({ request }) => {
      const url = new URL(request.url);
      queries.push(url);
      return HttpResponse.json(
        url.searchParams.get("offset") === "100"
          ? [...makeSummaries(2, 100), makeSummaries(1, 99)[0]]
          : makeSummaries(100),
      );
    }),
  );
  renderApp("/events");

  await userEvent.selectOptions(await screen.findByLabelText("机器标签"), "traffic");
  await userEvent.selectOptions(screen.getByLabelText("复核状态"), "false");
  await userEvent.click(await screen.findByRole("button", { name: "加载更多" }));

  expect(
    queries.some(
      (url) =>
        url.searchParams.get("limit") === "100" &&
        url.searchParams.get("offset") === "0" &&
        url.searchParams.get("machine_label") === "traffic" &&
        url.searchParams.get("reviewed") === "false",
    ),
  ).toBe(true);
  expect(queries.at(-1)?.searchParams.get("offset")).toBe("100");
  expect(screen.getAllByRole("button", { name: /^事件 / })).toHaveLength(102);
});

it("replaces initial loading with a retryable error", async () => {
  let attempts = 0;
  server.use(
    http.get("/api/v1/events", () => {
      attempts += 1;
      if (attempts === 1) {
        return HttpResponse.json({ detail: "failed" }, { status: 500 });
      }
      return HttpResponse.json([SUMMARY]);
    }),
  );
  renderApp("/events");

  expect(await screen.findByRole("alert")).toHaveTextContent("无法加载事件");
  expect(screen.queryByText("正在加载…")).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "重试" }));
  expect(await screen.findByText("2026-07-26 18:42:16")).toBeVisible();
});

it("shows an explicit empty collection state", async () => {
  server.use(http.get("/api/v1/events", () => HttpResponse.json([])));
  renderApp("/events");

  expect(await screen.findByText("暂无事件")).toBeVisible();
});

it("keeps loaded events visible when loading more fails", async () => {
  server.use(
    http.get("/api/v1/events", ({ request }) => {
      const offset = new URL(request.url).searchParams.get("offset");
      if (offset === "100") {
        return HttpResponse.json({ detail: "failed" }, { status: 500 });
      }
      return HttpResponse.json(makeSummaries(100));
    }),
  );
  renderApp("/events");

  await userEvent.click(await screen.findByRole("button", { name: "加载更多" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("无法加载更多事件");
  expect(screen.getAllByRole("button", { name: /^事件 / })).toHaveLength(100);
});

it("discards a superseded filter response and clears a filtered-out selection", async () => {
  const staleFilter = deferredResponse(1);
  const staleDetail = deferredResponse(2);
  server.use(
    http.get("/api/v1/events", async ({ request }) => {
      const label = new URL(request.url).searchParams.get("machine_label");
      if (label === "traffic") {
        staleFilter.markStarted();
        await staleFilter.promise;
        staleFilter.markDelivered();
        return HttpResponse.json([STALE_SIMILAR]);
      }
      if (label === "blast") return HttpResponse.json([SUMMARY_TWO]);
      return HttpResponse.json([SUMMARY]);
    }),
    http.get("/api/v1/events/event-1", async () => {
      staleDetail.markStarted();
      await staleDetail.promise;
      staleDetail.markDelivered();
      return HttpResponse.json(DETAIL);
    }),
    http.get("/api/v1/events/event-1/similar", async () => {
      staleDetail.markStarted();
      await staleDetail.promise;
      staleDetail.markDelivered();
      return HttpResponse.json([STALE_SIMILAR]);
    }),
  );
  renderApp("/events");

  await userEvent.click(await screen.findByRole("button", { name: "事件 event-1" }));
  await staleDetail.started;
  const filter = screen.getByLabelText("机器标签");
  await userEvent.selectOptions(filter, "traffic");
  await staleFilter.started;
  await userEvent.selectOptions(filter, "blast");

  expect(await screen.findByRole("button", { name: "事件 event-2" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "事件 event-1" })).not.toBeInTheDocument();
  await act(async () => {
    staleFilter.resolve();
    staleDetail.resolve();
    await staleFilter.delivered;
    await staleDetail.delivered;
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  expect(screen.queryByRole("button", { name: "事件 event-stale-similar" })).not.toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "2026-07-26 18:42:16" })).not.toBeInTheDocument();
});

it("uses the server page boundary for the next offset when a page contains a duplicate", async () => {
  const offsets: string[] = [];
  server.use(
    http.get("/api/v1/events", ({ request }) => {
      const offset = new URL(request.url).searchParams.get("offset") ?? "0";
      offsets.push(offset);
      if (offset === "0") return HttpResponse.json(makeSummaries(100));
      if (offset === "100") {
        return HttpResponse.json([...makeSummaries(99, 100), makeSummaries(1, 100)[0]]);
      }
      if (offset === "200") return HttpResponse.json(makeSummaries(2, 200));
      return HttpResponse.json([]);
    }),
  );
  renderApp("/events");

  await userEvent.click(await screen.findByRole("button", { name: "加载更多" }));
  await userEvent.click(await screen.findByRole("button", { name: "加载更多" }));

  expect(offsets).toEqual(["0", "100", "200"]);
  expect(screen.getAllByRole("button", { name: /^事件 / })).toHaveLength(201);
});

it("refreshes the selected detail and similar events after a review", async () => {
  let detailRequests = 0;
  let similarRequests = 0;
  server.use(
    http.get("/api/v1/events", () => HttpResponse.json([SUMMARY])),
    http.get("/api/v1/events/event-1", () => {
      detailRequests += 1;
      return HttpResponse.json(DETAIL);
    }),
    http.get("/api/v1/events/event-1/similar", () => {
      similarRequests += 1;
      return HttpResponse.json([]);
    }),
    http.post("/api/v1/events/event-1/reviews", () =>
      HttpResponse.json(
        { id: 13, label: "earthquake", note: null, created_at: "2026-07-26T11:00:00Z" },
        { status: 201 },
      ),
    ),
  );
  renderApp("/events");

  await userEvent.click(await screen.findByRole("button", { name: "事件 event-1" }));
  await userEvent.selectOptions(await screen.findByLabelText("人工标签"), "earthquake");
  await userEvent.click(screen.getByRole("button", { name: "保存复核" }));
  await screen.findByText("已保存");

  await waitFor(() => {
    expect(detailRequests).toBe(2);
    expect(similarRequests).toBe(2);
  });
});

it("aborts the latest load-more request when the page unmounts", async () => {
  const loadMore = deferredResponse(1);
  let loadMoreSignal: AbortSignal | undefined;
  server.use(
    http.get("/api/v1/events", async ({ request }) => {
      if (new URL(request.url).searchParams.get("offset") !== "100") {
        return HttpResponse.json(makeSummaries(100));
      }
      loadMoreSignal = request.signal;
      loadMore.markStarted();
      await loadMore.promise;
      loadMore.markDelivered();
      return HttpResponse.json([]);
    }),
  );
  const app = renderApp("/events");

  await userEvent.click(await screen.findByRole("button", { name: "加载更多" }));
  await loadMore.started;
  await act(async () => {
    app.unmount();
    expect(loadMoreSignal?.aborted).toBe(true);
    loadMore.resolve();
    await loadMore.delivered;
  });
});
