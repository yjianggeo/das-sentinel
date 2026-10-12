import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { transferableAbortController } from "node:util";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, beforeEach } from "vitest";
import { renderApp } from "../test/render";

const server = setupServer();
const domAbortController = globalThis.AbortController;
const SITE = {
  id: "site-1", name: "JJK", data_path: "/data", timezone: "Asia/Shanghai",
  cable_type: "downhole", enabled: true, created_at: "2026-06-16T00:00:00Z",
  current_profile: { id: "p1", version: 1, payload: {}, is_current: true, created_at: "2026-06-16T00:00:00Z" },
};
const SCAN = {
  id: "scan-1", site_id: SITE.id, relative_path: "jjk2025_2026/20260616",
  start_utc: "2026-06-16T09:05:00Z", end_utc: "2026-06-16T09:15:00Z",
  distance_start_m: 499, distance_stop_m: null, status: "succeeded",
  created_at: "2026-06-16T10:00:00Z", error_message: null,
};
const CANDIDATE = {
  id: "candidate-low", start_utc: SCAN.start_utc, end_utc: SCAN.end_utc,
  peak_utc: "2026-06-16T09:08:00Z", score: 0.6, snr_db: null,
  active_channel_fraction: 0.8, coherence: 0.75, dominant_frequency_hz: 0.05,
  frequency_resolution_hz: 0.005, energy_duration_s: 200, latest_review: null,
};
const DETAIL = {
  ...SCAN,
  result: {
    sample_rate_hz: 5, distance_start_m: 499, distance_stop_m: 3403,
    channel_count: 727, source_count: 20, warnings: ["边缘低频测量需结合波形复核"],
    measurements: { dominant_frequency_hz: 0.05 },
    candidates: [CANDIDATE, { ...CANDIDATE, id: "candidate-high", score: 0.95 }],
    plots: Object.fromEntries(["waterfall", "waveform", "psd", "stft"].map((kind) => [kind, `/api/v1/low-frequency/scans/scan-1/plots/${kind}`])),
  },
};

beforeAll(() => {
  globalThis.AbortController = transferableAbortController().constructor as typeof AbortController;
  server.listen({ onUnhandledRequest: "error" });
});
beforeEach(() => {
  server.resetHandlers();
  server.use(
    http.get("/api/v1/session", () => HttpResponse.json({ admin_id: 1, csrf_token: "csrf-lf" })),
    http.get("/api/v1/dashboard", () => HttpResponse.json({ window_status_counts: {}, model_scan_status_counts: {}, review_counts: { earthquake: 0, noise: 0, pending: 0 } })),
    http.get("/api/v1/sites", () => HttpResponse.json([SITE])),
    http.get("/api/v1/low-frequency/scans", () => HttpResponse.json([SCAN])),
    http.get("/api/v1/low-frequency/scans/scan-1", () => HttpResponse.json(DETAIL)),
  );
});
afterAll(() => { server.close(); globalThis.AbortController = domAbortController; });

async function openScan() {
  renderApp("/low-frequency");
  await userEvent.click(await screen.findByRole("button", { name: "扫描 scan-1" }));
  await screen.findByRole("button", { name: "候选 candidate-high" });
}

it("opens the independent page without scanning and submits UTC+8 inputs only on click", async () => {
  const submissions: { body: unknown; csrf: string | null }[] = [];
  server.use(http.post("/api/v1/low-frequency/scans", async ({ request }) => {
    submissions.push({ body: await request.json(), csrf: request.headers.get("X-CSRF-Token") });
    return HttpResponse.json(SCAN, { status: 202 });
  }));
  renderApp("/low-frequency");

  expect(await screen.findByRole("heading", { name: "低频连续扫描" })).toBeVisible();
  expect(screen.getByRole("link", { name: "低频扫描" })).toHaveAttribute("href", "/low-frequency");
  expect(screen.getByLabelText("开始时间（UTC+8）")).toHaveValue("2026-06-16T17:05");
  expect(screen.getByLabelText("结束时间（UTC+8）")).toHaveValue("2026-06-16T17:15");
  expect(screen.getByLabelText("起点距离（m）")).toHaveValue("499");
  expect(submissions).toEqual([]);
  await userEvent.click(screen.getByRole("button", { name: "开始低频扫描" }));

  await waitFor(() => expect(submissions).toHaveLength(1));
  expect(submissions[0]).toEqual({ csrf: "csrf-lf", body: {
    site_id: "site-1", relative_path: "jjk2025_2026/20260616",
    start_utc: "2026-06-16T09:05:00.000Z", end_utc: "2026-06-16T09:15:00.000Z",
    distance_start_m: 499, distance_stop_m: null,
  } });
});

it.each([
  ["结束时间（UTC+8）", "2026-06-16T17:14", "连续有效窗必须为 600–1200 秒"],
  ["结束时间（UTC+8）", "2026-06-16T17:26", "连续有效窗必须为 600–1200 秒"],
  ["起点距离（m）", "-1", "请输入有限且非负的距离"],
  ["终点距离（m，留空为文件末端）", "499", "终点距离必须大于起点距离"],
])("rejects invalid %s before starting a scan", async (label, value, message) => {
  let posted = false;
  server.use(http.post("/api/v1/low-frequency/scans", () => { posted = true; return HttpResponse.json(SCAN); }));
  renderApp("/low-frequency");
  await screen.findByRole("heading", { name: "低频连续扫描" });
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
  await userEvent.click(screen.getByRole("button", { name: "开始低频扫描" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(message);
  expect(posted).toBe(false);
});

it("shows the server start error without discarding entered bounds", async () => {
  server.use(http.post("/api/v1/low-frequency/scans", () => HttpResponse.json({ detail: "缺少前置 300 秒真实数据" }, { status: 422 })));
  renderApp("/low-frequency");
  await screen.findByRole("heading", { name: "低频连续扫描" });
  fireEvent.change(screen.getByLabelText("终点距离（m，留空为文件末端）"), { target: { value: "3300" } });
  await userEvent.click(screen.getByRole("button", { name: "开始低频扫描" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("缺少前置 300 秒真实数据");
  expect(screen.getByLabelText("终点距离（m，留空为文件末端）")).toHaveValue("3300");
});

it("clears history loading when a new completed scan supersedes the initial history request", async () => {
  let historyStarted = false;
  server.use(
    http.get("/api/v1/low-frequency/scans", async ({ request }) => {
      historyStarted = true;
      await new Promise<void>((resolve) => request.signal.addEventListener("abort", () => resolve()));
      return HttpResponse.json([SCAN]);
    }),
    http.post("/api/v1/low-frequency/scans", () => HttpResponse.json(SCAN, { status: 202 })),
  );
  renderApp("/low-frequency");
  await screen.findByRole("option", { name: "JJK" });
  await waitFor(() => expect(historyStarted).toBe(true));
  await userEvent.click(screen.getByRole("button", { name: "开始低频扫描" }));
  await screen.findByRole("button", { name: "候选 candidate-high" });
  expect(screen.queryByText("正在加载扫描历史…")).not.toBeInTheDocument();
});

it("lets a slow history poll finish before requesting the next update", async () => {
  let calls = 0;
  let interrupted = false;
  let release!: () => void;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  server.use(http.get("/api/v1/low-frequency/scans", async ({ request }) => {
    calls += 1;
    if (calls === 1) return HttpResponse.json([{ ...SCAN, status: "running" }]);
    request.signal.addEventListener("abort", () => { interrupted = true; });
    await pending;
    return HttpResponse.json([SCAN]);
  }));
  renderApp("/low-frequency");
  await screen.findByRole("button", { name: "扫描 scan-1" });
  try {
    await waitFor(() => expect(calls).toBe(2), { timeout: 4000 });
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 3100)); });
    expect(interrupted).toBe(false);
    expect(calls).toBe(2);
  } finally {
    await act(async () => { release(); await pending; });
  }
  expect(await screen.findByText("已完成")).toBeVisible();
}, 10000);

it("shows score-ordered unreviewed candidates and switches authenticated plot URLs", async () => {
  await openScan();
  expect(screen.getAllByRole("button", { name: /^候选 / }).map((item) => item.getAttribute("aria-label")))
    .toEqual(["候选 candidate-high", "候选 candidate-low"]);
  expect(screen.getAllByText("待复核")).toHaveLength(2);
  await userEvent.click(screen.getByText(/^诊断记录/));
  expect(screen.getByText("边缘低频测量需结合波形复核")).toBeVisible();
  expect(screen.getByRole("img", { name: "低频瀑布图" })).toHaveAttribute("src", DETAIL.result.plots.waterfall);
  await userEvent.click(screen.getByRole("tab", { name: "PSD" }));
  expect(screen.getByRole("img", { name: "低频功率谱" })).toHaveAttribute("src", DETAIL.result.plots.psd);
  await userEvent.click(screen.getByRole("button", { name: "候选 candidate-high" }));
  expect(screen.getByLabelText("低频人工标签")).toHaveValue("");
  expect(screen.getByRole("button", { name: "保存低频复核" })).toBeDisabled();
  expect(screen.getByText("信噪比（dB）").closest("tr")).toHaveTextContent("—");
});

it("reuses a review request ID on retry and changes it only after editing the review", async () => {
  const submissions: Record<string, unknown>[] = [];
  server.use(http.post("/api/v1/low-frequency/scans/scan-1/candidates/candidate-high/reviews", async ({ request }) => {
    const body = await request.json() as Record<string, unknown>; submissions.push(body);
    if (submissions.length === 1) return HttpResponse.json({ detail: "复核连接中断" }, { status: 503 });
    return HttpResponse.json({ label: body.label, notes: body.notes, created_at: "2026-06-16T10:00:00Z" });
  }));
  await openScan();
  await userEvent.click(screen.getByRole("button", { name: "候选 candidate-high" }));
  await userEvent.selectOptions(screen.getByLabelText("低频人工标签"), "earthquake");
  await userEvent.type(screen.getByLabelText("低频复核备注"), "连续面波");
  await userEvent.click(screen.getByRole("button", { name: "保存低频复核" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("复核连接中断");
  await userEvent.click(screen.getByRole("button", { name: "保存低频复核" }));
  expect(await screen.findByText("复核已保存")).toBeVisible();
  expect(submissions[0]).toEqual(submissions[1]);
  expect(submissions[0].request_id).toMatch(/^[0-9a-f-]{36}$/i);
  expect(within(screen.getByRole("button", { name: "候选 candidate-high" })).getByText("地震")).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "保存低频复核" }));
  await waitFor(() => expect(submissions).toHaveLength(3));
  expect(submissions[2]).toEqual(submissions[1]);
  await screen.findByText("复核已保存");
  await userEvent.type(screen.getByLabelText("低频复核备注"), "，已确认");
  await userEvent.click(screen.getByRole("button", { name: "保存低频复核" }));
  await waitFor(() => expect(submissions).toHaveLength(4));
  expect(submissions[3].request_id).not.toBe(submissions[2].request_id);
});

it("localizes low-frequency measurements and warnings without exposing processing metadata", async () => {
  server.use(http.get("/api/v1/low-frequency/scans/scan-1", () => HttpResponse.json({ ...DETAIL, result: {
    ...DETAIL.result,
    warnings: ["insufficient_low_frequency_resolution", "candidate_clipped_end", "文件时间存在微小偏差"],
    measurements: { band_rms: 0.02, input_unit: "rad/s", band_fraction_0_01_0_05: 0.4, valid_channel_count: 700,
      background_seconds: 300, frequency_band_hz: [0.01, 0.4], filter_context_complete: true,
      score_semantics: "uncalibrated heuristic metadata", algorithm_version: "lf-v1" },
  } })));
  await openScan();
  await userEvent.click(screen.getByText("窗口测量"));
  await userEvent.click(screen.getByText(/^诊断记录/));
  expect(screen.getByRole("row", { name: "0.01–0.05 Hz 能量占比 0.4" })).toBeVisible();
  expect(screen.getByRole("row", { name: "有效通道数 700" })).toBeVisible();
  expect(screen.getByText("幅值单位：rad/s")).toBeVisible();
  expect(screen.getByText("评分为检测证据强度，候选仍需人工判断。")).toBeVisible();
  expect(screen.getByText("低频分辨率不足，请结合实际窗口时长复核。")).toBeVisible();
  expect(screen.getByText("候选在窗口末端截断。")).toBeVisible();
  expect(screen.getByText("文件时间存在微小偏差")).toBeVisible();
  expect(screen.queryByText("uncalibrated heuristic metadata")).not.toBeInTheDocument();
  expect(screen.queryByText("filter_context_complete")).not.toBeInTheDocument();
});

it("collapses diagnostic details while showing the count and clock seam notice", async () => {
  server.use(http.get("/api/v1/low-frequency/scans/scan-1", () => HttpResponse.json({ ...DETAIL, result: {
    ...DETAIL.result, warnings: ["文件边界时钟间隙 +0.014 秒，已校正", "insufficient_background"],
  } })));
  await openScan();
  const summary = screen.getByText("诊断记录（2 条，含时钟接缝）");
  expect(summary).toBeVisible();
  expect(screen.getByText("文件边界时钟间隙 +0.014 秒，已校正")).not.toBeVisible();
  await userEvent.click(summary);
  expect(screen.getByText("文件边界时钟间隙 +0.014 秒，已校正")).toBeVisible();
  expect(screen.getByText("真实背景数据不足，信噪比仅供参考。")).toBeVisible();
});

it("labels FFT spacing and preserves an unavailable short-candidate dominant frequency", async () => {
  server.use(http.get("/api/v1/low-frequency/scans/scan-1", () => HttpResponse.json({ ...DETAIL, result: {
    ...DETAIL.result,
    warnings: ["candidate_insufficient_low_frequency_resolution"],
    measurements: { background_start_seconds: -200, background_end_seconds: -30 },
    candidates: [{ ...CANDIDATE, id: "candidate-high", dominant_frequency_hz: null, frequency_resolution_hz: 0.025 }],
  } })));
  await openScan();
  await userEvent.click(screen.getByText("窗口测量"));
  expect(screen.getByRole("row", { name: "背景起点（相对秒） -200" })).toBeVisible();
  expect(screen.getByRole("row", { name: "背景终点（相对秒） -30" })).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "候选 candidate-high" }));
  expect(screen.getByRole("row", { name: "频率采样间隔（Hz） 0.025" })).toBeVisible();
  expect(screen.getByRole("row", { name: "主频（Hz） —" })).toBeVisible();
  expect(screen.queryByText("频率分辨率（Hz）")).not.toBeInTheDocument();
  await userEvent.click(screen.getByText(/^诊断记录/));
  expect(screen.getByText("候选时长不足，无法可靠给出低频主频。")).toBeVisible();
});

it("explains fixed-background limits and STFT timing even when no candidate is detected", async () => {
  server.use(http.get("/api/v1/low-frequency/scans/scan-1", () => HttpResponse.json({ ...DETAIL, result: {
    ...DETAIL.result, candidates: [],
  } })));
  renderApp("/low-frequency");
  await userEvent.click(await screen.findByRole("button", { name: "扫描 scan-1" }));
  await screen.findByText("此窗口未检出低频候选。");
  expect(screen.getByText("背景固定为有效窗起点前 200–30 秒；前置或持续波包可能抬高阈值，未检出候选也不能排除信号。")).toBeVisible();
  expect(screen.getByText("虚线（如有）标记存在时间偏差的文件接缝；微小时间对齐也可能影响低频信号。")).toBeVisible();
  await userEvent.click(screen.getByRole("tab", { name: "STFT" }));
  expect(screen.getByText("STFT 使用 200 秒时间窗，颜色变化不能作为精确到时。")).toBeVisible();
});

it("preserves nonzero milliseconds in low-frequency candidate times in UTC+8", async () => {
  server.use(http.get("/api/v1/low-frequency/scans/scan-1", () => HttpResponse.json({ ...DETAIL, result: {
    ...DETAIL.result, candidates: [{ ...CANDIDATE, id: "candidate-high",
      start_utc: "2026-06-16T09:12:45.600Z", peak_utc: "2026-06-16T09:12:45.600Z",
      end_utc: "2026-06-16T09:13:21.000Z" }],
  } })));
  await openScan();
  const candidate = screen.getByRole("button", { name: "候选 candidate-high" });
  expect(candidate).toHaveTextContent("2026-06-16 17:12:45.600");
  await userEvent.click(candidate);
  expect(screen.getByText("候选时间：2026-06-16 17:12:45.600 – 2026-06-16 17:13:21")).toBeVisible();
});

it("preserves seconds in valid UTC+8 time bounds", async () => {
  let submitted: Record<string, unknown> | undefined;
  server.use(http.post("/api/v1/low-frequency/scans", async ({ request }) => {
    submitted = await request.json() as Record<string, unknown>;
    return HttpResponse.json(SCAN, { status: 202 });
  }));
  renderApp("/low-frequency");
  await screen.findByRole("option", { name: "JJK" });
  fireEvent.change(screen.getByLabelText("开始时间（UTC+8）"), { target: { value: "2026-06-16T17:05:30" } });
  fireEvent.change(screen.getByLabelText("结束时间（UTC+8）"), { target: { value: "2026-06-16T17:15:30" } });
  await userEvent.click(screen.getByRole("button", { name: "开始低频扫描" }));
  await waitFor(() => expect(submitted).toBeDefined());
  expect(submitted).toMatchObject({ start_utc: "2026-06-16T09:05:30.000Z", end_utc: "2026-06-16T09:15:30.000Z" });
});

it("ignores an old scan response after another scan is selected", async () => {
  let release!: () => void;
  let started = false;
  let aborted = false;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  server.use(
    http.get("/api/v1/low-frequency/scans", () => HttpResponse.json([SCAN, { ...SCAN, id: "scan-2" }])),
    http.get("/api/v1/low-frequency/scans/scan-1", async ({ request }) => {
      started = true;
      request.signal.addEventListener("abort", () => { aborted = true; });
      await pending; return HttpResponse.json(DETAIL);
    }),
    http.get("/api/v1/low-frequency/scans/scan-2", () => HttpResponse.json({ ...DETAIL, id: "scan-2", result: {
      ...DETAIL.result, candidates: [{ ...CANDIDATE, id: "candidate-second" }],
    } })),
  );
  renderApp("/low-frequency");
  await userEvent.click(await screen.findByRole("button", { name: "扫描 scan-1" }));
  await waitFor(() => expect(started).toBe(true));
  await userEvent.click(screen.getByRole("button", { name: "扫描 scan-2" }));
  await screen.findByRole("button", { name: "候选 candidate-second" });
  await act(async () => { release(); await pending; });
  expect(aborted).toBe(true);
  expect(screen.queryByRole("button", { name: "候选 candidate-high" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "扫描 scan-2" })).toHaveAttribute("aria-pressed", "true");
});

it("isolates a pending review from a different candidate form", async () => {
  let release!: () => void;
  let started = false;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  server.use(http.post("/api/v1/low-frequency/scans/scan-1/candidates/candidate-high/reviews", async () => {
    started = true; await pending;
    return HttpResponse.json({ label: "earthquake", notes: "旧候选", created_at: "2026-06-16T10:00:00Z" });
  }));
  await openScan();
  await userEvent.click(screen.getByRole("button", { name: "候选 candidate-high" }));
  await userEvent.selectOptions(screen.getByLabelText("低频人工标签"), "earthquake");
  await userEvent.click(screen.getByRole("button", { name: "保存低频复核" }));
  await waitFor(() => expect(started).toBe(true));
  await userEvent.click(screen.getByRole("button", { name: "候选 candidate-low" }));
  await userEvent.selectOptions(screen.getByLabelText("低频人工标签"), "traffic");
  await userEvent.type(screen.getByLabelText("低频复核备注"), "当前候选");
  await act(async () => { release(); await pending; });
  expect(screen.getByLabelText("低频人工标签")).toHaveValue("traffic");
  expect(screen.getByLabelText("低频复核备注")).toHaveValue("当前候选");
  expect(screen.getByRole("button", { name: "保存低频复核" })).toBeEnabled();
  expect(screen.queryByText("复核已保存")).not.toBeInTheDocument();
  expect(within(screen.getByRole("button", { name: "候选 candidate-low" })).getByText("待复核")).toBeVisible();
});

it("stops running scan polling after the page unmounts", async () => {
  let calls = 0;
  server.use(http.get("/api/v1/low-frequency/scans/scan-1", () => {
    calls += 1;
    return HttpResponse.json({ ...SCAN, status: "running", result: null });
  }));
  const view = renderApp("/low-frequency");
  await userEvent.click(await screen.findByRole("button", { name: "扫描 scan-1" }));
  await screen.findByText("扫描进行中，每 3 秒更新进度。");
  view.unmount();
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 3100)); });
  expect(calls).toBe(1);
});

it("polls a running selected scan and stops after it completes", async () => {
  let calls = 0;
  server.use(http.get("/api/v1/low-frequency/scans/scan-1", () => {
    calls += 1;
    return HttpResponse.json(calls === 1 ? { ...SCAN, status: "running", result: null } : DETAIL);
  }));
  renderApp("/low-frequency");
  await userEvent.click(await screen.findByRole("button", { name: "扫描 scan-1" }));
  await waitFor(() => expect(calls).toBe(1));
  await screen.findByRole("button", { name: "候选 candidate-high" }, { timeout: 4500 });
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 3100)); });
  expect(calls).toBe(2);
}, 10000);
