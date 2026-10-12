import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { transferableAbortController } from "node:util";
import { afterAll, beforeAll, beforeEach, expect, it } from "vitest";
import { renderApp } from "../test/render";

const server = setupServer();
const domAbortController = globalThis.AbortController;
let currentRequests: string[] = [];
let legacyRequests: string[] = [];

const site = {
  id: "site-1",
  name: "测试站点",
  data_path: "D:/data",
  timezone: "Asia/Shanghai",
  cable_type: "urban",
  enabled: true,
  created_at: "2026-09-27T00:00:00Z",
  current_profile: {
    id: "profile-1",
    version: 1,
    payload: { scan_distance_start_m: 0, scan_distance_stop_m: 1000 },
    is_current: true,
    created_at: "2026-09-27T00:00:00Z",
  },
};

const completedScan = {
  id: "scan-1",
  site_id: site.id,
  path: "2026-09",
  phase: "completed",
  initial_checked_files: 1,
  initial_found_files: 1,
  verification_checked_files: 1,
  verification_found_files: 1,
  stable_files: 1,
  submitted_files: 1,
  skipped_files: 0,
  stability_seconds_remaining: 0,
  error_message: null,
};

const activeScan = { ...completedScan, phase: "initial_scan" };
const createdWindowJob = {
  id: "new-window-job",
  window_id: "window-1",
  site_name: site.name,
  site_timezone: site.timezone,
  start_utc: "2026-09-27T00:00:00Z",
  status: "queued",
  attempt_count: 0,
  error_message: null,
};

beforeAll(() => {
  globalThis.AbortController = transferableAbortController().constructor as typeof AbortController;
  server.listen();
});
beforeEach(() => {
  sessionStorage.clear();
  currentRequests = [];
  legacyRequests = [];
  server.resetHandlers();
  server.use(
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
    http.get("/api/v1/sites", () => HttpResponse.json([])),
    http.get("/api/v1/scan-coverage", ({ request }) => {
      currentRequests.push(new URL(request.url).pathname);
      return HttpResponse.json({ year: 2026, timezone: "Asia/Shanghai", available_years: [2026],
        days: Array.from({ length: 365 }, (_, index) => ({
          date: new Date(Date.UTC(2026, 0, index + 1)).toISOString().slice(0, 10),
          scanned_seconds: 0,
        })), total_scanned_seconds: 0, scanned_days: 0 });
    }),
    http.get("/api/v1/model-scans", ({ request }) => {
      currentRequests.push(new URL(request.url).pathname);
      return HttpResponse.json({ items: [], total: 0, limit: 10, offset: 0 });
    }),
    http.get("/api/v1/jobs/estimate", ({ request }) => {
      currentRequests.push(new URL(request.url).pathname);
      return HttpResponse.json({
        as_of: "2026-09-27T00:00:00Z",
        status: "complete",
        queued: 0,
        running: 0,
        remaining_seconds: 0,
        estimated_finish_at: null,
        queues: [],
      });
    }),
    http.get("/api/v1/window-jobs", ({ request }) => {
      currentRequests.push(new URL(request.url).pathname);
      return HttpResponse.json({ items: [], total: 0, limit: 50, offset: 0 });
    }),
    http.get("/api/v1/jobs/page", ({ request }) => {
      legacyRequests.push(new URL(request.url).pathname);
      return HttpResponse.json({
        items: [],
        total: 0,
        global_counts: { queued: 0, running: 0, succeeded: 0, failed: 0, cancelled: 0 },
      });
    }),
    http.get("/api/v1/jobs/directories", ({ request }) => {
      legacyRequests.push(new URL(request.url).pathname);
      return HttpResponse.json([]);
    }),
  );
});
afterAll(() => {
  server.close();
  globalThis.AbortController = domAbortController;
});

it("shows only current task sections and never loads the legacy file-task APIs", async () => {
  renderApp("/jobs");

  expect(await screen.findByRole("region", { name: "普通整窗扫描" })).toBeVisible();
  expect(await screen.findByRole("region", { name: "模型扫描" })).toBeVisible();
  expect(screen.getByRole("region", { name: "整窗图件任务预计完成时间" })).toBeVisible();
  expect(screen.getByRole("region", { name: "整窗任务" })).toBeVisible();
  expect(await screen.findByText("没有符合条件的整窗任务")).toBeVisible();
  await waitFor(() => expect(currentRequests).toHaveLength(4));
  expect(currentRequests).toEqual(expect.arrayContaining([
    "/api/v1/scan-coverage",
    "/api/v1/model-scans",
    "/api/v1/jobs/estimate",
    "/api/v1/window-jobs",
  ]));
  expect(screen.queryByRole("heading", { name: "历史文件任务" })).not.toBeInTheDocument();
  expect(legacyRequests).toEqual([]);
});

it("shows the scanned-data calendar before task creation and preserves existing task sections", async () => {
  renderApp("/jobs");
  const calendar = await screen.findByRole("region", { name: "已扫描数据日历" });
  expect(calendar).toBeVisible();
  const scanner = screen.getByRole("region", { name: "普通整窗扫描" });
  expect(calendar.compareDocumentPosition(scanner) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(screen.getByRole("region", { name: "模型扫描" })).toBeVisible();
  expect(screen.getByRole("region", { name: "整窗任务" })).toBeVisible();
});

it("keeps the ordinary directory scanner without loading the historical task list", async () => {
  const ordinaryRequests: string[] = [];
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([site])),
    http.get("/api/v1/sites/site-1/active-scan", ({ request }) => {
      ordinaryRequests.push(new URL(request.url).pathname);
      return HttpResponse.json(null);
    }),
    http.get("/api/v1/sites/site-1/scan-directories", ({ request }) => {
      ordinaryRequests.push(new URL(request.url).pathname);
      return HttpResponse.json({
        site_id: "site-1",
        site_name: "测试站点",
        path: "",
        parent_path: null,
        directories: [{ name: "2026-09", path: "2026-09" }],
      });
    }),
  );

  renderApp("/jobs");
  const user = userEvent.setup();
  await user.click(await screen.findByText("选择目录并扫描"));
  await user.click(await screen.findByRole("button", { name: "选择目录并扫描 测试站点" }));

  expect(await screen.findByRole("dialog", { name: "选择 测试站点 扫描目录" })).toBeVisible();
  expect(await screen.findByRole("button", { name: "2026-09" })).toBeVisible();
  expect(ordinaryRequests).toEqual(expect.arrayContaining([
    "/api/v1/sites/site-1/active-scan",
    "/api/v1/sites/site-1/scan-directories",
  ]));
  expect(legacyRequests).toEqual([]);
});

async function startOrdinaryScan() {
  const user = userEvent.setup();
  await user.click(await screen.findByText("选择目录并扫描"));
  await user.click(await screen.findByRole("button", { name: "选择目录并扫描 测试站点" }));
  await user.click(await screen.findByRole("button", { name: "2026-09" }));
  await user.click(await screen.findByRole("button", { name: "选择当前目录" }));
  await user.click(await screen.findByRole("button", { name: "开始扫描" }));
}

function installScanCompletionHandlers(mode: "post" | "poll" | "restore") {
  let completed = mode === "restore";
  let windowJobCalls = 0;
  if (mode === "restore") {
    sessionStorage.setItem("das-sentinel.active-scans", JSON.stringify({ [site.id]: "scan-1" }));
  }
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([site])),
    http.get("/api/v1/sites/site-1/active-scan", () => HttpResponse.json(null)),
    http.get("/api/v1/sites/site-1/scan-directories", ({ request }) => {
      const path = new URL(request.url).searchParams.get("path") ?? "";
      return HttpResponse.json({
        site_id: site.id,
        site_name: site.name,
        path,
        parent_path: path ? "" : null,
        directories: path ? [] : [{ name: "2026-09", path: "2026-09" }],
      });
    }),
    http.post("/api/v1/sites/site-1/scan", () => {
      if (mode === "post") completed = true;
      return HttpResponse.json(mode === "post" ? completedScan : activeScan);
    }),
    http.get("/api/v1/scans/scan-1", () => {
      completed = true;
      return HttpResponse.json(completedScan);
    }),
    http.get("/api/v1/window-jobs", () => {
      windowJobCalls += 1;
      const items = completed && windowJobCalls > 1 ? [createdWindowJob] : [];
      return HttpResponse.json({ items, total: items.length, limit: 50, offset: 0 });
    }),
  );
  return () => windowJobCalls;
}

it("refreshes empty whole-window jobs when a scan POST is already complete", async () => {
  const windowJobCalls = installScanCompletionHandlers("post");
  renderApp("/jobs");
  expect(await screen.findByText("没有符合条件的整窗任务")).toBeVisible();

  await startOrdinaryScan();

  expect(await screen.findByText("new-wind")).toBeVisible();
  expect(windowJobCalls()).toBeGreaterThanOrEqual(2);
  expect(legacyRequests).toEqual([]);
});

it("refreshes an idle ETA immediately when a directory scan creates artifact jobs", async () => {
  installScanCompletionHandlers("post");
  let estimateReads = 0;
  server.use(http.get("/api/v1/jobs/estimate", () => {
    estimateReads++;
    return HttpResponse.json({ as_of: "2026-09-27T00:00:00Z", status: estimateReads === 1 ? "complete" : "estimating",
      queued: estimateReads === 1 ? 0 : 7, running: 0, remaining_seconds: null, estimated_finish_at: null, queues: [] });
  }));
  renderApp("/jobs");
  expect(await screen.findByText("0 个排队 · 0 个运行中")).toBeVisible();
  await startOrdinaryScan();
  expect(await screen.findByText("7 个排队 · 0 个运行中")).toBeVisible();
});

it("refreshes ETA immediately after cancelling an artifact job", async () => {
  let cancelled = false;
  const job = { ...createdWindowJob, end_utc: "2026-09-27T00:00:30Z", site_id: site.id, profile_id: "profile-1", worker_id: null,
    heartbeat_at: null, error_code: null, created_at: "2026-09-27T00:00:00Z" };
  server.use(
    http.get("/api/v1/window-jobs", () => HttpResponse.json({ items: [{ ...job, status: cancelled ? "cancelled" : "queued" }], total: 1, limit: 50, offset: 0 })),
    http.post("/api/v1/window-jobs/new-window-job/cancel", () => { cancelled = true; return HttpResponse.json({ ...job, status: "cancelled" }); }),
    http.get("/api/v1/jobs/estimate", () => HttpResponse.json({ as_of: "2026-09-27T00:00:00Z", status: cancelled ? "complete" : "waiting_worker",
      queued: cancelled ? 0 : 1, running: 0, remaining_seconds: null, estimated_finish_at: null, queues: [] })),
  );
  renderApp("/jobs");
  expect(await screen.findByText("1 个排队 · 0 个运行中")).toBeVisible();
  await userEvent.click(await screen.findByRole("button", { name: "取消" }));
  expect(await screen.findByText("0 个排队 · 0 个运行中")).toBeVisible();
});

it("refreshes empty whole-window jobs when scan polling reaches a terminal state", async () => {
  const windowJobCalls = installScanCompletionHandlers("poll");
  renderApp("/jobs");
  expect(await screen.findByText("没有符合条件的整窗任务")).toBeVisible();

  await startOrdinaryScan();

  expect(await screen.findByText("new-wind", {}, { timeout: 3000 })).toBeVisible();
  expect(windowJobCalls()).toBeGreaterThanOrEqual(2);
  expect(legacyRequests).toEqual([]);
});

it("refreshes empty whole-window jobs after restoring a terminal scan", async () => {
  const windowJobCalls = installScanCompletionHandlers("restore");
  renderApp("/jobs");

  expect(await screen.findByText("new-wind")).toBeVisible();
  expect(windowJobCalls()).toBeGreaterThanOrEqual(2);
  expect(sessionStorage.getItem("das-sentinel.active-scans")).toBeNull();
  expect(legacyRequests).toEqual([]);
});
