import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { transferableAbortController } from "node:util";
import { afterAll, afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import type { Job, ScanPhase, ScanStatus, Site } from "../api";
import { renderApp } from "../test/render";

const server = setupServer();
const ACTIVE_SCAN_STORAGE_KEY = "das-sentinel.active-scans";
const domAbortController = globalThis.AbortController;
const nativeAbortController = transferableAbortController()
  .constructor as typeof AbortController;

beforeAll(() => {
  globalThis.AbortController = nativeAbortController;
  server.listen();
});
beforeEach(() => {
  sessionStorage.clear();
  server.resetHandlers();
  server.use(
    http.get("/api/v1/jobs/directories", () => HttpResponse.json([])),
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
  );
});
afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});
afterAll(() => {
  server.close();
  globalThis.AbortController = domAbortController;
});

const SITE: Site = {
  id: "site-1",
  name: "JJK NAS Archive",
  data_path: "D:/private/nas/archive",
  timezone: "Asia/Shanghai",
  cable_type: "downhole",
  enabled: true,
  created_at: "2026-09-06T00:00:00Z",
  current_profile: {
    id: "profile-1",
    version: 1,
    payload: {},
    is_current: true,
    created_at: "2026-09-06T00:00:00Z",
  },
};

const OTHER_SITE: Site = {
  ...SITE,
  id: "site-2",
  name: "Secondary Archive",
};

function scanSnapshot(phase: ScanPhase, overrides: Partial<ScanStatus> = {}): ScanStatus {
  return {
    id: "scan-1",
    site_id: SITE.id,
    path: "jjk20240608",
    phase,
    initial_checked_files: 1250,
    initial_found_files: 86,
    verification_checked_files: 64,
    verification_found_files: 60,
    stable_files: 58,
    submitted_files: 55,
    skipped_files: 3,
    stability_seconds_remaining: 8,
    error_message: null,
    ...overrides,
  };
}

function directoryListing(request: Request) {
  const path = new URL(request.url).searchParams.get("path") ?? "";
  return {
    site_id: SITE.id,
    site_name: SITE.name,
    path,
    parent_path: path ? "" : null,
    directories: path ? [] : [{ name: "jjk20240608", path: "jjk20240608" }],
  };
}

function useJobsHandlers(sites: Site[] = [SITE]) {
  server.use(
    http.get("/api/v1/jobs/page", () => jobsResponse([])),
    http.get("/api/v1/sites", () => HttpResponse.json(sites)),
    http.get("/api/v1/sites/:siteId/active-scan", () => HttpResponse.json(null)),
    http.get("/api/v1/sites/:siteId/scan-directories", ({ request }) =>
      HttpResponse.json(directoryListing(request))),
  );
}

const FAILED_JOB: Job = {
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

function jobsResponse(items: Job[], overrides: Record<string, unknown> = {}) {
  const global_counts = { queued: 0, running: 0, succeeded: 0, failed: 0, cancelled: 0 };
  for (const job of items) global_counts[job.status] += 1;
  return HttpResponse.json({
    items: items.map((job) => ({
      ...job,
      source_filename: `${job.id}.h5`,
      source_directory: "jjk20240608",
      site_id: SITE.id,
      site_name: SITE.name,
    })),
    total: items.length,
    global_counts,
    ...overrides,
  });
}

const CANCELLED_JOB: Job = {
  ...FAILED_JOB,
  id: "cancelled-job-1",
  status: "cancelled",
};

it("deletes only a cancelled job after confirmation and leaves other rows intact", async () => {
  let jobs: Job[] = [CANCELLED_JOB, FAILED_JOB, { ...FAILED_JOB, id: "running-job-1", status: "running" }];
  let deletes = 0;
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([])),
    http.get("/api/v1/jobs/page", () => jobsResponse(jobs)),
    http.delete("/api/v1/jobs/cancelled-job-1", ({ request }) => {
      expect(request.headers.get("X-CSRF-Token")).toBe("token-1");
      deletes += 1;
      jobs = jobs.filter((job) => job.id !== CANCELLED_JOB.id);
      return new HttpResponse(null, { status: 204 });
    }),
  );
  renderApp("/jobs");
  const remove = await screen.findByRole("button", { name: "删除" });
  expect(screen.getAllByRole("button", { name: "删除" })).toHaveLength(1);
  await userEvent.click(remove);
  const dialog = screen.getByRole("dialog", { name: "删除已取消任务" });
  expect(dialog).toHaveTextContent("原始数据文件及已有识别结果会保留");
  expect(deletes).toBe(0);
  await userEvent.click(within(dialog).getByRole("button", { name: "返回" }));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.getByText("已取消")).toBeVisible();
  expect(deletes).toBe(0);
  await userEvent.click(remove);
  await userEvent.click(screen.getByRole("button", { name: "确认删除" }));
  await waitFor(() => expect(screen.queryByText("已取消")).not.toBeInTheDocument());
  expect(screen.getByText("失败")).toBeVisible();
  expect(screen.getByText("运行中")).toBeVisible();
  expect(screen.getByRole("status")).toHaveTextContent("已删除 1 个已取消任务");
  expect(screen.getByRole("button", { name: "删除全部已取消任务（0）" })).toBeDisabled();
  expect(deletes).toBe(1);
});

it("deletes all cancelled jobs once while the bulk request is pending", async () => {
  let jobs: Job[] = [CANCELLED_JOB, { ...CANCELLED_JOB, id: "cancelled-job-2" }, FAILED_JOB];
  let deletes = 0;
  let releaseDelete: (() => void) | undefined;
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([])),
    http.get("/api/v1/jobs/page", () => jobsResponse(jobs)),
    http.delete("/api/v1/jobs/cancelled", () => {
      deletes += 1;
      return new Promise((resolve) => {
        releaseDelete = () => {
          jobs = [FAILED_JOB];
          resolve(HttpResponse.json({ deleted_count: 2 }));
        };
      });
    }),
  );
  renderApp("/jobs");
  const bulk = await screen.findByRole("button", { name: "删除全部已取消任务（2）" });
  await userEvent.click(bulk);
  expect(screen.getByRole("dialog")).toHaveTextContent("2 个已取消任务");
  const confirm = screen.getByRole("button", { name: "确认删除" });
  await userEvent.click(confirm);
  await waitFor(() => expect(releaseDelete).toBeDefined());
  expect(confirm).toBeDisabled();
  expect(bulk).toBeDisabled();
  for (const button of screen.getAllByRole("button", { name: "删除" })) {
    expect(button).toBeDisabled();
  }
  await userEvent.click(confirm);
  expect(deletes).toBe(1);
  await act(async () => releaseDelete?.());
  await waitFor(() => expect(screen.queryByText("已取消")).not.toBeInTheDocument());
  expect(screen.getByText("失败")).toBeVisible();
  expect(screen.getByRole("status")).toHaveTextContent("已删除 2 个已取消任务");
});

it.each(["删除", "删除全部已取消任务（1）"])("preserves cancelled jobs and allows another attempt when %s fails", async (buttonName) => {
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([])),
    http.get("/api/v1/jobs/page", () => jobsResponse([CANCELLED_JOB])),
    http.delete("/api/v1/jobs/:jobId", () => HttpResponse.json({ detail: "temporarily unavailable" }, { status: 503 })),
  );
  renderApp("/jobs");
  await userEvent.click(await screen.findByRole("button", { name: buttonName }));
  await userEvent.click(screen.getByRole("button", { name: "确认删除" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("删除任务失败，请重试");
  expect(screen.getByText("已取消")).toBeVisible();
  expect(screen.getByRole("button", { name: "确认删除" })).toBeEnabled();
});

it("refreshes active jobs without overlapping requests and stops when they finish", async () => {
  let requests = 0;
  let releasePoll: (() => void) | undefined;
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([])),
    http.get("/api/v1/jobs/page", () => {
      requests += 1;
      if (requests === 2) {
        return new Promise((resolve) => {
          releasePoll = () => resolve(jobsResponse([{ ...FAILED_JOB, status: "running" }]));
        });
      }
      return jobsResponse([{ ...FAILED_JOB, status: requests === 1 ? "queued" : "succeeded" }]);
    }),
  );
  renderApp("/jobs");
  expect(await screen.findByText("排队")).toBeVisible();
  await waitFor(() => expect(releasePoll).toBeDefined(), { timeout: 4500 });
  await new Promise((resolve) => setTimeout(resolve, 3100));
  expect(requests).toBe(2);
  await act(async () => releasePoll?.());
  expect(await screen.findByText("运行中")).toBeVisible();
  expect(await screen.findByText("成功", {}, { timeout: 4500 })).toBeVisible();
  await new Promise((resolve) => setTimeout(resolve, 3100));
  expect(requests).toBe(3);
}, 16000);

it("aborts a pending job refresh on unmount and stops scheduling polls", async () => {
  let requests = 0;
  let pollSignal: AbortSignal | undefined;
  let releasePoll: (() => void) | undefined;
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([])),
    http.get("/api/v1/jobs/page", ({ request }) => {
      requests += 1;
      if (requests > 1) {
        pollSignal = request.signal;
        return new Promise((resolve) => {
          releasePoll = () => resolve(jobsResponse([{ ...FAILED_JOB, status: "running" }]));
        });
      }
      return jobsResponse([{ ...FAILED_JOB, status: "queued" }]);
    }),
  );
  const rendered = renderApp("/jobs");
  expect(await screen.findByText("排队")).toBeVisible();
  await waitFor(() => expect(pollSignal).toBeDefined(), { timeout: 4500 });
  rendered.unmount();
  expect(pollSignal?.aborted).toBe(true);
  releasePoll?.();
  await new Promise((resolve) => setTimeout(resolve, 3100));
  expect(requests).toBe(2);
}, 10000);

it.each([[0, 0], [0, 1], [4, 10], [1, 1]])(
  "shows a successful job as complete even with saved progress %i/%i",
  async (progress_numerator, progress_denominator) => {
    useJobsHandlers([]);
    server.use(http.get("/api/v1/jobs/page", () => jobsResponse([
      { ...FAILED_JOB, status: "succeeded", progress_numerator, progress_denominator },
    ])));
    renderApp("/jobs");

    expect(await screen.findByText("成功")).toBeVisible();
    const progress = screen.getByRole("progressbar");
    expect(progress).toHaveAttribute("aria-valuenow", "100");
    expect(progress.firstElementChild).toHaveStyle({ width: "100%" });
    expect(progress.firstElementChild).toHaveClass("status-succeeded");
  },
);

it.each(["queued", "running", "failed", "cancelled"] as const)(
  "keeps measured progress and the matching colour for a %s job",
  async (status) => {
    useJobsHandlers([]);
    server.use(http.get("/api/v1/jobs/page", () => jobsResponse([{ ...FAILED_JOB, status }])));
    renderApp("/jobs");

    const progress = await screen.findByRole("progressbar");
    expect(progress).toHaveAttribute("aria-valuenow", "40");
    expect(progress.firstElementChild).toHaveStyle({ width: "40%" });
    expect(progress.firstElementChild).toHaveClass(`status-${status}`);
  },
);

it("shows job progress and retries a failed job", async () => {
  let retried = false;
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([])),
    http.get("/api/v1/jobs/page", () =>
      jobsResponse([
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
    http.get("/api/v1/jobs/page", () =>
      jobsResponse([
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

it("requires confirmation, posts only the selected path once, and disables only the active site", async () => {
  let posts = 0;
  let postedBody: unknown;
  useJobsHandlers([SITE, OTHER_SITE]);
  server.use(
    http.post("/api/v1/sites/site-1/scan", async ({ request }) => {
      posts += 1;
      postedBody = await request.json();
      return HttpResponse.json(scanSnapshot("initial_scan"), { status: 202 });
    }),
  );
  renderApp("/jobs");

  await userEvent.click(
    await screen.findByRole("button", {
      name: "选择目录并扫描 JJK NAS Archive",
    }),
  );
  await userEvent.click(await screen.findByRole("button", { name: "jjk20240608" }));
  await userEvent.click(screen.getByRole("button", { name: "选择当前目录" }));

  expect(posts).toBe(0);
  expect(screen.getByText("扫描范围：JJK NAS Archive / jjk20240608")).toBeVisible();
  expect(screen.queryByText(SITE.data_path)).not.toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "开始扫描" }));

  expect(
    await screen.findByRole("button", { name: "正在扫描 JJK NAS Archive" }),
  ).toBeDisabled();
  expect(screen.getByRole("button", { name: "选择目录并扫描 Secondary Archive" })).toBeEnabled();
  expect(posts).toBe(1);
  expect(postedBody).toEqual({ path: "jjk20240608" });
  await userEvent.click(
    screen.getByRole("button", { name: "正在扫描 JJK NAS Archive" }),
  );
  expect(posts).toBe(1);
});

it("polls each second without overlap, stops at completion, clears storage, and refreshes jobs once", async () => {
  const user = userEvent.setup();
  let jobRequests = 0;
  let statusRequests = 0;
  let concurrentStatusRequests = 0;
  let peakConcurrentStatusRequests = 0;
  let releaseVerification: (() => void) | undefined;
  const snapshots = [
    scanSnapshot("verification_scan"),
    scanSnapshot("completed"),
  ];
  server.use(
    http.get("/api/v1/jobs/page", () => {
      jobRequests += 1;
      return jobsResponse([]);
    }),
    http.get("/api/v1/sites", () => HttpResponse.json([SITE])),
    http.get("/api/v1/sites/site-1/active-scan", () => HttpResponse.json(null)),
    http.get("/api/v1/sites/site-1/scan-directories", ({ request }) =>
      HttpResponse.json(directoryListing(request))),
    http.post("/api/v1/sites/site-1/scan", () =>
      HttpResponse.json(scanSnapshot("stability_wait"), { status: 202 }),
    ),
    http.get("/api/v1/scans/scan-1", async () => {
      statusRequests += 1;
      concurrentStatusRequests += 1;
      peakConcurrentStatusRequests = Math.max(
        peakConcurrentStatusRequests,
        concurrentStatusRequests,
      );
      if (statusRequests === 1) {
        await new Promise<void>((resolve) => {
          releaseVerification = resolve;
        });
      }
      const snapshot = snapshots.shift() ?? scanSnapshot("completed");
      concurrentStatusRequests -= 1;
      return HttpResponse.json(snapshot);
    }),
  );
  renderApp("/jobs");

  await user.click(
    await screen.findByRole("button", {
      name: "选择目录并扫描 JJK NAS Archive",
    }),
  );
  await user.click(await screen.findByRole("button", { name: "jjk20240608" }));
  await user.click(screen.getByRole("button", { name: "选择当前目录" }));
  await user.click(screen.getByRole("button", { name: "开始扫描" }));
  expect(await screen.findByRole("status")).toHaveTextContent("请等待稳定性复检");
  expect(sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY)).toBe(
    JSON.stringify({ "site-1": "scan-1" }),
  );
  expect(statusRequests).toBe(0);

  await waitFor(() => expect(releaseVerification).toBeDefined(), { timeout: 2500 });
  await new Promise((resolve) => setTimeout(resolve, 1100));
  expect(statusRequests).toBe(1);
  expect(peakConcurrentStatusRequests).toBe(1);

  await act(async () => {
    releaseVerification?.();
  });
  expect(await screen.findByRole("status")).toHaveTextContent(
    "正在进行稳定性复检",
  );

  await new Promise((resolve) => setTimeout(resolve, 900));
  expect(statusRequests).toBe(1);

  await waitFor(
    () => expect(screen.getByRole("status")).toHaveTextContent("扫描完成"),
    { timeout: 2000 },
  );
  expect(statusRequests).toBe(2);
  await waitFor(() => expect(jobRequests).toBe(2));
  expect(sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY)).toBeNull();

  await new Promise((resolve) => setTimeout(resolve, 1100));
  expect(statusRequests).toBe(2);
  expect(jobRequests).toBe(2);
}, 10000);

it("restores an active scan returned during initial page load", async () => {
  useJobsHandlers();
  server.use(
    http.get("/api/v1/sites/site-1/active-scan", () =>
      HttpResponse.json(scanSnapshot("initial_scan")),
    ),
  );
  renderApp("/jobs");

  expect(
    await screen.findByRole("button", { name: "正在扫描 JJK NAS Archive" }),
  ).toBeDisabled();
  expect(screen.getByRole("status")).toHaveTextContent(
    "正在扫描：已检查 1250 个文件",
  );
  expect(sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY)).toBe(
    JSON.stringify({ "site-1": "scan-1" }),
  );
});

it("clears a saved missing scan and shows a controlled restart interruption", async () => {
  sessionStorage.setItem(
    ACTIVE_SCAN_STORAGE_KEY,
    JSON.stringify({ "site-1": "missing-scan" }),
  );
  useJobsHandlers();
  server.use(
    http.get("/api/v1/scans/missing-scan", () =>
      HttpResponse.json({ detail: "internal absolute path D:/private/nas/archive" }, { status: 404 }),
    ),
  );
  renderApp("/jobs");

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "扫描因服务重启而中断，请重新扫描",
  );
  expect(sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY)).toBeNull();
  expect(screen.queryByText(SITE.data_path)).not.toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "选择目录并扫描 JJK NAS Archive" }),
  ).toBeEnabled();
});

it("restores directory selection after a failed scan", async () => {
  useJobsHandlers();
  server.use(
    http.post("/api/v1/sites/site-1/scan", () =>
      HttpResponse.json(
        scanSnapshot("failed", { error_message: "扫描目录不可读" }),
        { status: 202 },
      ),
    ),
  );
  renderApp("/jobs");

  await userEvent.click(
    await screen.findByRole("button", {
      name: "选择目录并扫描 JJK NAS Archive",
    }),
  );
  await userEvent.click(await screen.findByRole("button", { name: "jjk20240608" }));
  await userEvent.click(screen.getByRole("button", { name: "选择当前目录" }));
  await userEvent.click(screen.getByRole("button", { name: "开始扫描" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("扫描目录不可读");
  expect(
    screen.getByRole("button", { name: "选择目录并扫描 JJK NAS Archive" }),
  ).toBeEnabled();
  expect(sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY)).toBeNull();
});

it("aborts an in-flight poll and leaves no timer on unmount", async () => {
  const clearTimeout = vi.spyOn(window, "clearTimeout");
  let statusRequests = 0;
  let pollSignal: AbortSignal | undefined;
  let releasePoll: (() => void) | undefined;
  useJobsHandlers();
  server.use(
    http.get("/api/v1/sites/site-1/active-scan", () =>
      HttpResponse.json(scanSnapshot("initial_scan")),
    ),
    http.get("/api/v1/scans/scan-1", ({ request }) => {
      statusRequests += 1;
      pollSignal = request.signal;
      return new Promise((resolve) => {
        releasePoll = () => resolve(HttpResponse.json(scanSnapshot("initial_scan")));
      });
    }),
  );
  const rendered = renderApp("/jobs");

  expect(
    await screen.findByRole("button", { name: "正在扫描 JJK NAS Archive" }),
  ).toBeDisabled();
  await waitFor(() => expect(pollSignal).toBeDefined(), { timeout: 2500 });

  rendered.unmount();

  expect(pollSignal?.aborted).toBe(true);
  expect(clearTimeout).toHaveBeenCalled();
  releasePoll?.();
  await new Promise((resolve) => setTimeout(resolve, 1100));
  expect(statusRequests).toBe(1);
}, 5000);

it("keeps successful initial active-scan recovery when another site lookup fails", async () => {
  server.use(
    http.get("/api/v1/jobs/page", () => jobsResponse([])),
    http.get("/api/v1/sites", () => HttpResponse.json([SITE, OTHER_SITE])),
    http.get("/api/v1/sites/:siteId/active-scan", ({ params }) => {
      if (params.siteId === SITE.id) {
        return HttpResponse.json(scanSnapshot("initial_scan"));
      }
      return HttpResponse.json({ detail: "site temporarily unavailable" }, { status: 503 });
    }),
  );
  renderApp("/jobs");

  expect(
    await screen.findByRole("button", { name: "正在扫描 JJK NAS Archive" }),
  ).toBeDisabled();
  expect(
    screen.getByRole("button", { name: "选择目录并扫描 Secondary Archive" }),
  ).toBeEnabled();
  expect(sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY)).toBe(
    JSON.stringify({ "site-1": "scan-1" }),
  );
});

it("does not let delayed saved-scan recovery overwrite a newly started scan", async () => {
  let jobsRequests = 0;
  let posts = 0;
  let releaseOldScan: (() => void) | undefined;
  sessionStorage.setItem(
    ACTIVE_SCAN_STORAGE_KEY,
    JSON.stringify({ "site-1": "old-scan" }),
  );
  useJobsHandlers();
  server.use(
    http.get("/api/v1/jobs/page", () => {
      jobsRequests += 1;
      return jobsResponse([]);
    }),
    http.get("/api/v1/scans/old-scan", () =>
      new Promise((resolve) => {
        releaseOldScan = () => resolve(HttpResponse.json(
          scanSnapshot("completed", { id: "old-scan" }),
        ));
      }),
    ),
    http.post("/api/v1/sites/site-1/scan", () => {
      posts += 1;
      return HttpResponse.json(
        scanSnapshot("initial_scan", {
          id: "new-scan",
          initial_checked_files: 2000,
        }),
        { status: 202 },
      );
    }),
  );
  renderApp("/jobs");

  await userEvent.click(
    await screen.findByRole("button", {
      name: "选择目录并扫描 JJK NAS Archive",
    }),
  );
  await userEvent.click(await screen.findByRole("button", { name: "jjk20240608" }));
  await userEvent.click(screen.getByRole("button", { name: "选择当前目录" }));
  await userEvent.click(screen.getByRole("button", { name: "开始扫描" }));
  expect(await screen.findByRole("status")).toHaveTextContent("已检查 2000 个文件");
  expect(sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY)).toBe(
    JSON.stringify({ "site-1": "new-scan" }),
  );

  await act(async () => {
    releaseOldScan?.();
    await new Promise((resolve) => setTimeout(resolve, 50));
  });

  expect(screen.getByRole("status")).toHaveTextContent("已检查 2000 个文件");
  expect(sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY)).toBe(
    JSON.stringify({ "site-1": "new-scan" }),
  );
  expect(posts).toBe(1);
  expect(jobsRequests).toBe(1);
});

it("isolates polling failures by site and keeps polling after the failed batch", async () => {
  let primaryPolls = 0;
  let secondaryPolls = 0;
  server.use(
    http.get("/api/v1/jobs/page", () => jobsResponse([])),
    http.get("/api/v1/sites", () => HttpResponse.json([SITE, OTHER_SITE])),
    http.get("/api/v1/sites/:siteId/active-scan", ({ params }) => {
      if (params.siteId === SITE.id) {
        return HttpResponse.json(scanSnapshot("initial_scan"));
      }
      return HttpResponse.json(scanSnapshot("initial_scan", {
        id: "scan-2",
        site_id: OTHER_SITE.id,
      }));
    }),
    http.get("/api/v1/scans/:scanId", ({ params }) => {
      if (params.scanId === "scan-1") {
        primaryPolls += 1;
        if (primaryPolls === 1) {
          return HttpResponse.json({ detail: "temporary poll failure" }, { status: 503 });
        }
        return HttpResponse.json(scanSnapshot("verification_scan"));
      }
      secondaryPolls += 1;
      return HttpResponse.json(scanSnapshot("completed", {
        id: "scan-2",
        site_id: OTHER_SITE.id,
      }));
    }),
  );
  renderApp("/jobs");

  expect(
    await screen.findByRole("button", { name: "正在扫描 JJK NAS Archive" }),
  ).toBeDisabled();
  expect(screen.getByRole("button", { name: "正在扫描 Secondary Archive" })).toBeDisabled();

  expect(
    await screen.findByRole(
      "button",
      { name: "选择目录并扫描 Secondary Archive" },
      { timeout: 2500 },
    ),
  ).toBeEnabled();
  expect(secondaryPolls).toBe(1);
  await waitFor(() => expect(primaryPolls).toBe(2), { timeout: 2500 });
  expect(screen.getByText(/正在进行稳定性复检/)).toBeVisible();
  expect(
    screen.getByRole("button", { name: "正在扫描 JJK NAS Archive" }),
  ).toBeDisabled();
}, 7000);

it("clears an active scan when polling reports it missing", async () => {
  server.use(
    http.get("/api/v1/jobs/page", () => jobsResponse([])),
    http.get("/api/v1/sites", () => HttpResponse.json([SITE])),
    http.get("/api/v1/sites/site-1/active-scan", () =>
      HttpResponse.json(scanSnapshot("initial_scan"))),
    http.get("/api/v1/scans/scan-1", () =>
      HttpResponse.json({ detail: "scan missing after restart" }, { status: 404 })),
  );
  renderApp("/jobs");

  expect(
    await screen.findByRole("button", { name: "正在扫描 JJK NAS Archive" }),
  ).toBeDisabled();
  expect(
    await screen.findByRole(
      "button",
      { name: "选择目录并扫描 JJK NAS Archive" },
      { timeout: 2500 },
    ),
  ).toBeEnabled();
  expect(screen.getByRole("alert")).toHaveTextContent(
    "扫描因服务重启而中断，请重新扫描",
  );
  expect(sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY)).toBeNull();
});

it("allows only one POST while the first scan request is pending", async () => {
  let posts = 0;
  let releasePost: (() => void) | undefined;
  useJobsHandlers();
  server.use(
    http.post("/api/v1/sites/site-1/scan", () => {
      posts += 1;
      return new Promise((resolve) => {
        releasePost = () => resolve(HttpResponse.json(
          scanSnapshot("initial_scan"),
          { status: 202 },
        ));
      });
    }),
  );
  renderApp("/jobs");

  await userEvent.click(
    await screen.findByRole("button", {
      name: "选择目录并扫描 JJK NAS Archive",
    }),
  );
  await userEvent.click(await screen.findByRole("button", { name: "jjk20240608" }));
  await userEvent.click(screen.getByRole("button", { name: "选择当前目录" }));
  const start = screen.getByRole("button", { name: "开始扫描" });
  await userEvent.click(start);
  await waitFor(() => expect(posts).toBe(1));
  expect(start).toBeDisabled();
  await userEvent.click(start);
  expect(posts).toBe(1);

  await act(async () => {
    releasePost?.();
  });
  expect(
    await screen.findByRole("button", { name: "正在扫描 JJK NAS Archive" }),
  ).toBeDisabled();
  expect(posts).toBe(1);
});

it("recovers an existing active scan after a racing scan start fails", async () => {
  let activeScanRequests = 0;
  let releaseInitialActiveScan: (() => void) | undefined;
  const existingScan = scanSnapshot("initial_scan", {
    id: "existing-scan",
    initial_checked_files: 3333,
  });
  useJobsHandlers();
  server.use(
    http.get("/api/v1/sites/site-1/active-scan", () => {
      activeScanRequests += 1;
      if (activeScanRequests === 1) {
        return new Promise((resolve) => {
          releaseInitialActiveScan = () => resolve(HttpResponse.json(existingScan));
        });
      }
      return HttpResponse.json(existingScan);
    }),
    http.post("/api/v1/sites/site-1/scan", () =>
      HttpResponse.json({ detail: "scan already active" }, { status: 503 })),
  );
  renderApp("/jobs");

  await userEvent.click(
    await screen.findByRole("button", {
      name: "选择目录并扫描 JJK NAS Archive",
    }),
  );
  await userEvent.click(await screen.findByRole("button", { name: "jjk20240608" }));
  await userEvent.click(screen.getByRole("button", { name: "选择当前目录" }));
  await userEvent.click(screen.getByRole("button", { name: "开始扫描" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("触发扫描失败");

  await act(async () => {
    releaseInitialActiveScan?.();
  });

  expect(
    await screen.findByRole("button", { name: "正在扫描 JJK NAS Archive" }),
  ).toBeDisabled();
  expect(screen.getByRole("status")).toHaveTextContent("已检查 3333 个文件");
  expect(sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY)).toBe(
    JSON.stringify({ "site-1": "existing-scan" }),
  );
});


it("requests bounded pages and resets to the first page when status or directory changes", async () => {
  const queries: Record<string, string>[] = [];
  let siteRequests = 0;
  let directoryRequests = 0;
  server.use(
    http.get("/api/v1/sites", () => { siteRequests += 1; return HttpResponse.json([SITE]); }),
    http.get("/api/v1/sites/site-1/active-scan", () => HttpResponse.json(null)),
    http.get("/api/v1/jobs/directories", () => {
      directoryRequests += 1;
      return HttpResponse.json([
        { site_id: SITE.id, site_name: SITE.name, directory: "jjk20240608", job_count: 61 },
        { site_id: OTHER_SITE.id, site_name: OTHER_SITE.name, directory: "", job_count: 2 },
      ]);
    }),
    http.get("/api/v1/jobs/page", ({ request }) => {
      const params = new URL(request.url).searchParams;
      queries.push(Object.fromEntries(params));
      const offset = Number(params.get("offset"));
      return jobsResponse([{ ...FAILED_JOB, id: `row-${offset}`, status: "succeeded" }], { total: 61 });
    }),
  );
  renderApp("/jobs");
  expect(await screen.findByText("row-0.h5")).toBeVisible();
  expect(screen.getByText("第 1 / 2 页，共 61 个任务")).toBeVisible();
  expect(screen.getByRole("button", { name: "上一页" })).toBeDisabled();
  expect(queries).toEqual([{ status: "all", limit: "50", offset: "0" }]);

  await userEvent.click(screen.getByRole("button", { name: "下一页" }));
  expect(await screen.findByText("row-50.h5")).toBeVisible();
  expect(screen.getByRole("button", { name: "下一页" })).toBeDisabled();
  expect(queries[1]).toEqual({ status: "all", limit: "50", offset: "50" });
  await userEvent.selectOptions(screen.getByLabelText("任务状态"), "succeeded");
  expect(await screen.findByText("row-0.h5")).toBeVisible();
  expect(queries[2]).toEqual({ status: "succeeded", limit: "50", offset: "0" });
  await userEvent.click(screen.getByRole("button", { name: "下一页" }));
  expect(await screen.findByText("row-50.h5")).toBeVisible();
  await userEvent.selectOptions(screen.getByLabelText("源文件目录"),
    screen.getByRole("option", { name: "Secondary Archive / 根目录（2）" }));
  expect(await screen.findByText("row-0.h5")).toBeVisible();
  expect(queries[4]).toEqual({ status: "succeeded", limit: "50", offset: "0", site_id: "site-2", directory: "" });
  expect(siteRequests).toBe(1);
  expect(directoryRequests).toBe(1);
  expect(screen.queryByText(SITE.data_path)).not.toBeInTheDocument();
});

it("uses the global cancelled count and explains bulk deletion across all directories", async () => {
  useJobsHandlers([]);
  server.use(http.get("/api/v1/jobs/page", () => jobsResponse([FAILED_JOB], {
    total: 1, global_counts: { queued: 0, running: 0, succeeded: 8, failed: 1, cancelled: 72 },
  })));
  renderApp("/jobs");
  await userEvent.click(await screen.findByRole("button", { name: "删除全部已取消任务（72）" }));
  expect(screen.getByRole("dialog")).toHaveTextContent("所有目录");
  expect(screen.getByRole("dialog")).toHaveTextContent("72 个已取消任务");
});

it("returns to the last valid page after its only cancelled job is deleted", async () => {
  let deleted = false;
  const offsets: string[] = [];
  useJobsHandlers([]);
  server.use(
    http.get("/api/v1/jobs/page", ({ request }) => {
      const offset = new URL(request.url).searchParams.get("offset") ?? "";
      offsets.push(offset);
      return jobsResponse(offset === "50" ? (deleted ? [] : [CANCELLED_JOB]) : [FAILED_JOB], {
        total: deleted ? 50 : 51,
        global_counts: { queued: 0, running: 0, succeeded: 0, failed: 50, cancelled: deleted ? 0 : 1 },
      });
    }),
    http.delete("/api/v1/jobs/cancelled-job-1", () => { deleted = true; return new HttpResponse(null, { status: 204 }); }),
  );
  renderApp("/jobs");
  await userEvent.click(await screen.findByRole("button", { name: "下一页" }));
  await userEvent.click(await screen.findByRole("button", { name: "删除" }));
  await userEvent.click(screen.getByRole("button", { name: "确认删除" }));
  expect(await screen.findByText("第 1 / 1 页，共 50 个任务")).toBeVisible();
  expect(screen.getByText("job-failed-1.h5")).toBeVisible();
  expect(offsets).toEqual(["0", "50", "50", "0"]);
});

it("keeps polling when active jobs are outside the selected page and does not reload directories", async () => {
  let filteredRequests = 0;
  let directoryRequests = 0;
  useJobsHandlers([]);
  server.use(
    http.get("/api/v1/jobs/directories", () => { directoryRequests += 1; return HttpResponse.json([]); }),
    http.get("/api/v1/jobs/page", ({ request }) => {
      if (new URL(request.url).searchParams.get("status") === "succeeded") filteredRequests += 1;
      return jobsResponse([{ ...FAILED_JOB, id: `finished-${filteredRequests}`, status: "succeeded" }], {
        global_counts: { queued: 1, running: 0, succeeded: 20, failed: 0, cancelled: 0 },
      });
    }),
  );
  renderApp("/jobs");
  await userEvent.selectOptions(await screen.findByLabelText("任务状态"), "succeeded");
  expect(await screen.findByText("finished-1.h5")).toBeVisible();
  expect(await screen.findByText("finished-2.h5", {}, { timeout: 4500 })).toBeVisible();
  expect(directoryRequests).toBe(1);
}, 6000);

it("ignores a delayed old filter response without aborting scan recovery", async () => {
  let releaseOld: (() => void) | undefined;
  let releaseScan: (() => void) | undefined;
  let scanSignal: AbortSignal | undefined;
  useJobsHandlers();
  server.use(
    http.get("/api/v1/sites/site-1/active-scan", ({ request }) => {
      scanSignal = request.signal;
      return new Promise((resolve) => {
        releaseScan = () => resolve(HttpResponse.json(scanSnapshot("initial_scan")));
      });
    }),
    http.get("/api/v1/jobs/page", ({ request }) => {
      const status = new URL(request.url).searchParams.get("status");
      if (status === "failed") return new Promise((resolve) => {
        releaseOld = () => resolve(jobsResponse([{ ...FAILED_JOB, id: "stale-failed" }]));
      });
      return jobsResponse([{ ...FAILED_JOB, id: `current-${status}`, status: "succeeded" }]);
    }),
  );
  renderApp("/jobs");
  await userEvent.selectOptions(await screen.findByLabelText("任务状态"), "failed");
  await waitFor(() => expect(releaseOld).toBeDefined());
  await userEvent.selectOptions(screen.getByLabelText("任务状态"), "succeeded");
  expect(await screen.findByText("current-succeeded.h5")).toBeVisible();
  expect(scanSignal?.aborted).toBe(false);
  await act(async () => { releaseOld?.(); releaseScan?.(); });
  expect(await screen.findByRole("button", { name: "正在扫描 JJK NAS Archive" })).toBeDisabled();
  expect(screen.getByText("current-succeeded.h5")).toBeVisible();
  expect(screen.queryByText("stale-failed.h5")).not.toBeInTheDocument();
});


it("shows an initial page failure and lets the user retry loading", async () => {
  let failed = true;
  useJobsHandlers([]);
  server.use(http.get("/api/v1/jobs/page", () => failed
    ? HttpResponse.json({ detail: "temporarily unavailable" }, { status: 503 })
    : jobsResponse([FAILED_JOB])));
  renderApp("/jobs");
  expect(await screen.findByRole("alert")).toHaveTextContent("无法加载任务");
  failed = false;
  await userEvent.click(screen.getByRole("button", { name: "刷新任务" }));
  expect(await screen.findByText("job-failed-1.h5")).toBeVisible();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("keeps a selected directory visible as empty when its final cancelled task is deleted", async () => {
  let deleted = false;
  const directory = { site_id: SITE.id, site_name: SITE.name, directory: "jjk20240608", job_count: 1 };
  useJobsHandlers([]);
  server.use(
    http.get("/api/v1/jobs/directories", () => HttpResponse.json(deleted ? [] : [directory])),
    http.get("/api/v1/jobs/page", () => jobsResponse(deleted ? [] : [CANCELLED_JOB])),
    http.delete("/api/v1/jobs/cancelled-job-1", () => { deleted = true; return new HttpResponse(null, { status: 204 }); }),
  );
  renderApp("/jobs");
  await userEvent.selectOptions(await screen.findByLabelText("源文件目录"),
    await screen.findByRole("option", { name: "JJK NAS Archive / jjk20240608（1）" }));
  await userEvent.click(await screen.findByRole("button", { name: "删除" }));
  await userEvent.click(screen.getByRole("button", { name: "确认删除" }));
  expect(await screen.findByText("当前筛选下没有任务。")).toBeVisible();
  expect(screen.getByRole("option", { name: "JJK NAS Archive / jjk20240608（0）" })).toBeVisible();
  expect(screen.getByLabelText("源文件目录")).toHaveValue(JSON.stringify({ site_id: SITE.id, directory: "jjk20240608" }));
});


it("reloads directory options when the user refreshes after the initial directory request fails", async () => {
  let requests = 0;
  useJobsHandlers([]);
  server.use(http.get("/api/v1/jobs/directories", () => {
    requests += 1;
    return requests === 1
      ? HttpResponse.json({ detail: "temporarily unavailable" }, { status: 503 })
      : HttpResponse.json([{ site_id: SITE.id, site_name: SITE.name, directory: "jjk20240608", job_count: 3 }]);
  }));
  renderApp("/jobs");
  expect(await screen.findByRole("alert")).toHaveTextContent("无法加载任务目录");
  await userEvent.click(screen.getByRole("button", { name: "刷新任务" }));
  expect(await screen.findByRole("option", { name: "JJK NAS Archive / jjk20240608（3）" })).toBeVisible();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(requests).toBe(2);
});

it.each(["page", "directory"] as const)(
  "hides old task actions while a new %s is loading or failed and allows retry",
  async (change) => {
    let fail = true;
    let releaseFailure: (() => void) | undefined;
    useJobsHandlers([]);
    server.use(
      http.get("/api/v1/jobs/directories", () => HttpResponse.json([
        { site_id: SITE.id, site_name: SITE.name, directory: "other-folder", job_count: 51 },
      ])),
      http.get("/api/v1/jobs/page", ({ request }) => {
        const params = new URL(request.url).searchParams;
        if (params.get("offset") === "50" || params.has("site_id")) {
          if (fail) return new Promise((resolve) => {
            releaseFailure = () => resolve(HttpResponse.json({ detail: "unavailable" }, { status: 503 }));
          });
          return jobsResponse([{ ...FAILED_JOB, id: "new-page-job", status: "succeeded" }], { total: 51 });
        }
        return jobsResponse([FAILED_JOB, CANCELLED_JOB, { ...FAILED_JOB, id: "old-queued", status: "queued" }], { total: 51 });
      }),
    );
    renderApp("/jobs");
    expect(await screen.findByText("job-failed-1.h5")).toBeVisible();
    if (change === "page") {
      await userEvent.click(screen.getByRole("button", { name: "下一页" }));
    } else {
      await userEvent.selectOptions(screen.getByLabelText("源文件目录"),
        await screen.findByRole("option", { name: "JJK NAS Archive / other-folder（51）" }));
    }
    await waitFor(() => expect(releaseFailure).toBeDefined());
    expect(screen.queryByText("job-failed-1.h5")).not.toBeInTheDocument();
    for (const name of ["重试", "取消", "删除"]) {
      expect(screen.queryByRole("button", { name })).not.toBeInTheDocument();
    }
    expect(screen.getByText("正在加载任务…")).toBeVisible();
    expect(screen.getByText(/全部任务：排队 1/)).toBeVisible();
    await act(async () => releaseFailure?.());
    expect(await screen.findByRole("alert")).toHaveTextContent("无法加载任务");
    expect(screen.queryByText("job-failed-1.h5")).not.toBeInTheDocument();
    expect(screen.queryByText("第 1 / 2 页，共 51 个任务")).not.toBeInTheDocument();
    expect(screen.queryByText("第 2 / 2 页，共 51 个任务")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "下一页" })).toBeDisabled();
    fail = false;
    await userEvent.click(screen.getByRole("button", { name: "刷新任务" }));
    expect(await screen.findByText("new-page-job.h5")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  },
);
