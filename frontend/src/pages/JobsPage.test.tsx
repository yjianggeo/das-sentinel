import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { transferableAbortController } from "node:util";
import { afterAll, afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import type { ScanPhase, ScanStatus, Site } from "../api";
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
    http.get("/api/v1/jobs", () => HttpResponse.json([])),
    http.get("/api/v1/sites", () => HttpResponse.json(sites)),
    http.get("/api/v1/sites/:siteId/active-scan", () => HttpResponse.json(null)),
    http.get("/api/v1/sites/:siteId/scan-directories", ({ request }) =>
      HttpResponse.json(directoryListing(request))),
  );
}

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
    http.get("/api/v1/jobs", () => {
      jobRequests += 1;
      return HttpResponse.json([]);
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
    http.get("/api/v1/jobs", () => HttpResponse.json([])),
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
    http.get("/api/v1/jobs", () => {
      jobsRequests += 1;
      return HttpResponse.json([]);
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
    http.get("/api/v1/jobs", () => HttpResponse.json([])),
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
    http.get("/api/v1/jobs", () => HttpResponse.json([])),
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
