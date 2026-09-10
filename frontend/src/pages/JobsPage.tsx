import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, type Job, type JobDirectory, type JobPage, type ScanStatus, type Site } from "../api";
import ScanDirectoryDialog from "../components/ScanDirectoryDialog";
import ScanStatusCard, { isActiveScan } from "../components/ScanStatusCard";

const ACTIVE_SCAN_STORAGE_KEY = "das-sentinel.active-scans";
const PAGE_SIZE = 50;

type JobFilter = Job["status"] | "all" | "active";

const STATUS_NAMES: Record<Job["status"], string> = {
  queued: "排队",
  running: "运行中",
  succeeded: "成功",
  failed: "失败",
  cancelled: "已取消",
};

function savedScanIds(): Record<string, string> {
  const value = sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY);
  if (value === null) return {};
  try {
    const parsed = JSON.parse(value) as unknown;
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, string] => typeof entry[1] === "string",
      ),
    );
  } catch {
    return {};
  }
}

function rememberActiveScan(siteId: string, scanId: string): void {
  sessionStorage.setItem(
    ACTIVE_SCAN_STORAGE_KEY,
    JSON.stringify({ ...savedScanIds(), [siteId]: scanId }),
  );
}

function forgetActiveScan(siteId: string, expectedScanId?: string): void {
  const saved = savedScanIds();
  if (expectedScanId !== undefined && saved[siteId] !== expectedScanId) return;
  delete saved[siteId];
  if (Object.keys(saved).length === 0) {
    sessionStorage.removeItem(ACTIVE_SCAN_STORAGE_KEY);
  } else {
    sessionStorage.setItem(ACTIVE_SCAN_STORAGE_KEY, JSON.stringify(saved));
  }
}

export default function JobsPage() {
  const [jobPage, setJobPage] = useState<JobPage | null>(null);
  const [query, setQuery] = useState({ status: "all" as JobFilter, directory: "", offset: 0 });
  const [loadedQuery, setLoadedQuery] = useState<typeof query | null>(null);
  const queryRef = useRef(query);
  queryRef.current = query;
  const [directories, setDirectories] = useState<JobDirectory[]>([]);
  const [directoryError, setDirectoryError] = useState("");
  const [sites, setSites] = useState<Site[]>([]);
  const [scans, setScans] = useState<Record<string, ScanStatus>>({});
  const [choosingSite, setChoosingSite] = useState<Site | null>(null);
  const [pendingScan, setPendingScan] = useState<{ site: Site; path: string } | null>(null);
  const [startingSiteIds, setStartingSiteIds] = useState<Set<string>>(() => new Set());
  const [refreshing, setRefreshing] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<Job | "all" | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const mounted = useRef(true);
  const requestControllers = useRef(new Set<AbortController>());
  const jobsController = useRef<AbortController | null>(null);
  const directoriesController = useRef<AbortController | null>(null);
  const deletingRef = useRef(false);
  const startingSiteIdsRef = useRef(new Set<string>());
  const siteScanGenerations = useRef<Record<string, number>>({});

  const requestController = useCallback(() => {
    const controller = new AbortController();
    requestControllers.current.add(controller);
    return controller;
  }, []);

  const releaseController = useCallback((controller: AbortController) => {
    requestControllers.current.delete(controller);
  }, []);

  const refresh = useCallback(() => {
    jobsController.current?.abort();
    const controller = requestController();
    jobsController.current = controller;
    setRefreshing(true);
    const requestedQuery = queryRef.current;
    const params: Record<string, string> = {
      status: requestedQuery.status,
      limit: String(PAGE_SIZE),
      offset: String(requestedQuery.offset),
    };
    if (requestedQuery.directory !== "") {
      const selected = JSON.parse(requestedQuery.directory) as { site_id: string; directory: string };
      params.site_id = selected.site_id;
      params.directory = selected.directory;
    }
    return api
      .get<JobPage>("/api/v1/jobs/page", params, controller.signal)
      .then((response) => {
        if (!controller.signal.aborted && queryRef.current === requestedQuery) {
          const lastOffset = Math.max(0, Math.ceil(response.total / PAGE_SIZE) - 1) * PAGE_SIZE;
          if (requestedQuery.offset > lastOffset) {
            setQuery((current) => current === requestedQuery ? { ...current, offset: lastOffset } : current);
            return;
          }
          setJobPage(response);
          setLoadedQuery(requestedQuery);
          setError((current) => current === "无法加载任务" ? "" : current);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted && queryRef.current === requestedQuery) setError("无法加载任务");
      })
      .finally(() => {
        releaseController(controller);
        if (jobsController.current === controller) {
          jobsController.current = null;
          if (mounted.current) setRefreshing(false);
        }
      });
  }, [releaseController, requestController]);

  const refreshDirectories = useCallback(() => {
    directoriesController.current?.abort();
    const controller = requestController();
    directoriesController.current = controller;
    return api.get<JobDirectory[]>("/api/v1/jobs/directories", undefined, controller.signal)
      .then((response) => {
        if (controller.signal.aborted) return;
        setDirectories((current) => {
          const selected = current.find((entry) => JSON.stringify({
            site_id: entry.site_id, directory: entry.directory,
          }) === queryRef.current.directory);
          return selected && !response.some((entry) => (
            entry.site_id === selected.site_id && entry.directory === selected.directory
          )) ? [...response, { ...selected, job_count: 0 }] : response;
        });
        setDirectoryError("");
      })
      .catch(() => {
        if (!controller.signal.aborted) setDirectoryError("无法加载任务目录");
      })
      .finally(() => {
        releaseController(controller);
        if (directoriesController.current === controller) directoriesController.current = null;
      });
  }, [releaseController, requestController]);

  const refreshJobsAndDirectories = useCallback(() => {
    void refresh();
    void refreshDirectories();
  }, [refresh, refreshDirectories]);

  const hasActiveJobs = (jobPage?.global_counts.queued ?? 0) + (jobPage?.global_counts.running ?? 0) > 0;

  useEffect(() => {
    void refresh();
    return () => jobsController.current?.abort();
  }, [query, refresh]);

  useEffect(() => {
    if (!hasActiveJobs) return;
    let stopped = false;
    let timer: number;
    const poll = async () => {
      await refresh();
      if (!stopped) timer = window.setTimeout(() => void poll(), 3000);
    };
    timer = window.setTimeout(() => void poll(), 3000);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [hasActiveJobs, query, refresh]);

  useEffect(() => {
    mounted.current = true;
    void refreshDirectories();
    const controller = requestController();
    const restoreScans = async () => {
      try {
        const loadedSites = await api.get<Site[]>(
          "/api/v1/sites",
          undefined,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setSites(loadedSites);
        const saved = savedScanIds();
        await Promise.allSettled(loadedSites.map(async (site) => {
          const generation = siteScanGenerations.current[site.id] ?? 0;
          const isCurrent = () => (
            !controller.signal.aborted
            && (siteScanGenerations.current[site.id] ?? 0) === generation
          );
          try {
            const active = await api.get<ScanStatus | null>(
              `/api/v1/sites/${site.id}/active-scan`,
              undefined,
              controller.signal,
            );
            if (!isCurrent()) return;
            if (active !== null) {
              rememberActiveScan(site.id, active.id);
              setScans((current) => isCurrent()
                ? { ...current, [site.id]: active }
                : current);
              return;
            }
            const savedId = saved[site.id];
            if (savedId === undefined) return;
            try {
              const prior = await api.get<ScanStatus>(
                `/api/v1/scans/${savedId}`,
                undefined,
                controller.signal,
              );
              if (!isCurrent()) return;
              setScans((current) => isCurrent()
                ? { ...current, [site.id]: prior }
                : current);
              if (isActiveScan(prior)) {
                rememberActiveScan(site.id, prior.id);
              } else {
                forgetActiveScan(site.id, savedId);
                refreshJobsAndDirectories();
              }
            } catch (cause: unknown) {
              if (!isCurrent()) return;
              if (cause instanceof ApiError && cause.status === 404) {
                forgetActiveScan(site.id, savedId);
                setError("扫描因服务重启而中断，请重新扫描");
                return;
              }
              throw cause;
            }
          } catch (cause: unknown) {
            if (controller.signal.aborted) return;
            if (cause instanceof ApiError && cause.status === 404) {
              forgetActiveScan(site.id);
              if (isCurrent()) {
                setError("扫描因服务重启而中断，请重新扫描");
              }
            } else {
              throw cause;
            }
          }
        }));
      } catch {
        // The jobs table remains usable if site scan controls cannot be loaded.
      } finally {
        releaseController(controller);
      }
    };
    void restoreScans();
    return () => {
      mounted.current = false;
      for (const activeController of requestControllers.current) {
        activeController.abort();
      }
      requestControllers.current.clear();
    };
  }, [refreshJobsAndDirectories, refreshDirectories, releaseController, requestController]);

  useEffect(() => {
    const activeScans = Object.values(scans).filter(isActiveScan);
    if (activeScans.length === 0) return;
    let controller: AbortController | null = null;
    const timer = window.setTimeout(() => {
      controller = requestController();
      const generations = activeScans.map(
        (scan) => siteScanGenerations.current[scan.site_id] ?? 0,
      );
      Promise.allSettled(
        activeScans.map((scan) =>
          api.get<ScanStatus>(
            `/api/v1/scans/${scan.id}`,
            undefined,
            controller?.signal,
          ),
        ),
      )
        .then((results) => {
          if (controller?.signal.aborted) return;
          const applicableResults = results.flatMap((result, index) => {
            const polled = activeScans[index];
            if (
              (siteScanGenerations.current[polled.site_id] ?? 0) !== generations[index]
              || scans[polled.site_id]?.id !== polled.id
            ) {
              return [];
            }
            return [{ generation: generations[index], polled, result }];
          });
          let terminalCount = 0;
          for (const { polled, result } of applicableResults) {
            if (result.status === "fulfilled" && !isActiveScan(result.value)) {
              forgetActiveScan(polled.site_id, polled.id);
              terminalCount += 1;
            } else if (
              result.status === "rejected"
              && result.reason instanceof ApiError
              && result.reason.status === 404
            ) {
              forgetActiveScan(polled.site_id, polled.id);
              setError("扫描因服务重启而中断，请重新扫描");
            }
          }
          setScans((current) => {
            const next = { ...current };
            let changed = false;
            applicableResults.forEach(({ generation, polled, result }) => {
              if (
                (siteScanGenerations.current[polled.site_id] ?? 0) !== generation
                || current[polled.site_id]?.id !== polled.id
              ) {
                return;
              }
              changed = true;
              if (result.status === "fulfilled") {
                next[polled.site_id] = result.value;
              } else if (result.reason instanceof ApiError && result.reason.status === 404) {
                delete next[polled.site_id];
              }
            });
            if (!changed) {
              return current;
            }
            return next;
          });
          if (terminalCount > 0) refreshJobsAndDirectories();
        })
        .finally(() => {
          if (controller !== null) releaseController(controller);
        });
    }, 1000);
    return () => {
      window.clearTimeout(timer);
      controller?.abort();
    };
  }, [refreshJobsAndDirectories, releaseController, requestController, scans]);

  async function act(job: Job, action: "retry" | "cancel") {
    const controller = requestController();
    try {
      await api.post(`/api/v1/jobs/${job.id}/${action}`, undefined, controller.signal);
      refresh();
    } catch {
      if (!controller.signal.aborted) setError("任务操作失败");
    } finally {
      releaseController(controller);
    }
  }

  async function deleteCancelled() {
    if (pendingDelete === null || deletingRef.current) return;
    if (pendingDelete !== "all" && pendingDelete.status !== "cancelled") return;
    deletingRef.current = true;
    setDeleting(true);
    setError("");
    setNotice("");
    const controller = requestController();
    try {
      let deletedCount = 1;
      if (pendingDelete === "all") {
        const response = await api.delete<{ deleted_count: number }>(
          "/api/v1/jobs/cancelled",
          controller.signal,
        );
        deletedCount = response.deleted_count;
      } else {
        await api.delete(`/api/v1/jobs/${pendingDelete.id}`, controller.signal);
      }
      if (controller.signal.aborted) return;
      setPendingDelete(null);
      setNotice(`已删除 ${deletedCount} 个已取消任务`);
      await Promise.all([refresh(), refreshDirectories()]);
    } catch {
      if (!controller.signal.aborted) setError("删除任务失败，请重试");
    } finally {
      releaseController(controller);
      deletingRef.current = false;
      if (mounted.current) setDeleting(false);
    }
  }

  async function startScan(site: Site, path: string) {
    if (
      startingSiteIdsRef.current.has(site.id)
      || (scans[site.id] !== undefined && isActiveScan(scans[site.id]))
    ) {
      return;
    }
    startingSiteIdsRef.current.add(site.id);
    const generation = (siteScanGenerations.current[site.id] ?? 0) + 1;
    siteScanGenerations.current[site.id] = generation;
    setStartingSiteIds((current) => new Set(current).add(site.id));
    const controller = requestController();
    try {
      const scan = await api.post<ScanStatus>(
        `/api/v1/sites/${site.id}/scan`,
        { path },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setScans((current) => ({ ...current, [site.id]: scan }));
      setPendingScan(null);
      if (isActiveScan(scan)) {
        rememberActiveScan(site.id, scan.id);
      } else {
        forgetActiveScan(site.id);
        refreshJobsAndDirectories();
      }
    } catch {
      if (!controller.signal.aborted) {
        setError("触发扫描失败");
        try {
          const active = await api.get<ScanStatus | null>(
            `/api/v1/sites/${site.id}/active-scan`,
            undefined,
            controller.signal,
          );
          if (
            !controller.signal.aborted
            && siteScanGenerations.current[site.id] === generation
            && active !== null
          ) {
            setScans((current) => ({ ...current, [site.id]: active }));
            setPendingScan(null);
            rememberActiveScan(site.id, active.id);
          }
        } catch {
          // Keep the original start failure visible if reconciliation is unavailable.
        }
      }
    } finally {
      releaseController(controller);
      startingSiteIdsRef.current.delete(site.id);
      if (mounted.current) {
        setStartingSiteIds((current) => {
          const next = new Set(current);
          next.delete(site.id);
          return next;
        });
      }
    }
  }

  if (jobPage === null) {
    if (error) return <section>
      <h1>任务</h1>
      <p className="error" role="alert">{error}</p>
      <button className="subtle" type="button" disabled={refreshing} onClick={refreshJobsAndDirectories}>
        刷新任务
      </button>
    </section>;
    return <p className="loading">正在加载…</p>;
  }
  const isCurrentPage = loadedQuery === query;
  const cancelledCount = jobPage.global_counts.cancelled;
  const pageCount = Math.max(1, Math.ceil(jobPage.total / PAGE_SIZE));
  return (
    <section>
      <h1>任务</h1>
      {error && pendingDelete === null && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {notice && <p role="status">{notice}</p>}
      <p className="job-summary">
        全部任务：排队 {jobPage.global_counts.queued} · 运行中 {jobPage.global_counts.running}
        {" · "}成功 {jobPage.global_counts.succeeded} · 失败 {jobPage.global_counts.failed}
        {" · "}已取消 {cancelledCount}
      </p>
      <div className="job-filters">
        <label>
          任务状态
          <select value={query.status} onChange={(event) => setQuery((current) => ({
            ...current, status: event.target.value as JobFilter, offset: 0,
          }))}>
            <option value="all">全部状态</option>
            <option value="active">进行中任务</option>
            <option value="queued">排队任务</option>
            <option value="running">运行中任务</option>
            <option value="succeeded">成功任务</option>
            <option value="failed">失败任务</option>
            <option value="cancelled">已取消任务</option>
          </select>
        </label>
        <label>
          源文件目录
          <select value={query.directory} onChange={(event) => setQuery((current) => ({
            ...current, directory: event.target.value, offset: 0,
          }))}>
            <option value="">全部目录</option>
            {directories.map((entry) => {
              const value = JSON.stringify({ site_id: entry.site_id, directory: entry.directory });
              return <option key={value} value={value}>
                {entry.site_name} / {entry.directory || "根目录"}（{entry.job_count}）
              </option>;
            })}
          </select>
        </label>
      </div>
      {directoryError && <p className="error" role="alert">{directoryError}</p>}
      <div className="scan-controls">
        <button className="subtle" type="button" disabled={refreshing || deleting} onClick={refreshJobsAndDirectories}>
          刷新任务
        </button>{" "}
        <button
          className="danger"
          type="button"
          disabled={cancelledCount === 0 || deleting}
          onClick={() => {
            setError("");
            setPendingDelete("all");
          }}
        >
          删除全部已取消任务（{cancelledCount}）
        </button>
      </div>
      {hasActiveJobs && <p>排队和运行中的任务每 3 秒自动刷新。</p>}
      {sites.length > 0 && (
        <div className="scan-controls">
          {sites.map((site) => {
            const scan = scans[site.id];
            const active = startingSiteIds.has(site.id)
              || (scan !== undefined && isActiveScan(scan));
            return (
              <div className="scan-site-control" key={site.id}>
                <button
                  className="subtle"
                  type="button"
                  disabled={active}
                  onClick={() => {
                    setError("");
                    setPendingScan(null);
                    setChoosingSite(site);
                  }}
                >
                  {active
                    ? `正在扫描 ${site.name}`
                    : `选择目录并扫描 ${site.name}`}
                </button>
                {pendingScan?.site.id === site.id && (
                  <div className="scan-scope">
                    <p>扫描范围：{site.name} / {pendingScan.path}</p>
                    <button
                      className="primary"
                      type="button"
                      disabled={active}
                      onClick={() => void startScan(site, pendingScan.path)}
                    >
                      开始扫描
                    </button>
                  </div>
                )}
                {scan !== undefined && <ScanStatusCard scan={scan} />}
              </div>
            );
          })}
        </div>
      )}
      {choosingSite !== null && (
        <ScanDirectoryDialog
          site={choosingSite}
          onCancel={() => setChoosingSite(null)}
          onSelect={(path) => {
            if (path) setPendingScan({ site: choosingSite, path });
            setChoosingSite(null);
          }}
        />
      )}
      <div className="job-pagination" aria-label="任务分页">
        <button className="subtle" type="button" disabled={query.offset === 0 || refreshing || deleting}
          onClick={() => setQuery((current) => ({ ...current, offset: current.offset - PAGE_SIZE }))}>
          上一页
        </button>
        {isCurrentPage
          ? <span>第 {query.offset / PAGE_SIZE + 1} / {pageCount} 页，共 {jobPage.total} 个任务</span>
          : <span>第 {query.offset / PAGE_SIZE + 1} 页{refreshing ? "加载中…" : "尚未加载"}</span>}
        <button className="subtle" type="button" disabled={!isCurrentPage || query.offset + PAGE_SIZE >= jobPage.total || refreshing || deleting}
          onClick={() => setQuery((current) => ({ ...current, offset: current.offset + PAGE_SIZE }))}>
          下一页
        </button>
      </div>
      <table className="data jobs-table" aria-busy={refreshing}>
        <thead>
          <tr>
            <th>任务 ID</th>
            <th>源文件 / 目录</th>
            <th>状态</th>
            <th>进度</th>
            <th>尝试次数</th>
            <th>错误</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {!isCurrentPage && <tr><td colSpan={7}>
            {refreshing ? "正在加载任务…" : "当前筛选的任务尚未加载，请刷新重试。"}
          </td></tr>}
          {isCurrentPage && jobPage.items.length === 0 && <tr><td colSpan={7}>当前筛选下没有任务。</td></tr>}
          {isCurrentPage && jobPage.items.map((job) => {
            const total = job.progress_denominator || 0;
            const percent = job.status === "succeeded"
              ? 100
              : total > 0
                ? Math.min(100, (job.progress_numerator / total) * 100)
                : 0;
            return (
              <tr key={job.id}>
                <td>{job.id.slice(0, 8)}</td>
                <td className="job-source">
                  <div title={job.source_filename}>{job.source_filename}</div>
                  <small title={`${job.site_name} / ${job.source_directory || "根目录"}`}>
                    {job.site_name} / {job.source_directory || "根目录"}
                  </small>
                </td>
                <td>
                  <span className={`status-pill status-${job.status}`}>
                    {STATUS_NAMES[job.status]}
                  </span>
                </td>
                <td>
                  <div
                    className="progress-track"
                    role="progressbar"
                    aria-valuenow={Math.round(percent)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  >
                    <div className={`progress-fill status-${job.status}`} style={{ width: `${percent}%` }} />
                  </div>
                </td>
                <td>{job.attempt_count}</td>
                <td>{job.error_code ?? ""}</td>
                <td>
                  {job.status === "failed" && (
                    <button
                      className="subtle"
                      type="button"
                      disabled={deleting}
                      onClick={() => void act(job, "retry")}
                    >
                      重试
                    </button>
                  )}{" "}
                  {(job.status === "queued" || job.status === "running") && (
                    <button
                      className="danger"
                      type="button"
                      disabled={deleting}
                      onClick={() => void act(job, "cancel")}
                    >
                      取消
                    </button>
                  )}
                  {job.status === "cancelled" && (
                    <button
                      className="danger"
                      type="button"
                      disabled={deleting}
                      onClick={() => {
                        setError("");
                        setPendingDelete(job);
                      }}
                    >
                      删除
                    </button>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {pendingDelete !== null && (
        <div className="dialog-backdrop">
          <div className="dialog" role="dialog" aria-modal="true" aria-label="删除已取消任务">
            <p>
              {pendingDelete === "all"
                ? `确认删除所有目录中的全部 ${cancelledCount} 个已取消任务？`
                : `确认删除已取消任务 ${pendingDelete.id.slice(0, 8)}？`}
            </p>
            <p>仅删除已取消的任务记录，原始数据文件及已有识别结果会保留。</p>
            {error && <p className="error" role="alert">{error}</p>}
            <button className="danger" type="button" disabled={deleting} onClick={() => void deleteCancelled()}>
              确认删除
            </button>{" "}
            <button className="subtle" type="button" disabled={deleting} onClick={() => setPendingDelete(null)}>
              返回
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
