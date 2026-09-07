import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, type Job, type ScanStatus, type Site } from "../api";
import ScanDirectoryDialog from "../components/ScanDirectoryDialog";
import ScanStatusCard, { isActiveScan } from "../components/ScanStatusCard";

const ACTIVE_SCAN_STORAGE_KEY = "das-sentinel.active-scans";

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
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [sites, setSites] = useState<Site[]>([]);
  const [scans, setScans] = useState<Record<string, ScanStatus>>({});
  const [choosingSite, setChoosingSite] = useState<Site | null>(null);
  const [pendingScan, setPendingScan] = useState<{ site: Site; path: string } | null>(null);
  const [startingSiteIds, setStartingSiteIds] = useState<Set<string>>(() => new Set());
  const [error, setError] = useState("");
  const mounted = useRef(true);
  const requestControllers = useRef(new Set<AbortController>());
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
    const controller = requestController();
    api
      .get<Job[]>("/api/v1/jobs", undefined, controller.signal)
      .then((response) => {
        if (!controller.signal.aborted) setJobs(response);
      })
      .catch(() => {
        if (!controller.signal.aborted) setError("无法加载任务");
      })
      .finally(() => releaseController(controller));
  }, [releaseController, requestController]);

  useEffect(() => {
    mounted.current = true;
    refresh();
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
                refresh();
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
  }, [refresh, releaseController, requestController]);

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
          if (terminalCount > 0) refresh();
        })
        .finally(() => {
          if (controller !== null) releaseController(controller);
        });
    }, 1000);
    return () => {
      window.clearTimeout(timer);
      controller?.abort();
    };
  }, [refresh, releaseController, requestController, scans]);

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

  async function startScan(site: Site, path: string) {
    if (
      startingSiteIdsRef.current.has(site.id)
      || (scans[site.id] !== undefined && isActiveScan(scans[site.id]))
    ) {
      return;
    }
    startingSiteIdsRef.current.add(site.id);
    siteScanGenerations.current[site.id] =
      (siteScanGenerations.current[site.id] ?? 0) + 1;
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
        refresh();
      }
    } catch {
      if (!controller.signal.aborted) setError("触发扫描失败");
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

  if (jobs === null) {
    return <p className="loading">正在加载…</p>;
  }
  return (
    <section>
      <h1>任务</h1>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
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
      <table className="data">
        <thead>
          <tr>
            <th>任务 ID</th>
            <th>状态</th>
            <th>进度</th>
            <th>尝试次数</th>
            <th>错误</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => {
            const total = job.progress_denominator || 0;
            const percent =
              total > 0
                ? Math.min(100, (job.progress_numerator / total) * 100)
                : 0;
            return (
              <tr key={job.id}>
                <td>{job.id.slice(0, 8)}</td>
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
                    <div className="progress-fill" style={{ width: `${percent}%` }} />
                  </div>
                </td>
                <td>{job.attempt_count}</td>
                <td>{job.error_code ?? ""}</td>
                <td>
                  {job.status === "failed" && (
                    <button
                      className="subtle"
                      type="button"
                      onClick={() => void act(job, "retry")}
                    >
                      重试
                    </button>
                  )}{" "}
                  {(job.status === "queued" || job.status === "running") && (
                    <button
                      className="danger"
                      type="button"
                      onClick={() => void act(job, "cancel")}
                    >
                      取消
                    </button>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
