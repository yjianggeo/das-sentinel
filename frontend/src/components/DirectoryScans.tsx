import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ApiError, api, type ScanStatus, type Site } from "../api";
import { formatScanDistance } from "../scanDistance";
import ScanDirectoryDialog from "./ScanDirectoryDialog";
import ScanStatusCard, { isActiveScan } from "./ScanStatusCard";

const ACTIVE_SCAN_STORAGE_KEY = "das-sentinel.active-scans";

function savedScanIds(): Record<string, string> {
  const value = sessionStorage.getItem(ACTIVE_SCAN_STORAGE_KEY);
  if (value === null) return {};
  try {
    const parsed = JSON.parse(value) as unknown;
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};
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

type Props = { onComplete: () => void };

export default function DirectoryScans({ onComplete }: Props) {
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

  useEffect(() => {
    mounted.current = true;
    const controller = requestController();
    const restoreScans = async () => {
      try {
        const loadedSites = await api.get<Site[]>("/api/v1/sites", undefined, controller.signal);
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
            const prior = await api.get<ScanStatus>(
              `/api/v1/scans/${savedId}`,
              undefined,
              controller.signal,
            );
            if (!isCurrent()) return;
            setScans((current) => isCurrent()
              ? { ...current, [site.id]: prior }
              : current);
            if (!isActiveScan(prior)) {
              forgetActiveScan(site.id, savedId);
              onComplete();
            }
          } catch (cause: unknown) {
            if (!isCurrent()) return;
            if (cause instanceof ApiError && cause.status === 404) {
              forgetActiveScan(site.id);
              setError("扫描因服务重启而中断，请重新扫描");
            }
          }
        }));
      } catch {
        if (!controller.signal.aborted) setError("无法加载普通整窗扫描站点");
      } finally {
        releaseController(controller);
      }
    };
    void restoreScans();
    return () => {
      mounted.current = false;
      for (const activeController of requestControllers.current) activeController.abort();
      requestControllers.current.clear();
    };
  }, [onComplete, releaseController, requestController]);

  useEffect(() => {
    const activeScans = Object.values(scans).filter(isActiveScan);
    if (activeScans.length === 0) return;
    let controller: AbortController | null = null;
    const timer = window.setTimeout(() => {
      controller = requestController();
      void Promise.allSettled(activeScans.map((scan) =>
        api.get<ScanStatus>(`/api/v1/scans/${scan.id}`, undefined, controller?.signal),
      )).then((results) => {
        if (controller?.signal.aborted) return;
        const applicableResults = results.flatMap((result, index) => {
          const prior = activeScans[index];
          return scans[prior.site_id]?.id === prior.id ? [{ prior, result }] : [];
        });
        const completed = applicableResults.some(({ prior, result }) => {
          if (result.status === "fulfilled" && !isActiveScan(result.value)) {
            forgetActiveScan(prior.site_id, prior.id);
            return true;
          }
          if (
            result.status === "rejected"
            && result.reason instanceof ApiError
            && result.reason.status === 404
          ) {
            forgetActiveScan(prior.site_id, prior.id);
            setError("扫描因服务重启而中断，请重新扫描");
          }
          return false;
        });
        setScans((current) => {
          const next = { ...current };
          applicableResults.forEach(({ prior, result }) => {
            if (current[prior.site_id]?.id !== prior.id) return;
            if (result.status === "fulfilled") {
              next[prior.site_id] = result.value;
            } else if (result.reason instanceof ApiError && result.reason.status === 404) {
              delete next[prior.site_id];
            }
          });
          return next;
        });
        if (completed) onComplete();
      }).finally(() => {
        if (controller !== null) releaseController(controller);
      });
    }, 1000);
    return () => {
      window.clearTimeout(timer);
      controller?.abort();
    };
  }, [onComplete, releaseController, requestController, scans]);

  async function startScan(site: Site, path: string) {
    if (
      startingSiteIdsRef.current.has(site.id)
      || (scans[site.id] !== undefined && isActiveScan(scans[site.id]))
    ) return;
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
      if (isActiveScan(scan)) rememberActiveScan(site.id, scan.id);
      else {
        forgetActiveScan(site.id);
        onComplete();
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

  return <section aria-label="普通整窗扫描">
    <h2>普通整窗扫描</h2>
    <p className="job-summary">扫描所选目录中的稳定 H5 文件，并创建普通整窗图件任务。</p>
    <details>
      <summary>选择目录并扫描</summary>
      {error && <p className="error" role="alert">{error}</p>}
      {sites.length === 0 && !error && <p className="loading">正在加载扫描站点…</p>}
      {sites.length > 0 && <div className="scan-controls">
      {sites.map((site) => {
        const scan = scans[site.id];
        const active = startingSiteIds.has(site.id)
          || (scan !== undefined && isActiveScan(scan));
        return <div className="scan-site-control" key={site.id}>
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
            {active ? `正在扫描 ${site.name}` : `选择目录并扫描 ${site.name}`}
          </button>
          <p>扫描距离：{formatScanDistance(site.current_profile.payload)}</p>
          <Link to="/sites">设置扫描距离</Link>
          <p>按原文件光缆距离扫描；修改用于新建任务，已有任务沿用原范围。</p>
          {pendingScan?.site.id === site.id && <div className="scan-scope">
            <p>扫描范围：{site.name} / {pendingScan.path}</p>
            <button
              className="primary"
              type="button"
              disabled={active}
              onClick={() => void startScan(site, pendingScan.path)}
            >
              开始扫描
            </button>
          </div>}
          {scan !== undefined && <ScanStatusCard scan={scan} />}
        </div>;
      })}
      </div>}
      {choosingSite !== null && <ScanDirectoryDialog
        site={choosingSite}
        onCancel={() => setChoosingSite(null)}
        onSelect={(path) => {
          if (path) setPendingScan({ site: choosingSite, path });
          setChoosingSite(null);
        }}
      />}
    </details>
  </section>;
}
