import { useEffect, useRef, useState } from "react";
import { api } from "../api";

type Readiness = { window_count: number; reviewed_windows: number; eligible_windows: number; ready: boolean; reasons: string[]; classes: Record<string, { present: number; absent: number; unknown: number }>; training_grid?: { channels: number; spacing_m: number; start_m: number; stop_m: number; source_grid_count: number } | null };
type Job = { id: string; status: string; error: string | null; metrics: unknown };

export default function WindowTraining({ siteId }: { siteId: string }) {
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submitting = useRef(false);
  const hasActiveJobs = jobs.some((job) => ["queued", "running"].includes(job.status));
  useEffect(() => {
    let alive = true;
    setReadiness(null);
    setJobs([]);
    setError("");
    if (!siteId) return;
    const refresh = () => Promise.all([
      api.get<Readiness>(`/api/v1/window-training/readiness?site_id=${encodeURIComponent(siteId)}`),
      api.get<Job[]>(`/api/v1/window-training/jobs?site_id=${encodeURIComponent(siteId)}`),
    ]).then(([state, list]) => { if (alive) { setReadiness(state); setJobs(list); } }).catch(() => { if (alive) setError("无法加载整窗训练状态"); });
    void refresh();
    return () => { alive = false; };
  }, [siteId]);
  useEffect(() => {
    if (!siteId || !hasActiveJobs) return;
    let alive = true;
    let polling = false;
    const timer = window.setInterval(() => {
      if (polling) return;
      polling = true;
      void (async () => {
        try {
          const list = await api.get<Job[]>(`/api/v1/window-training/jobs?site_id=${encodeURIComponent(siteId)}`);
          let state: Readiness | null = null;
          if (!list.some((job) => ["queued", "running"].includes(job.status))) {
            try { state = await api.get<Readiness>(`/api/v1/window-training/readiness?site_id=${encodeURIComponent(siteId)}`); }
            catch { if (alive) setError("无法加载整窗训练状态"); }
          }
          if (alive) { setJobs(list); if (state) setReadiness(state); }
        } catch { if (alive) setError("无法加载整窗训练状态"); }
        finally { polling = false; }
      })();
    }, 5000);
    return () => { alive = false; window.clearInterval(timer); };
  }, [siteId, hasActiveJobs]);
  async function train() {
    if (!readiness?.ready || submitting.current) return;
    submitting.current = true;
    setBusy(true);
    setError("");
    try {
      const job = await api.post<Job>("/api/v1/window-training/jobs", { site_id: siteId });
      setJobs((list) => [job, ...list]);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "整窗训练失败"); }
    finally { submitting.current = false; setBusy(false); }
  }
  return <section aria-label="整窗二分类训练">
    <h2>整窗二分类训练</h2>
    <p>输入为 30 秒全通道窗口，只区分地震（含爆破）与噪声（含车辆）。同窗存在地震或爆破时归为地震；未确定窗口不参与监督训练。原始复核标签保留，模型先作离线验证。</p>
    {error && <p role="alert">{error}</p>}
    {readiness && <>
      <p>窗口记录 {readiness.window_count} · 当前已复核 {readiness.reviewed_windows} · 可训练 {readiness.eligible_windows}</p>
      {readiness.classes.earthquake && <p>地震（含爆破）{readiness.classes.earthquake.present} / 噪声（含车辆）{readiness.classes.earthquake.absent} / 未确定或未复核 {readiness.classes.earthquake.unknown}</p>}
      {readiness.training_grid && <p>统一训练网格：{readiness.training_grid.channels} 道 × 3000 点 · 间距 {readiness.training_grid.spacing_m.toFixed(6)} m · {readiness.training_grid.source_grid_count} 种来源网格已对齐。缺失通道以无效掩码排除，原有波形数值保留。</p>}
      {!readiness.ready && <p>尚未就绪：{readiness.reasons.join(" ")}</p>}
    </>}
    <button type="button" className="primary" disabled={!readiness?.ready || busy || hasActiveJobs} onClick={() => void train()}>训练整窗二分类模型</button>
    <ul>{jobs.map((job) => <li key={job.id}>{job.status}{job.error && `：${job.error}`}{job.metrics != null && <details><summary>离线候选指标</summary><pre>{JSON.stringify(job.metrics, null, 2)}</pre></details>}</li>)}</ul>
  </section>;
}
