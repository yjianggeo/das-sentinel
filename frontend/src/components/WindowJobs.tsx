import { formatOccurredAt } from "./EventQueue";
import { useEffect, useRef, useState } from "react";
import { api, type Site } from "../api";
import { windowApi, windowStatusNames, type WindowJob, type WindowPage } from "../windowApi";
import "./window-events.css";

const PAGE_SIZE = 50;
const progressNames: Record<WindowJob["status"], string> = {
    pending: "等待分析", queued: "等待分析", running: "分析及图件生成中",
    succeeded: "100% · 图件已生成", failed: "分析未完成", cancelled: "已停止",
};

type Props = { revision?: number; onComplete?: () => void };

export default function WindowJobs({ revision = 0, onComplete }: Props) {
    const [query, setQuery] = useState({ status: "", siteId: "", offset: 0 });
    const [sites, setSites] = useState<Site[]>([]);
    const [siteError, setSiteError] = useState("");
    const [localRevision, setLocalRevision] = useState(0);
    const [page, setPage] = useState<WindowPage<WindowJob> | null>(null);
    const [error, setError] = useState("");
    const [busy, setBusy] = useState<string | null>(null);
    const [refreshing, setRefreshing] = useState(false);
    const actionLock = useRef(false);
    const mounted = useRef(true);
    const requestedQuery = useRef(query);
    useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
    useEffect(() => {
        const controller = new AbortController();
        void api.get<Site[]>("/api/v1/sites", undefined, controller.signal).then(value => {
            if (!controller.signal.aborted) setSites(value);
        }).catch(cause => {
            if (!controller.signal.aborted) setSiteError(`站点列表加载失败：${cause instanceof Error ? cause.message : String(cause)}`);
        });
        return () => controller.abort();
    }, []);
    useEffect(() => {
        const controller = new AbortController();
        if (requestedQuery.current !== query) setPage(null);
        requestedQuery.current = query;
        setError("");
        setRefreshing(true);
        const params: Record<string, string> = { limit: String(PAGE_SIZE), offset: String(query.offset), window_seconds: "30" };
        if (query.status) params.status = query.status;
        if (query.siteId) params.site_id = query.siteId;
        void windowApi.jobs(params, controller.signal).then(value => {
            if (controller.signal.aborted) return;
            const lastOffset = Math.max(0, Math.ceil(value.total / PAGE_SIZE) - 1) * PAGE_SIZE;
            if (query.offset > lastOffset) {
                setQuery({ ...query, offset: lastOffset });
                return;
            }
            setPage(value);
        }).catch(cause => {
            if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : String(cause));
        }).finally(() => {
            if (!controller.signal.aborted) setRefreshing(false);
        });
        return () => controller.abort();
    }, [query, revision, localRevision]);
    useEffect(() => {
        if (!page?.items.some(job => ["queued", "running"].includes(job.status)) || error) return;
        const timer = window.setTimeout(() => setLocalRevision(value => value + 1), 5000);
        return () => window.clearTimeout(timer);
    }, [page, error]);
    async function action(id: string, kind: "cancel" | "retry") {
        if (actionLock.current)
            return;
        actionLock.current = true;
        setBusy(id);
        setError("");
        try {
            await windowApi.jobAction(id, kind);
            if (mounted.current) {
                setLocalRevision(v => v + 1);
                onComplete?.();
            }
        }
        catch (cause) {
            if (mounted.current)
                setError(cause instanceof Error ? cause.message : String(cause));
        }
        finally {
            actionLock.current = false;
            if (mounted.current)
                setBusy(null);
        }
    }
    const pageSummary = page && Object.entries(windowStatusNames).map(([status, label]) => {
        const count = page.items.filter(job => job.status === status).length;
        return count ? `${label} ${count}` : null;
    }).filter(Boolean).join(" · ");

    return <section className="window-jobs" aria-label="整窗任务">
        <h2>整窗任务 · 30 秒全通道</h2>
        <p className="job-summary">每个任务分析一个 30 秒全通道窗口，并生成原始和滤波瀑布图。目录扫描提交数为源文件数，与窗口任务数不同。</p>
        <div className="job-filters">
            <label>整窗任务状态
                <select value={query.status} onChange={e => setQuery({ ...query, status: e.target.value, offset: 0 })}>
                    <option value="">全部状态</option>
                    {Object.entries(windowStatusNames).filter(([value]) => value !== "pending").map(([value, label]) => <option key={value} value={value}>整窗 · {label}</option>)}
                </select>
            </label>
            <label>整窗任务站点
                <select value={query.siteId} onChange={e => setQuery({ ...query, siteId: e.target.value, offset: 0 })}>
                    <option value="">全部站点</option>
                    {sites.map(site => <option key={site.id} value={site.id}>{site.name}</option>)}
                </select>
            </label>
            <button className="subtle" disabled={refreshing} onClick={() => setLocalRevision(v => v + 1)}>刷新整窗任务</button>
        </div>
        {siteError && <p role="alert">{siteError}</p>}
        {error && <p role="alert">{error}</p>}
        {!page && !error && <p>正在加载整窗任务…</p>}
        {page && <>
            <div className="job-pagination">
                <span>共 {page.total} 个整窗任务 · 第 {Math.floor(query.offset / PAGE_SIZE) + 1} / {Math.max(1, Math.ceil(page.total / PAGE_SIZE))} 页 · 每页 {PAGE_SIZE} 条</span>
                <button className="subtle" aria-label="上一页整窗任务" disabled={refreshing || query.offset === 0} onClick={() => setQuery({ ...query, offset: Math.max(0, query.offset - PAGE_SIZE) })}>上一页</button>
                <button className="subtle" aria-label="下一页整窗任务" disabled={refreshing || query.offset + PAGE_SIZE >= page.total} onClick={() => setQuery({ ...query, offset: query.offset + PAGE_SIZE })}>下一页</button>
                {refreshing && <span className="job-summary" role="status">正在刷新…</span>}
            </div>
            {pageSummary && <p className="job-summary window-job-summary">本页：{pageSummary}</p>}
            <div className="window-table-scroll" role="region" aria-label="整窗任务列表" tabIndex={0}>
                <table className="data jobs-table" aria-busy={refreshing}>
                    <thead><tr><th>ID</th><th>时间窗及站点</th><th>状态</th><th>进度</th><th>尝试次数</th><th>错误</th><th>操作</th></tr></thead>
                    <tbody>{page.items.map(job => {
                        const progress = job.status === "succeeded" ? 100 : ["queued", "pending"].includes(job.status) ? 0 : undefined;
                        const coverageIncomplete = job.status === "failed" && job.error_code === "coverage_incomplete";
                        return <tr key={job.id}>
                            <td title={job.id}>{job.id.slice(0, 8)}</td>
                            <td className="job-source" title={job.window_id}>
                                <div>{formatOccurredAt(job.start_utc, job.site_timezone)}</div>
                                <small>{job.site_name} · 30 秒 · 全通道</small>
                            </td>
                            <td><span className={`status-pill status-${job.status}`}>{windowStatusNames[job.status]}</span></td>
                            <td className={`window-job-progress window-job-progress-${job.status}`}>
                                <div className="progress-track" role="progressbar" aria-label={`任务 ${job.id} 进度`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress} aria-valuetext={progressNames[job.status]}>
                                    <div className={`progress-fill${job.status === "running" ? " progress-indeterminate" : ""}`} style={job.status === "running" ? undefined : { width: `${progress ?? 0}%` }} />
                                </div>
                                <small>{progressNames[job.status]}</small>
                            </td>
                            <td>{job.attempt_count}</td>
                            <td className="window-job-error"><div>{job.error_message ?? "—"}</div>{coverageIncomplete && <small>补齐数据后重新扫描</small>}</td>
                            <td>
                                {job.model_scan_id ? <a href={`#model-scan-${job.model_scan_id}`}>由模型批次管理</a> : <>
                                    {["pending", "queued", "running"].includes(job.status) && <button className="danger" disabled={busy !== null} onClick={() => void action(job.id, "cancel")}>取消</button>}
                                    {job.status === "failed" && !coverageIncomplete && <button className="subtle" disabled={busy !== null} onClick={() => void action(job.id, "retry")}>重试</button>}
                                </>}
                            </td>
                        </tr>;
                    })}</tbody>
                </table>
            </div>
            {page.items.length === 0 && <p>没有符合条件的整窗任务</p>}
        </>}
    </section>;
}
