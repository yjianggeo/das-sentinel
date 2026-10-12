import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, type Site } from "../api";
import { MODEL_SCAN_DIRECTORIES, modelScanApi, type ModelScan } from "../modelScanApi";
import type { WindowPage } from "../windowApi";
import { formatOccurredAt } from "./EventQueue";
import "./model-scans.css";

const STATUS: Record<string, string> = { queued: "排队", running: "扫描中", paused: "已暂停", cancelled: "已取消", failed: "需要处理", succeeded: "全部完成", completed_with_errors: "已结束，部分数据无法判断" };
const ESTIMATE: Record<string, string> = { discovering: "正在统计数据范围", estimating: "正在积累速度样本", waiting_worker: "等待扫描服务", paused: "暂停后不估算完成时间", cancelled: "已取消", failed: "处理错误后可重试", complete: "处理已结束" };
function duration(seconds: number) {
    if (seconds < 60) return "不足 1 分钟";
    const minutes = Math.ceil(seconds / 60);
    return minutes < 60 ? `${minutes} 分钟` : `${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分钟`;
}
export default function ModelScans() {
    const [page, setPage] = useState<WindowPage<ModelScan> | null>(null);
    const [offset, setOffset] = useState(0);
    const [revision, setRevision] = useState(0);
    const [error, setError] = useState("");
    const [busy, setBusy] = useState(false);
    const lock = useRef(false);
    const pending = useRef<{ key: string; id: string } | null>(null);
    const [sites, setSites] = useState<Site[]>([]);
    const [site, setSite] = useState("");
    const [model, setModel] = useState<{ id: string; name: string } | null>(null);
    const [directories, setDirectories] = useState<string[]>([]);
    const active = useRef(false);
    useEffect(() => {
        const controller = new AbortController();
        void api.get<Site[]>("/api/v1/sites", undefined, controller.signal).then(value => {
            if (!controller.signal.aborted) setSites(value);
        }).catch(cause => { if (!controller.signal.aborted) setError(String(cause)); });
        return () => controller.abort();
    }, []);
    useEffect(() => {
        const controller = new AbortController();
        setModel(null);
        if (site) void api.get<{ id: string; name: string } | null>("/api/v1/window-models/active", { site_id: site }, controller.signal)
            .then(value => { if (!controller.signal.aborted) setModel(value); })
            .catch(cause => { if (!controller.signal.aborted) setError(String(cause)); });
        return () => controller.abort();
    }, [site]);
    useEffect(() => {
        const controller = new AbortController();
        let stopped = false;
        let timer: number | undefined;
        const schedule = () => {
            window.clearTimeout(timer);
            timer = window.setTimeout(() => setRevision(v => v + 1), document.visibilityState === "visible" && active.current ? 10000 : 60000);
        };
        const onVisibility = () => { if (document.visibilityState === "visible") setRevision(v => v + 1); else schedule(); };
        document.addEventListener("visibilitychange", onVisibility);
        const timeout = window.setTimeout(() => controller.abort(), 15000);
        void modelScanApi.list(offset, controller.signal).then(value => {
            if (stopped || controller.signal.aborted) return;
            active.current = value.items.some(scan => ["queued", "running"].includes(scan.status));
            setPage(value); setError("");
        }).catch(cause => { if (!stopped) setError(String(cause)); }).finally(() => {
            window.clearTimeout(timeout);
            if (!stopped) schedule();
        });
        return () => { stopped = true; controller.abort(); window.clearTimeout(timer); window.clearTimeout(timeout); document.removeEventListener("visibilitychange", onVisibility); };
    }, [offset, revision]);
    async function act(run: () => Promise<unknown>) {
        if (lock.current) return;
        lock.current = true; setBusy(true); setError("");
        try { await run(); setRevision(v => v + 1); }
        catch (cause) { setError(String(cause)); }
        finally { lock.current = false; setBusy(false); }
    }
    async function create() {
        if (!model || !directories.length) return;
        const key = JSON.stringify([site, model.id, directories]);
        if (pending.current?.key !== key) pending.current = { key, id: crypto.randomUUID() };
        await modelScanApi.create(site, model.id, directories, pending.current.id);
        pending.current = null; setDirectories([]); setOffset(0);
    }
    return <section className="model-scans" id="model-scans" aria-label="模型扫描">
        <h2>模型扫描 · 30 秒窗口</h2>
        <p className="job-summary">直接用模型筛选地震类窗口，仅为入选窗口生成图件。包含传统规则跳过的数据，保留已有人工标签。</p>
        <p className="job-summary">当前页有活动批次且页面可见时每 10 秒更新；空闲或页面隐藏时每 60 秒检查，返回页面或操作后立即刷新。</p>
        <details><summary>新建模型扫描</summary><fieldset disabled={busy}>
            <label>扫描站点 <select value={site} onChange={e => setSite(e.target.value)}><option value="">选择站点</option>{sites.map(s => <option value={s.id} key={s.id}>{s.name}</option>)}</select></label>
            <p>{model ? `使用 ${model.name} · ${model.id.slice(0, 8)}` : "选择已启用整窗模型的站点"}</p>
            <div className="model-scan-directories">{MODEL_SCAN_DIRECTORIES.map(dir => <label key={dir}><input type="checkbox" checked={directories.includes(dir)} onChange={e => setDirectories(old => e.target.checked ? [...old, dir] : old.filter(d => d !== dir))} />{dir}</label>)}</div>
            <button disabled={!model || !directories.length} onClick={() => void act(create)}>创建扫描</button>
        </fieldset></details>
        {error && <p role="alert">{error} <button onClick={() => setRevision(v => v + 1)}>刷新模型扫描</button></p>}
        {!page && !error && <p>正在加载模型扫描…</p>}
        {page?.items.map(scan => <section className="model-scan-card" key={scan.id} id={`model-scan-${scan.id}`} aria-label={`模型批次 ${scan.id}`}>
            <div className="model-scan-heading"><strong>{scan.model.name} · {scan.id.slice(0, 8)}</strong><span>{STATUS[scan.status] ?? scan.status}</span></div>
            <p className="model-scan-paths">{scan.directories.join("、")}</p>
            <p>来源 {scan.counts.sources} · 已判断 {scan.counts.judged}{scan.total === null ? "" : ` / ${scan.total}`} · 入选 {scan.counts.selected} · 待出图 {scan.counts.pending_artifacts} · 可复核 {scan.counts.ready}</p>
            <p className="job-summary">模型阴性 {scan.counts.negative} · 无法判断 {scan.counts.unjudgeable} · 来源错误 {scan.counts.source_errors} · 已有人工复核保护 {scan.counts.protected ?? 0}</p>
            <p>{scan.remaining_seconds != null && scan.remaining_seconds > 0 ? `预计剩余 ${duration(scan.remaining_seconds)}${scan.estimated_finish_at ? ` · 预计完成 ${formatOccurredAt(scan.estimated_finish_at, "Asia/Shanghai")}` : ""}` : ESTIMATE[scan.estimate_status] ?? "等待速度估算"}</p>
            {scan.errors.map((e, i) => <p role="alert" key={i}>{e.code}：{e.message}</p>)}
            {(scan.counts.source_errors > 0 || scan.counts.unjudgeable > 0) && <p className="job-summary">读取失败可重试；缺失、变更或不兼容的源文件需修复后新建批次，重新统计覆盖范围。</p>}
            <div className="window-toolbar">
                {["queued", "running"].includes(scan.status) && <button disabled={busy} onClick={() => void act(() => modelScanApi.action(scan.id, "pause"))}>暂停</button>}
                {scan.status === "paused" && <button disabled={busy} onClick={() => void act(() => modelScanApi.action(scan.id, "resume"))}>继续</button>}
                {(scan.status === "failed" || (scan.status === "completed_with_errors" && scan.retryable)) && <button disabled={busy} onClick={() => void act(() => modelScanApi.action(scan.id, "retry"))}>重试</button>}
                {["queued", "running", "paused", "failed"].includes(scan.status) && <button className="danger" disabled={busy} onClick={() => void act(() => modelScanApi.action(scan.id, "cancel"))}>取消批次</button>}
                <Link to={`/events?range=model_recommended&scan_id=${scan.id}`}>查看入选窗口</Link>
            </div>
        </section>)}
        {page && <div className="job-pagination"><span>共 {page.total} 个模型扫描批次</span><button disabled={offset === 0} onClick={() => setOffset(v => Math.max(0, v - 10))}>上一页批次</button><button disabled={offset + 10 >= page.total} onClick={() => setOffset(v => v + 10)}>下一页批次</button></div>}
    </section>;
}
