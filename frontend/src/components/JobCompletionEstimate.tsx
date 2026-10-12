import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { formatOccurredAt } from "./EventQueue";
import "./job-completion-estimate.css";

type EstimateStatus = "complete" | "running" | "estimating" | "waiting_worker";
type QueueEstimate = {
    key: string; label: string; status: EstimateStatus; queued: number; running: number;
    sample_count: number; remaining_seconds: number | null;
};
type Estimate = {
    as_of: string; status: EstimateStatus; queued: number; running: number;
    remaining_seconds: number | null; estimated_finish_at: string | null; queues: QueueEstimate[];
};

function duration(seconds: number): string {
    const total = Math.max(1, Math.ceil(seconds / 60));
    const days = Math.floor(total / 1440), hours = Math.floor(total % 1440 / 60), minutes = total % 60;
    return "约 " + [days && `${days} 天`, hours && `${hours} 小时`, minutes && `${minutes} 分钟`].filter(Boolean).join(" ");
}

const statusText: Record<EstimateStatus, string> = {
    complete: "无待扫描任务", running: "按近期处理速度估算",
    estimating: "正在积累估算样本", waiting_worker: "等待处理服务恢复",
};

export default function JobCompletionEstimate({ revision = 0 }: { revision?: number }) {
    const [estimate, setEstimate] = useState<Estimate | null>(null);
    const [error, setError] = useState(false);
    const active = useRef(false);
    const [visible, setVisible] = useState(document.visibilityState === "visible");
    useEffect(() => {
        let stopped = false;
        let controller: AbortController | undefined;
        let timer: number | undefined;
        const schedule = () => {
            window.clearTimeout(timer);
            timer = window.setTimeout(() => void refresh(), document.visibilityState === "visible" && active.current ? 5000 : 60000);
        };
        const refresh = async () => {
            window.clearTimeout(timer);
            controller?.abort();
            const request = new AbortController();
            controller = request;
            const timeout = window.setTimeout(() => request.abort(), 15000);
            try {
                const value = await api.get<Estimate>("/api/v1/jobs/estimate", undefined, request.signal);
                if (!stopped && request === controller && !request.signal.aborted) {
                    active.current = value.queued + value.running > 0;
                    setEstimate(value); setError(false);
                }
            } catch {
                if (!stopped && request === controller) setError(true);
            } finally {
                window.clearTimeout(timeout);
                if (!stopped && request === controller) schedule();
            }
        };
        const onVisibility = () => {
            setVisible(document.visibilityState === "visible");
            if (document.visibilityState === "visible") void refresh(); else schedule();
        };
        document.addEventListener("visibilitychange", onVisibility);
        void refresh();
        return () => { stopped = true; controller?.abort(); window.clearTimeout(timer); document.removeEventListener("visibilitychange", onVisibility); };
    }, [revision]);
    const available = estimate?.status === "running" && !error;
    return <section className="job-completion-estimate" aria-label="整窗图件任务预计完成时间">
        <div className="job-estimate-heading">
            <h2>整窗图件任务预计完成时间</h2>
            <span>{error ? "估算更新失败" : estimate ? statusText[estimate.status] : "正在读取队列…"}</span>
        </div>
        <div className="job-estimate-values">
            <div><span>预计剩余时间</span><strong>{available && estimate.remaining_seconds !== null
                ? duration(estimate.remaining_seconds) : estimate?.status === "complete" && !error ? "无待扫描任务" : "暂无法估算"}</strong></div>
            <div><span>预计完成（北京时间）</span><strong>{available && estimate.estimated_finish_at
                ? formatOccurredAt(estimate.estimated_finish_at, "Asia/Shanghai").slice(0, 16) : "—"}</strong></div>
            <div><span>待扫描任务</span><strong>{estimate ? `${estimate.queued + estimate.running} 个` : "—"}</strong>
                {estimate && <small>{estimate.queued} 个排队 · {estimate.running} 个运行中</small>}</div>
        </div>
        {error && <p role="status">暂时无法获取最新估算，{visible && active.current ? 5 : 60} 秒后自动重试。</p>}
        {!error && estimate?.queues.filter(q => q.status === "waiting_worker" || q.status === "estimating").map(q =>
            <p key={q.key}>{q.label}：{q.status === "waiting_worker" ? "尚未收到近期处理活动，请检查处理服务。" : "近期完成样本不足，扫描继续后将自动显示估算。"}</p>)}
        <p className="job-estimate-note">仅统计已创建的整窗图件任务；模型扫描的完整进度与预计完成时间见上方独立批次。不受分页和筛选影响，失败及已取消任务不计入；有活动任务且页面可见时每 5 秒更新，空闲或页面隐藏时每 60 秒检查，返回页面或操作后立即刷新。</p>
    </section>;
}
