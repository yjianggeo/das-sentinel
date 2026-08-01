import { useEffect, useMemo, useState } from "react";
import { api, type EventSummary, type Job } from "../api";

const LABEL_NAMES: Record<string, string> = {
  earthquake: "地震",
  blast: "爆破",
  traffic: "车辆",
  noise: "噪声",
  unknown: "待定",
};

export default function DashboardPage() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [events, setEvents] = useState<EventSummary[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      api.get<Job[]>("/api/v1/jobs"),
      api.get<EventSummary[]>("/api/v1/events", { limit: "100" }),
    ])
      .then(([jobList, eventList]) => {
        if (cancelled) return;
        setJobs(jobList);
        setEvents(eventList);
      })
      .catch(() => {
        if (!cancelled) setError("无法加载运行数据");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const counts = useMemo(() => {
    const byStatus = { queued: 0, running: 0, failed: 0 };
    for (const job of jobs ?? []) {
      if (job.status in byStatus) {
        byStatus[job.status as keyof typeof byStatus] += 1;
      }
    }
    const today = new Date().toISOString().slice(0, 10);
    const todayEvents = (events ?? []).filter((event) =>
      event.occurred_at.startsWith(today),
    ).length;
    const byLabel: Record<string, number> = {};
    for (const event of events ?? []) {
      byLabel[event.machine_label] = (byLabel[event.machine_label] ?? 0) + 1;
    }
    return { byStatus, todayEvents, byLabel };
  }, [jobs, events]);

  if (error) {
    return <p className="error" role="alert">{error}</p>;
  }
  if (jobs === null || events === null) {
    return <p className="loading">正在加载…</p>;
  }
  return (
    <section>
      <h1>运行总览</h1>
      <div className="card-grid">
        <div className="card">
          <div className="card-value">{counts.byStatus.queued}</div>
          <div className="card-title">排队任务</div>
        </div>
        <div className="card">
          <div className="card-value">{counts.byStatus.running}</div>
          <div className="card-title">运行中任务</div>
        </div>
        <div className="card">
          <div className="card-value">{counts.byStatus.failed}</div>
          <div className="card-title">失败任务</div>
        </div>
        <div className="card">
          <div className="card-value">{counts.todayEvents}</div>
          <div className="card-title">今日事件</div>
        </div>
      </div>
      <div className="card">
        <div className="card-title">机器分类分布（最近 100 条）</div>
        <ul>
          {Object.entries(counts.byLabel).map(([label, count]) => (
            <li key={label}>
              {LABEL_NAMES[label] ?? label}：{count}
            </li>
          ))}
          {Object.keys(counts.byLabel).length === 0 && <li>暂无事件</li>}
        </ul>
      </div>
    </section>
  );
}
