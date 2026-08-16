import { useEffect, useState } from "react";
import { ApiError, api, type Dashboard } from "../api";

const LABEL_NAMES: Record<string, string> = {
  earthquake: "地震",
  blast: "爆破",
  traffic: "车辆",
  noise: "噪声",
  unknown: "待定",
};

export default function DashboardPage() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    api
      .get<Dashboard>("/api/v1/dashboard")
      .then((response) => {
        if (active) setDashboard(response);
      })
      .catch((cause) => {
        if (active && !(cause instanceof ApiError && cause.status === 401)) {
          setError("无法加载运行数据");
        }
      });
    return () => {
      active = false;
    };
  }, []);

  if (error) {
    return <p className="error" role="alert">{error}</p>;
  }
  if (dashboard === null) {
    return <p className="loading">正在加载…</p>;
  }
  return (
    <section>
      <h1>运行总览</h1>
      <div className="card-grid">
        <div className="card">
          <div className="card-value">{dashboard.job_counts.queued ?? 0}</div>
          <div className="card-title">排队任务</div>
        </div>
        <div className="card">
          <div className="card-value">{dashboard.job_counts.running ?? 0}</div>
          <div className="card-title">运行中任务</div>
        </div>
        <div className="card">
          <div className="card-value">{dashboard.job_counts.failed ?? 0}</div>
          <div className="card-title">失败任务</div>
        </div>
        <div className="card">
          <div className="card-value">{dashboard.today_event_count}</div>
          <div className="card-title">今日事件</div>
        </div>
      </div>
      <div className="card">
        <div className="card-title">机器分类分布</div>
        <ul>
          {Object.entries(dashboard.machine_label_counts).map(([label, count]) => (
            <li key={label}>
              {LABEL_NAMES[label] ?? label}：{count}
            </li>
          ))}
          {Object.keys(dashboard.machine_label_counts).length === 0 && <li>暂无事件</li>}
        </ul>
      </div>
    </section>
  );
}
