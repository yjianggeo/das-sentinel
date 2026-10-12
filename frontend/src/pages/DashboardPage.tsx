import { useEffect, useState } from "react";
import { ApiError, api, type Dashboard } from "../api";

const STATUS_NAMES: Record<string, string> = {
  planned: "待规划", pending: "待分析", queued: "排队", running: "运行中", succeeded: "成功",
  failed: "失败", cancelled: "已取消", cancelling: "取消中",
  paused: "已暂停", completed_with_errors: "已结束，部分数据无法判断",
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
        <section className="card" aria-label="30 秒窗口任务">
          <div className="card-title">30 秒窗口任务</div>
          <ul>{Object.entries(dashboard.window_status_counts).map(([status, count]) => <li key={status}>{STATUS_NAMES[status] ?? status}：{count}</li>)}</ul>
        </section>
        <section className="card" aria-label="模型扫描批次">
          <div className="card-title">模型扫描批次</div>
          <ul>{Object.entries(dashboard.model_scan_status_counts).map(([status, count]) => <li key={status}>{STATUS_NAMES[status] ?? status}：{count}</li>)}</ul>
        </section>
        <section className="card" aria-label="人工复核">
          <div className="card-title">人工复核</div>
          <ul>
            <li>地震类：{dashboard.review_counts.earthquake}</li>
            <li>噪声类：{dashboard.review_counts.noise}</li>
            <li>待复核：{dashboard.review_counts.pending}</li>
          </ul>
          <p>成功的 30 秒窗口；待复核表示当前输入尚无有效的地震类或噪声类人工标签。</p>
        </section>
      </div>
    </section>
  );
}
