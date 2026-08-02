import { useCallback, useEffect, useState } from "react";
import { api, type Job, type Site } from "../api";

const STATUS_NAMES: Record<Job["status"], string> = {
  queued: "排队",
  running: "运行中",
  succeeded: "成功",
  failed: "失败",
  cancelled: "已取消",
};

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[] | null>(null);
  const [sites, setSites] = useState<Site[]>([]);
  const [error, setError] = useState("");

  const refresh = useCallback(() => {
    api
      .get<Job[]>("/api/v1/jobs")
      .then(setJobs)
      .catch(() => setError("无法加载任务"));
  }, []);

  useEffect(() => {
    refresh();
    api.get<Site[]>("/api/v1/sites").then(setSites).catch(() => undefined);
  }, [refresh]);

  async function act(job: Job, action: "retry" | "cancel") {
    try {
      await api.post(`/api/v1/jobs/${job.id}/${action}`);
      refresh();
    } catch {
      setError("任务操作失败");
    }
  }

  async function scan(site: Site) {
    try {
      await api.post(`/api/v1/sites/${site.id}/scan`, {});
      refresh();
    } catch {
      setError("触发扫描失败");
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
        <p>
          {sites.map((site) => (
            <button
              key={site.id}
              className="subtle"
              type="button"
              onClick={() => void scan(site)}
            >
              扫描 {site.name}
            </button>
          ))}
        </p>
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
