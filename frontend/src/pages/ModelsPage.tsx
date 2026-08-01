import { useCallback, useEffect, useState } from "react";
import { api, ApiError, type ModelInfo, type Site } from "../api";
import ValidationReport from "../components/ValidationReport";

const STATUS_NAMES: Record<ModelInfo["status"], string> = {
  candidate: "候选",
  active: "已发布",
  retired: "已退役",
  invalid: "无效",
};

export default function ModelsPage() {
  const [sites, setSites] = useState<Site[]>([]);
  const [siteId, setSiteId] = useState("");
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [selected, setSelected] = useState<ModelInfo | null>(null);
  const [confirm, setConfirm] = useState<"publish" | "rollback" | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback((id: string) => {
    if (!id) {
      setModels([]);
      return;
    }
    api
      .get<ModelInfo[]>(`/api/v1/sites/${id}/models`)
      .then((list) => {
        setModels(list);
        setSelected((current) =>
          current ? (list.find((item) => item.id === current.id) ?? null) : null,
        );
      })
      .catch(() => setError("无法加载模型"));
  }, []);

  useEffect(() => {
    api
      .get<Site[]>("/api/v1/sites")
      .then((list) => {
        setSites(list);
        if (list.length > 0) {
          setSiteId(list[0].id);
        }
      })
      .catch(() => setError("无法加载站点"));
  }, []);

  useEffect(() => {
    setModels(null);
    refresh(siteId);
  }, [siteId, refresh]);

  async function train() {
    try {
      await api.post<ModelInfo>(`/api/v1/sites/${siteId}/models`);
      refresh(siteId);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : "训练失败");
    }
  }

  async function publish() {
    if (!selected) return;
    try {
      await api.post<ModelInfo>(`/api/v1/models/${selected.id}/publish`);
      setConfirm(null);
      refresh(siteId);
    } catch (cause) {
      setConfirm(null);
      setError(cause instanceof ApiError ? cause.message : "发布失败");
    }
  }

  async function rollback() {
    try {
      await api.post<ModelInfo>(`/api/v1/sites/${siteId}/models/rollback`);
      setConfirm(null);
      refresh(siteId);
    } catch (cause) {
      setConfirm(null);
      setError(cause instanceof ApiError ? cause.message : "回滚失败");
    }
  }

  return (
    <section>
      <h1>模型</h1>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <label>
        站点
        <select value={siteId} onChange={(event) => setSiteId(event.target.value)}>
          {sites.map((site) => (
            <option key={site.id} value={site.id}>
              {site.name}
            </option>
          ))}
        </select>
      </label>{" "}
      <button className="primary" type="button" onClick={() => void train()}>
        训练新模型
      </button>
      {models === null ? (
        <p className="loading">正在加载…</p>
      ) : (
        <table className="data">
          <thead>
            <tr>
              <th>版本</th>
              <th>状态</th>
              <th>宏平均 F1</th>
              <th>创建时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.id}>
                <td>v{model.version}</td>
                <td>{STATUS_NAMES[model.status]}</td>
                <td>{(model.metrics.macro_f1 ?? 0).toFixed(3)}</td>
                <td>{model.created_at.replace("T", " ").replace(/(\.\d+)?Z$/, "")}</td>
                <td>
                  <button
                    className="subtle"
                    type="button"
                    onClick={() => setSelected(model)}
                  >
                    查看验证报告
                  </button>{" "}
                  {model.status === "candidate" && (
                    <button
                      className="primary"
                      type="button"
                      onClick={() => {
                        setSelected(model);
                        setConfirm("publish");
                      }}
                    >
                      发布候选模型
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {models.length === 0 && (
              <tr>
                <td colSpan={5}>暂无模型，请先积累复核样本并训练。</td>
              </tr>
            )}
          </tbody>
        </table>
      )}
      {models !== null && models.some((item) => item.status === "active") && (
        <button className="danger" type="button" onClick={() => setConfirm("rollback")}>
          回滚到上一模型
        </button>
      )}
      {selected !== null && <ValidationReport model={selected} />}
      {confirm === "publish" && selected !== null && (
        <div className="dialog-backdrop">
          <div className="dialog" role="dialog" aria-label="确认发布模型">
            <p>
              确认发布模型 v{selected.version}（宏平均 F1{" "}
              {(selected.metrics.macro_f1 ?? 0).toFixed(3)}）？当前模型将退役。
            </p>
            <button className="primary" type="button" onClick={() => void publish()}>
              确认发布
            </button>{" "}
            <button className="subtle" type="button" onClick={() => setConfirm(null)}>
              取消
            </button>
          </div>
        </div>
      )}
      {confirm === "rollback" && (
        <div className="dialog-backdrop">
          <div className="dialog" role="dialog" aria-label="确认回滚模型">
            <p>确认回滚到上一个有效模型？当前模型将退役。</p>
            <button className="danger" type="button" onClick={() => void rollback()}>
              确认回滚
            </button>{" "}
            <button className="subtle" type="button" onClick={() => setConfirm(null)}>
              取消
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
