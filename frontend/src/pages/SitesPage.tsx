import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, ApiError, type Site } from "../api";

const CABLE_TYPES = [
  { value: "downhole", label: "井下光缆" },
  { value: "urban", label: "城市光缆" },
  { value: "subsea", label: "海底光缆" },
] as const;

export default function SitesPage() {
  const [sites, setSites] = useState<Site[] | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [error, setError] = useState("");

  const refresh = useCallback(() => {
    api
      .get<Site[]>("/api/v1/sites")
      .then(setSites)
      .catch(() => setError("无法加载站点"));
  }, []);

  useEffect(refresh, [refresh]);

  async function toggleEnabled(site: Site) {
    try {
      await api.patch(`/api/v1/sites/${site.id}`, { enabled: !site.enabled });
      refresh();
    } catch {
      setError("更新站点状态失败");
    }
  }

  if (sites === null) {
    return <p className="loading">正在加载…</p>;
  }
  return (
    <section>
      <h1>站点</h1>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <button className="primary" type="button" onClick={() => setShowForm(true)}>
        新建站点
      </button>
      {showForm && (
        <SiteForm
          onCreated={() => {
            setShowForm(false);
            refresh();
          }}
          onCancel={() => setShowForm(false)}
        />
      )}
      <table className="data">
        <thead>
          <tr>
            <th>名称</th>
            <th>数据路径</th>
            <th>时区</th>
            <th>光缆类型</th>
            <th>配置版本</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          {sites.map((site) => (
            <tr key={site.id}>
              <td>{site.name}</td>
              <td>{site.data_path}</td>
              <td>{site.timezone}</td>
              <td>
                {CABLE_TYPES.find((item) => item.value === site.cable_type)
                  ?.label ?? site.cable_type}
              </td>
              <td>v{site.current_profile.version}</td>
              <td>{site.enabled ? "启用" : "停用"}</td>
              <td>
                <button
                  className="subtle"
                  type="button"
                  onClick={() => void toggleEnabled(site)}
                >
                  {site.enabled ? "停用" : "启用"}
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function SiteForm({
  onCreated,
  onCancel,
}: {
  onCreated: () => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState("");
  const [dataPath, setDataPath] = useState("");
  const [timezone, setTimezone] = useState("Asia/Shanghai");
  const [cableType, setCableType] = useState<string>("downhole");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await api.post("/api/v1/sites", {
        name,
        data_path: dataPath,
        timezone,
        cable_type: cableType,
      });
      onCreated();
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : "创建站点失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="stack" onSubmit={(event) => void onSubmit(event)}>
      <label>
        名称
        <input
          value={name}
          onChange={(event) => setName(event.target.value)}
          required
        />
      </label>
      <label>
        数据路径
        <input
          value={dataPath}
          onChange={(event) => setDataPath(event.target.value)}
          placeholder="/data/jjk"
          required
        />
      </label>
      <label>
        IANA 时区
        <input
          value={timezone}
          onChange={(event) => setTimezone(event.target.value)}
          required
        />
      </label>
      <label>
        光缆类型
        <select
          value={cableType}
          onChange={(event) => setCableType(event.target.value)}
        >
          {CABLE_TYPES.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
      </label>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <div>
        <button className="primary" type="submit" disabled={busy}>
          保存
        </button>{" "}
        <button className="subtle" type="button" onClick={onCancel}>
          取消
        </button>
      </div>
    </form>
  );
}
