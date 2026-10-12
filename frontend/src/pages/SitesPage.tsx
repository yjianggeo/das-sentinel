import { useCallback, useEffect, useState, type FormEvent } from "react";
import { api, ApiError, type Site } from "../api";
import { formatScanDistance } from "../scanDistance";

const CABLE_TYPES = [
  { value: "downhole", label: "井下光缆" },
  { value: "urban", label: "城市光缆" },
  { value: "subsea", label: "海底光缆" },
] as const;

export default function SitesPage() {
  const [sites, setSites] = useState<Site[] | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [editingDistance, setEditingDistance] = useState<Site | null>(null);
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
      <button className="primary" type="button" disabled={editingDistance !== null} onClick={() => setShowForm(true)}>
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
      {editingDistance !== null && (
        <ScanDistanceForm
          key={editingDistance.id}
          site={editingDistance}
          onSaved={(updated) => {
            setSites((current) => current?.map((site) => site.id === updated.id ? updated : site) ?? null);
            setEditingDistance(null);
          }}
          onCancel={() => setEditingDistance(null)}
        />
      )}
      <table className="data">
        <thead>
          <tr>
            <th>名称</th>
            <th>数据路径</th>
            <th>时区</th>
            <th>光缆类型</th>
            <th>扫描距离</th>
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
              <td>{formatScanDistance(site.current_profile.payload)}</td>
              <td>v{site.current_profile.version}</td>
              <td>{site.enabled ? "启用" : "停用"}</td>
              <td>
                <button
                  className="subtle"
                  type="button"
                  disabled={showForm || editingDistance !== null}
                  onClick={() => setEditingDistance(site)}
                >
                  设置扫描距离
                </button>{" "}
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

function ScanDistanceForm({
  site,
  onSaved,
  onCancel,
}: {
  site: Site;
  onSaved: (site: Site) => void;
  onCancel: () => void;
}) {
  const payload = site.current_profile.payload;
  const [start, setStart] = useState(String(payload.scan_distance_start_m ?? ""));
  const [stop, setStop] = useState(String(payload.scan_distance_stop_m ?? ""));
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (busy) return;
    const startValue = start.trim() === "" ? null : Number(start);
    const stopValue = stop.trim() === "" ? null : Number(stop);
    if ([startValue, stopValue].some((value) => value !== null && (!Number.isFinite(value) || value < 0))) {
      setError("请输入有限且非负的距离");
      return;
    }
    if (startValue !== null && stopValue !== null && stopValue <= startValue) {
      setError("终点距离必须大于起点距离");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const updated = await api.patch<Site>(`/api/v1/sites/${site.id}`, {
        profile: { scan_distance_start_m: startValue, scan_distance_stop_m: stopValue },
      });
      onSaved(updated);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : "保存扫描距离失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="stack" aria-label={`${site.name} 扫描距离`} onSubmit={(event) => void onSubmit(event)}>
      <h2>{site.name} · 扫描距离</h2>
      <p>按原文件光缆距离设置，两端包含；留空表示该端不限。修改用于新建任务，已有任务沿用原范围。</p>
      <label>
        起点距离（m）
        <input inputMode="decimal" value={start} onChange={(event) => setStart(event.target.value)} disabled={busy} autoFocus />
      </label>
      <label>
        终点距离（m）
        <input inputMode="decimal" value={stop} onChange={(event) => setStop(event.target.value)} disabled={busy} />
      </label>
      {error && <p className="error" role="alert">{error}</p>}
      <div>
        <button className="primary" type="submit" disabled={busy}>
          {busy ? "正在保存…" : "保存扫描距离"}
        </button>{" "}
        <button className="subtle" type="button" onClick={onCancel} disabled={busy}>取消</button>
      </div>
    </form>
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
