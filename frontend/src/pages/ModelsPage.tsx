import { useEffect, useState } from "react";
import { api, type Site } from "../api";
import WindowTraining from "../components/WindowTraining";
import { WindowActiveModel } from "../components/WindowPrediction";

export default function ModelsPage() {
  const [sites, setSites] = useState<Site[]>([]);
  const [siteId, setSiteId] = useState("");
  const [error, setError] = useState("");

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
      <WindowActiveModel key={`active-${siteId}`} siteId={siteId} />
      <WindowTraining key={siteId} siteId={siteId} />
    </section>
  );
}
