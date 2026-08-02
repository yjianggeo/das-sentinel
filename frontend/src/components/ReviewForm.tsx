import { useState, type FormEvent } from "react";
import { api, ApiError, type EventSummary } from "../api";

const LABELS = [
  { value: "earthquake", name: "地震" },
  { value: "blast", name: "爆破" },
  { value: "traffic", name: "车辆" },
  { value: "noise", name: "噪声" },
] as const;

export default function ReviewForm({
  eventId,
  similar,
  onReviewed,
  onSelectSimilar,
}: {
  eventId: string;
  similar: EventSummary[];
  onReviewed: () => void;
  onSelectSimilar: (event: EventSummary) => void;
}) {
  const [label, setLabel] = useState("");
  const [note, setNote] = useState("");
  const [status, setStatus] = useState<"idle" | "saved" | "error">("idle");
  const [error, setError] = useState("");

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!label) return;
    try {
      await api.post(`/api/v1/events/${eventId}/reviews`, {
        label,
        note: note || null,
      });
      setStatus("saved");
      onReviewed();
    } catch (cause) {
      setStatus("error");
      setError(cause instanceof ApiError ? cause.message : "保存复核失败");
    }
  }

  return (
    <div className="review-form">
      <form className="stack" onSubmit={(event) => void onSubmit(event)}>
        <label>
          人工标签
          <select
            value={label}
            onChange={(event) => setLabel(event.target.value)}
            required
          >
            <option value="">请选择…</option>
            {LABELS.map((item) => (
              <option key={item.value} value={item.value}>
                {item.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          备注
          <textarea
            value={note}
            onChange={(event) => setNote(event.target.value)}
            rows={3}
          />
        </label>
        <button className="primary" type="submit" disabled={!label}>
          保存复核
        </button>
        {status === "saved" && <p role="status">已保存</p>}
        {status === "error" && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </form>
      {similar.length > 0 && (
        <section>
          <h3>相似事件</h3>
          <ul>
            {similar.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  className="link-button"
                  onClick={() => onSelectSimilar(item)}
                >
                  {item.occurred_at.replace("T", " ").replace(/(\.\d+)?Z$/, "")}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
