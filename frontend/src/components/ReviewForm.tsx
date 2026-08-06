import { useLayoutEffect, useRef, useState, type FormEvent } from "react";
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
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const requestIdRef = useRef<string | null>(null);
  const eventGenerationRef = useRef(0);
  const activeSubmissionRef = useRef<object | null>(null);

  useLayoutEffect(() => {
    eventGenerationRef.current += 1;
    activeSubmissionRef.current = null;
    busyRef.current = false;
    requestIdRef.current = null;
    setBusy(false);
    setStatus("idle");
    setError("");
    return () => {
      eventGenerationRef.current += 1;
      activeSubmissionRef.current = null;
    };
  }, [eventId]);

  function onLabelChange(nextLabel: string) {
    if (busyRef.current || nextLabel === label) return;
    setLabel(nextLabel);
    requestIdRef.current = null;
    setStatus("idle");
    setError("");
  }

  function onNoteChange(nextNote: string) {
    if (busyRef.current || nextNote === note) return;
    setNote(nextNote);
    requestIdRef.current = null;
    setStatus("idle");
    setError("");
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!label || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    requestIdRef.current ??= crypto.randomUUID();
    const requestId = requestIdRef.current;
    const eventGeneration = eventGenerationRef.current;
    const submission = {};
    activeSubmissionRef.current = submission;
    const isCurrentSubmission = () =>
      eventGeneration === eventGenerationRef.current &&
      activeSubmissionRef.current === submission;
    try {
      await api.post(`/api/v1/events/${eventId}/reviews`, {
        label,
        note: note || null,
        request_id: requestId,
      });
      if (!isCurrentSubmission()) return;
      setStatus("saved");
      requestIdRef.current = null;
      onReviewed();
    } catch (cause) {
      if (!isCurrentSubmission()) return;
      setStatus("error");
      setError(cause instanceof ApiError ? cause.message : "保存复核失败");
    } finally {
      if (!isCurrentSubmission()) return;
      activeSubmissionRef.current = null;
      busyRef.current = false;
      setBusy(false);
    }
  }

  return (
    <div className="review-form">
      <form className="stack" onSubmit={(event) => void onSubmit(event)}>
        <label>
          人工标签
          <select
            value={label}
            onChange={(event) => onLabelChange(event.target.value)}
            disabled={busy}
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
            onChange={(event) => onNoteChange(event.target.value)}
            disabled={busy}
            rows={3}
          />
        </label>
        <button className="primary" type="submit" disabled={!label || busy}>
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
