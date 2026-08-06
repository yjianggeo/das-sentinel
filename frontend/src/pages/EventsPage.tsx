import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  type EventDetail,
  type EventSummary,
} from "../api";
import ArtifactViewer from "../components/ArtifactViewer";
import EventQueue, { formatOccurredAt } from "../components/EventQueue";
import EvidencePanel from "../components/EvidencePanel";
import ReviewForm from "../components/ReviewForm";

const LABEL_NAMES: Record<string, string> = {
  earthquake: "地震",
  blast: "爆破",
  traffic: "车辆",
  noise: "噪声",
  unknown: "待定",
};

export default function EventsPage() {
  const [events, setEvents] = useState<EventSummary[] | null>(null);
  const [selected, setSelected] = useState<EventSummary | null>(null);
  const [detail, setDetail] = useState<EventDetail | null>(null);
  const [similar, setSimilar] = useState<EventSummary[]>([]);
  const [error, setError] = useState("");
  const requestSequence = useRef(0);
  const activeController = useRef<AbortController | null>(null);

  useEffect(() => {
    api
      .get<EventSummary[]>("/api/v1/events", { limit: "100" })
      .then(setEvents)
      .catch(() => setError("无法加载事件"));
  }, []);

  useEffect(() => () => activeController.current?.abort(), []);

  const openEvent = useCallback((event: EventSummary) => {
    const sequence = ++requestSequence.current;
    activeController.current?.abort();
    const controller = new AbortController();
    activeController.current = controller;
    setSelected(event);
    setDetail(null);
    setSimilar([]);
    setError("");
    api
      .get<EventDetail>(`/api/v1/events/${event.id}`, undefined, controller.signal)
      .then((value) => {
        if (sequence === requestSequence.current) setDetail(value);
      })
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        if (sequence === requestSequence.current) setError("无法加载事件详情");
      });
    api
      .get<EventSummary[]>(
        `/api/v1/events/${event.id}/similar`,
        undefined,
        controller.signal,
      )
      .then((value) => {
        if (sequence === requestSequence.current) setSimilar(value);
      })
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        if (sequence === requestSequence.current) setSimilar([]);
      });
  }, []);

  const reloadEvents = useCallback(() => {
    api
      .get<EventSummary[]>("/api/v1/events", { limit: "100" })
      .then(setEvents)
      .catch(() => undefined);
    if (selected) {
      const sequence = requestSequence.current;
      api
        .get<EventDetail>(`/api/v1/events/${selected.id}`)
        .then((value) => {
          if (sequence === requestSequence.current) setDetail(value);
        })
        .catch(() => undefined);
    }
  }, [selected]);

  if (events === null) {
    return <p className="loading">正在加载…</p>;
  }
  return (
    <section className="events-layout">
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <EventQueue events={events} selectedId={selected?.id ?? null} onSelect={openEvent} />
      <div className="event-detail">
        {selected === null && <p className="loading">从左侧队列选择一个事件。</p>}
        {selected !== null && detail === null && (
          <p className="loading">正在加载详情…</p>
        )}
        {selected !== null && detail !== null && selected.id === detail.id && (
          <>
            <header className="event-detail-header">
              <h2>{formatOccurredAt(detail.occurred_at)}</h2>
              <p>
                机器标签：
                <span className={`status-pill label-${detail.machine_label}`}>
                  {LABEL_NAMES[detail.machine_label]}
                </span>
                　人工复核：
                {detail.latest_review
                  ? LABEL_NAMES[detail.latest_review.label] ??
                    detail.latest_review.label
                  : "未复核"}
              </p>
            </header>
            <div className="event-detail-body">
              <div className="event-artifacts">
                <ArtifactViewer artifacts={detail.artifacts} />
              </div>
              <EvidencePanel event={detail} />
            </div>
            <ReviewForm
              eventId={detail.id}
              similar={similar}
              onReviewed={reloadEvents}
              onSelectSimilar={openEvent}
            />
          </>
        )}
      </div>
    </section>
  );
}
