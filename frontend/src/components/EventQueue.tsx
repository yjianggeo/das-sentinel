import { useState } from "react";
import type { EventSummary } from "../api";

const LABEL_NAMES: Record<string, string> = {
  earthquake: "地震",
  blast: "爆破",
  traffic: "车辆",
  noise: "噪声",
  unknown: "待定",
};

export function formatOccurredAt(iso: string): string {
  return iso.replace("T", " ").replace(/(\.\d+)?Z$/, "");
}

export default function EventQueue({
  events,
  selectedId,
  onSelect,
}: {
  events: EventSummary[];
  selectedId: string | null;
  onSelect: (event: EventSummary) => void;
}) {
  const [labelFilter, setLabelFilter] = useState("");
  const [reviewFilter, setReviewFilter] = useState("");

  const filtered = events.filter((event) => {
    if (labelFilter && event.machine_label !== labelFilter) return false;
    if (reviewFilter === "yes" && event.latest_review === null) return false;
    if (reviewFilter === "no" && event.latest_review !== null) return false;
    return true;
  });

  return (
    <aside className="event-queue" aria-label="事件队列">
      <div className="queue-filters">
        <label>
          机器标签
          <select
            value={labelFilter}
            onChange={(event) => setLabelFilter(event.target.value)}
          >
            <option value="">全部</option>
            {Object.entries(LABEL_NAMES).map(([value, name]) => (
              <option key={value} value={value}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label>
          复核状态
          <select
            value={reviewFilter}
            onChange={(event) => setReviewFilter(event.target.value)}
          >
            <option value="">全部</option>
            <option value="yes">已复核</option>
            <option value="no">未复核</option>
          </select>
        </label>
      </div>
      <ul className="queue-list">
        {filtered.map((event) => (
          <li key={event.id}>
            <button
              type="button"
              className={event.id === selectedId ? "queue-item selected" : "queue-item"}
              onClick={() => onSelect(event)}
            >
              <span className="queue-time">
                {formatOccurredAt(event.occurred_at)}
              </span>
              <span className={`status-pill label-${event.machine_label}`}>
                {LABEL_NAMES[event.machine_label]}
              </span>
              {event.latest_review && (
                <span className="queue-review">
                  ✓ {LABEL_NAMES[event.latest_review.label] ?? event.latest_review.label}
                </span>
              )}
            </button>
          </li>
        ))}
        {filtered.length === 0 && <li className="loading">无匹配事件</li>}
      </ul>
    </aside>
  );
}
