import type { EventSummary } from "../api";

const LABEL_NAMES: Record<string, string> = {
  earthquake: "地震",
  blast: "爆破",
  traffic: "车辆",
  noise: "噪声",
  unknown: "待定",
};

export function formatOccurredAt(iso: string, timeZone: string): string {
  const parts = new Intl.DateTimeFormat("zh-CN", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
  }).formatToParts(new Date(iso));
  const value = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${value.year}-${value.month}-${value.day} ${value.hour}:${value.minute}:${value.second}`;
}

export default function EventQueue({
  events,
  selectedId,
  onSelect,
  machineLabel,
  reviewed,
  onMachineLabelChange,
  onReviewedChange,
  hasMore,
  loadingMore,
  onLoadMore,
  showEmpty,
}: {
  events: EventSummary[];
  selectedId: string | null;
  onSelect: (event: EventSummary) => void;
  machineLabel: "" | EventSummary["machine_label"];
  reviewed: "" | "true" | "false";
  onMachineLabelChange: (value: "" | EventSummary["machine_label"]) => void;
  onReviewedChange: (value: "" | "true" | "false") => void;
  hasMore: boolean;
  loadingMore: boolean;
  onLoadMore: () => void;
  showEmpty: boolean;
}) {
  return (
    <aside className="event-queue" aria-label="事件队列">
      <div className="queue-filters">
        <label>
          机器标签
          <select
            value={machineLabel}
            onChange={(event) =>
              onMachineLabelChange(event.target.value as "" | EventSummary["machine_label"])
            }
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
            value={reviewed}
            onChange={(event) =>
              onReviewedChange(event.target.value as "" | "true" | "false")
            }
          >
            <option value="">全部</option>
            <option value="true">已复核</option>
            <option value="false">未复核</option>
          </select>
        </label>
      </div>
      <ul className="queue-list">
        {events.map((event) => (
          <li key={event.id}>
            <button
              type="button"
              className={event.id === selectedId ? "queue-item selected" : "queue-item"}
              onClick={() => onSelect(event)}
              aria-label={`事件 ${event.id}`}
            >
              <span className="queue-time">
                {formatOccurredAt(event.occurred_at, event.site_timezone)}
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
        {showEmpty && events.length === 0 && <li className="loading">暂无事件</li>}
      </ul>
      {hasMore && (
        <button
          type="button"
          className="subtle queue-load-more"
          onClick={onLoadMore}
          disabled={loadingMore}
        >
          {loadingMore ? "正在加载更多…" : "加载更多"}
        </button>
      )}
    </aside>
  );
}
