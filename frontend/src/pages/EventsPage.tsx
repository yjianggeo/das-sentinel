import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  api,
  type EventDetail,
  type EventSummary,
} from "../api";
import ArtifactViewer from "../components/ArtifactViewer";
import EventQueue, { formatOccurredAt } from "../components/EventQueue";
import EvidencePanel, { EventFeatures } from "../components/EvidencePanel";
import ReviewForm from "../components/ReviewForm";

const LABEL_NAMES: Record<string, string> = {
  earthquake: "地震",
  blast: "爆破",
  traffic: "车辆",
  noise: "噪声",
  unknown: "待定",
};

const PAGE_SIZE = 100;

export interface EventFilters {
  machineLabel: "" | EventSummary["machine_label"];
  reviewed: "" | "true" | "false";
}

type CollectionState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | {
      kind: "ready";
      items: EventSummary[];
      hasMore: boolean;
      loadingMore: boolean;
    };

function isAbortError(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === "AbortError";
}

export default function EventsPage() {
  const [filters, setFilters] = useState<EventFilters>({
    machineLabel: "",
    reviewed: "",
  });
  const [collection, setCollection] = useState<CollectionState>({ kind: "loading" });
  const [reloadVersion, setReloadVersion] = useState(0);
  const [moreError, setMoreError] = useState("");
  const [selected, setSelected] = useState<EventSummary | null>(null);
  const [detail, setDetail] = useState<EventDetail | null>(null);
  const [similar, setSimilar] = useState<EventSummary[]>([]);
  const [detailError, setDetailError] = useState("");
  const requestSequence = useRef(0);
  const collectionSequence = useRef(0);
  const activeDetailController = useRef<AbortController | null>(null);
  const activeCollectionController = useRef<AbortController | null>(null);
  const selectedRef = useRef<EventSummary | null>(null);
  const nextOffset = useRef(0);
  const layoutRef = useRef<HTMLElement>(null);
  const detailRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const layout = layoutRef.current;
    const detailElement = detailRef.current;
    const form = detailElement?.querySelector<HTMLFormElement>(".review-form form");
    if (!layout || !detailElement || !form) return;

    const alignPanels = () => {
      if (window.innerWidth <= 900) {
        layout.style.removeProperty("--event-panel-height");
        return;
      }
      const height = form.getBoundingClientRect().bottom - layout.getBoundingClientRect().top;
      layout.style.setProperty("--event-panel-height", `${Math.max(0, height)}px`);
    };
    alignPanels();
    const observer = new ResizeObserver(alignPanels);
    observer.observe(detailElement);
    observer.observe(form);
    window.addEventListener("resize", alignPanels);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", alignPanels);
      layout.style.removeProperty("--event-panel-height");
    };
  }, [selected?.id, detail?.id, collection.kind]);

  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);

  const clearSelected = useCallback(() => {
    requestSequence.current += 1;
    activeDetailController.current?.abort();
    setSelected(null);
    setDetail(null);
    setSimilar([]);
    setDetailError("");
  }, []);

  useEffect(() => {
    activeCollectionController.current?.abort();
    const controller = new AbortController();
    activeCollectionController.current = controller;
    const sequence = ++collectionSequence.current;
    const params: Record<string, string> = {
      limit: String(PAGE_SIZE),
      offset: "0",
    };
    if (filters.machineLabel) params.machine_label = filters.machineLabel;
    if (filters.reviewed) params.reviewed = filters.reviewed;
    nextOffset.current = 0;
    setCollection({ kind: "loading" });
    setMoreError("");
    api
      .get<EventSummary[]>("/api/v1/events", params, controller.signal)
      .then((items) => {
        if (sequence !== collectionSequence.current) return;
        setCollection({
          kind: "ready",
          items,
          hasMore: items.length === PAGE_SIZE,
          loadingMore: false,
        });
        nextOffset.current = items.length;
        const currentSelected = selectedRef.current;
        if (currentSelected && !items.some((event) => event.id === currentSelected.id)) {
          clearSelected();
        }
      })
      .catch((cause: unknown) => {
        if (isAbortError(cause) || sequence !== collectionSequence.current) return;
        setCollection({ kind: "error", message: "无法加载事件" });
      });
    return () => controller.abort();
  }, [clearSelected, filters.machineLabel, filters.reviewed, reloadVersion]);

  useEffect(
    () => () => {
      activeDetailController.current?.abort();
      activeCollectionController.current?.abort();
    },
    [],
  );

  const openEvent = useCallback((event: EventSummary, preserveDetail = false) => {
    const sequence = ++requestSequence.current;
    activeDetailController.current?.abort();
    const controller = new AbortController();
    activeDetailController.current = controller;
    setSelected(event);
    if (!preserveDetail) setDetail(null);
    setSimilar([]);
    setDetailError("");
    api
      .get<EventDetail>(`/api/v1/events/${event.id}`, undefined, controller.signal)
      .then((value) => {
        if (sequence === requestSequence.current) setDetail(value);
      })
      .catch((cause: unknown) => {
        if (isAbortError(cause)) return;
        if (sequence === requestSequence.current) setDetailError("无法加载事件详情");
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
        if (isAbortError(cause)) return;
        if (sequence === requestSequence.current) setSimilar([]);
      });
  }, []);

  const loadMore = useCallback(() => {
    if (collection.kind !== "ready" || !collection.hasMore || collection.loadingMore) return;
    activeCollectionController.current?.abort();
    const controller = new AbortController();
    activeCollectionController.current = controller;
    const sequence = collectionSequence.current;
    const params: Record<string, string> = {
      limit: String(PAGE_SIZE),
      offset: String(nextOffset.current),
    };
    if (filters.machineLabel) params.machine_label = filters.machineLabel;
    if (filters.reviewed) params.reviewed = filters.reviewed;
    setCollection({ ...collection, loadingMore: true });
    setMoreError("");
    api
      .get<EventSummary[]>("/api/v1/events", params, controller.signal)
      .then((page) => {
        if (sequence !== collectionSequence.current) return;
        nextOffset.current += page.length;
        setCollection((current) => {
          if (current.kind !== "ready") return current;
          const seen = new Set(current.items.map((event) => event.id));
          const newItems = page.filter((event) => {
            if (seen.has(event.id)) return false;
            seen.add(event.id);
            return true;
          });
          return {
            kind: "ready",
            items: [...current.items, ...newItems],
            hasMore: page.length === PAGE_SIZE,
            loadingMore: false,
          };
        });
      })
      .catch((cause: unknown) => {
        if (isAbortError(cause) || sequence !== collectionSequence.current) return;
        setCollection((current) =>
          current.kind === "ready" ? { ...current, loadingMore: false } : current,
        );
        setMoreError("无法加载更多事件");
      });
  }, [collection, filters.machineLabel, filters.reviewed]);

  const reloadEvents = useCallback(() => {
    setReloadVersion((current) => current + 1);
    if (selectedRef.current) openEvent(selectedRef.current, true);
  }, [openEvent]);

  if (collection.kind === "error") {
    return (
      <div>
        <p className="error" role="alert">{collection.message}</p>
        <button type="button" className="subtle" onClick={reloadEvents}>重试</button>
      </div>
    );
  }

  return (
    <section className="events-layout" ref={layoutRef}>
      <EventQueue
        events={collection.kind === "ready" ? collection.items : []}
        selectedId={selected?.id ?? null}
        onSelect={openEvent}
        machineLabel={filters.machineLabel}
        reviewed={filters.reviewed}
        onMachineLabelChange={(machineLabel) =>
          setFilters((current) => ({ ...current, machineLabel }))
        }
        onReviewedChange={(reviewed) => setFilters((current) => ({ ...current, reviewed }))}
        hasMore={collection.kind === "ready" && collection.hasMore}
        loadingMore={collection.kind === "ready" && collection.loadingMore}
        onLoadMore={loadMore}
        showEmpty={collection.kind === "ready"}
        loading={collection.kind === "loading"}
        loadMoreError={moreError}
      />
      <div className="event-detail" ref={detailRef}>
        {detailError && <p className="error" role="alert">{detailError}</p>}
        {selected === null && <p className="loading">从左侧队列选择一个事件。</p>}
        {selected !== null && detail === null && !detailError && (
          <p className="loading">正在加载详情…</p>
        )}
        {selected !== null && detail !== null && selected.id === detail.id && (
          <>
            <header className="event-detail-header">
              <h2>{formatOccurredAt(detail.occurred_at, detail.site_timezone)}</h2>
              <p>
                机器标签：
                <span className={`status-pill label-${detail.machine_label}`}>
                  {LABEL_NAMES[detail.machine_label]}
                </span>
                　人工复核：
                {detail.latest_review
                  ? LABEL_NAMES[detail.latest_review.label] ?? detail.latest_review.label
                  : "未复核"}
              </p>
            </header>
            <div className="event-detail-body">
              <div className="event-artifacts">
                <ArtifactViewer eventId={detail.id} artifacts={detail.artifacts} />
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
      {selected !== null && detail !== null && selected.id === detail.id && (
        <EventFeatures event={detail} />
      )}
    </section>
  );
}
