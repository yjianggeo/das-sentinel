import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
} from "react";
import { api, ApiError, type EventArtifact } from "../api";
import WaterfallRangeControls from "./WaterfallRangeControls";
import { useWaterfallPreview } from "./useWaterfallPreview";

const KIND_META: Record<string, { name: string; alt: string }> = {
  waterfall_raw: { name: "瀑布图", alt: "事件瀑布图" },
  waterfall_filtered: { name: "滤波瀑布图", alt: "事件滤波瀑布图" },
  waveform: { name: "波形", alt: "事件波形图" },
  psd: { name: "PSD", alt: "事件功率谱" },
  stft: { name: "STFT", alt: "事件时频图" },
};

const IMAGE_KINDS = Object.keys(KIND_META);

type ArtifactLoadState =
  | { kind: "loading"; eventId?: string }
  | { kind: "ready"; url: string; eventId: string }
  | { kind: "error"; message: string };

function isAbortError(cause: unknown): boolean {
  return cause instanceof Error && cause.name === "AbortError";
}

export function artifactUrl(id: string): string {
  return `/api/v1/artifacts/${id}`;
}

export default function ArtifactViewer({
  eventId,
  artifacts,
}: {
  eventId: string;
  artifacts: EventArtifact[];
}) {
  const images = artifacts.filter((item) => IMAGE_KINDS.includes(item.kind));
  const [activeKind, setActiveKind] = useState<string>(
    images[0]?.kind ?? "waterfall_raw",
  );
  const [fullscreen, setFullscreen] = useState(false);
  const [loadState, setLoadState] = useState<ArtifactLoadState>({
    kind: "loading",
  });
  const [retryNumber, setRetryNumber] = useState(0);
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const fullscreenTriggerRef = useRef<HTMLButtonElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const displayedEventIdRef = useRef(eventId);
  const { options, optionsError, preview, busy, error, generate, reset } = useWaterfallPreview(eventId);

  useEffect(() => {
    if (images.length > 0 && !images.some((item) => item.kind === activeKind)) {
      setActiveKind(images[0].kind);
    }
  }, [artifacts, activeKind, images]);

  const active = images.find((item) => item.kind === activeKind) ?? images[0];
  const activeId = active?.id;

  useEffect(() => {
    if (displayedEventIdRef.current !== eventId) {
      displayedEventIdRef.current = eventId;
      setLoadState({ kind: "loading", eventId });
    }
  }, [eventId]);

  useEffect(() => {
    if (!activeId) return;
    const controller = new AbortController();
    let current = true;
    let objectUrl: string | null = null;
    setLoadState({ kind: "loading" });
    api
      .getBlob(artifactUrl(activeId), controller.signal)
      .then((blob) => {
        if (!current) return;
        objectUrl = URL.createObjectURL(blob);
        setLoadState({ kind: "ready", url: objectUrl, eventId });
      })
      .catch((cause: unknown) => {
        if (!current || isAbortError(cause)) return;
        if (cause instanceof ApiError && cause.status === 401) return;
        setLoadState({
          kind: "error",
          message:
            cause instanceof ApiError && cause.status === 409
              ? "图件完整性校验失败，请联系管理员"
              : "图件加载失败",
        });
      });
    return () => {
      current = false;
      controller.abort();
      if (objectUrl !== null) URL.revokeObjectURL(objectUrl);
    };
  }, [activeId, retryNumber]);

  useEffect(() => {
    if (fullscreen) closeButtonRef.current?.focus();
  }, [fullscreen]);

  const closeFullscreen = useCallback(() => {
    setFullscreen(false);
    fullscreenTriggerRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!fullscreen) return;
    const handleDialogKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Tab") {
        event.preventDefault();
        closeButtonRef.current?.focus();
        return;
      }
      if (event.key === "Escape") closeFullscreen();
    };
    document.addEventListener("keydown", handleDialogKeyDown);
    return () => document.removeEventListener("keydown", handleDialogKeyDown);
  }, [closeFullscreen, fullscreen]);

  if (!active) {
    return <p className="loading">此事件没有图件。</p>;
  }
  const meta = KIND_META[active.kind] ?? { name: active.kind, alt: active.kind };
  const previewUrl =
    active.kind === "waterfall_raw"
      ? preview?.waterfall_raw
      : active.kind === "waterfall_filtered"
        ? preview?.waterfall_filtered
        : active.kind === "waveform"
          ? preview?.waveform
          : undefined;
  const imageUrl = previewUrl
    ?? (loadState.kind === "ready" && loadState.eventId === eventId
      ? loadState.url
      : loadState.kind === "loading" && loadState.eventId === eventId
        ? artifactUrl(active.id)
        : undefined);

  const selectIndex = (index: number) => {
    if (images[index].kind === active.kind) return;
    setLoadState({ kind: "loading" });
    setActiveKind(images[index].kind);
  };

  const handleTabKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    index: number,
  ) => {
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % images.length;
    if (event.key === "ArrowLeft") {
      nextIndex = (index - 1 + images.length) % images.length;
    }
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = images.length - 1;
    if (nextIndex === null) return;
    event.preventDefault();
    selectIndex(nextIndex);
    tabRefs.current[nextIndex]?.focus();
  };

  const openFullscreen = (event: MouseEvent<HTMLButtonElement>) => {
    fullscreenTriggerRef.current = event.currentTarget;
    setFullscreen(true);
  };

  const showDecodeError = () => {
    setLoadState({ kind: "error", message: "图件加载失败" });
    if (fullscreen) closeFullscreen();
  };

  const retry = () => {
    setLoadState({ kind: "loading" });
    setRetryNumber((current) => current + 1);
  };

  return (
    <div className="artifact-viewer">
      <WaterfallRangeControls
        options={options}
        optionsError={optionsError}
        busy={busy}
        error={error}
        onGenerate={generate}
        onReset={reset}
      />
      <div className="artifact-tabs">
        <div className="artifact-tablist" role="tablist" aria-label="图件类型">
          {images.map((item, index) => (
            <button
              key={item.kind}
              ref={(element) => {
                tabRefs.current[index] = element;
              }}
              id={`artifact-tab-${item.kind}`}
              type="button"
              role="tab"
              aria-controls={`artifact-panel-${item.kind}`}
              aria-selected={item.kind === active.kind}
              tabIndex={item.kind === active.kind ? 0 : -1}
              className={item.kind === active.kind ? "tab active" : "tab"}
              onClick={() => selectIndex(index)}
              onKeyDown={(event) => handleTabKeyDown(event, index)}
            >
              {KIND_META[item.kind]?.name ?? item.kind}
            </button>
          ))}
        </div>
        <a className="tab download" href={previewUrl ?? artifactUrl(active.id)} download>
          下载
        </a>
      </div>
      {images.map((item) => {
        const isActive = item.kind === active.kind;
        return (
          <div
            key={item.kind}
            id={`artifact-panel-${item.kind}`}
            role="tabpanel"
            aria-labelledby={`artifact-tab-${item.kind}`}
            className="artifact-panel"
            hidden={!isActive}
          >
            {isActive && !previewUrl && loadState.kind === "loading" && (
              <p className="loading">正在加载图件…</p>
            )}
            {isActive && !previewUrl && loadState.kind === "error" && (
              <div className="artifact-error">
                <p className="error" role="alert">{loadState.message}</p>
                <button type="button" className="subtle" onClick={retry}>
                  重试图件
                </button>
              </div>
            )}
            {isActive && imageUrl && (
              <>
                {previewUrl && <p>当前显示临时预览</p>}
                <button
                  type="button"
                  className="artifact-image-button"
                  onClick={openFullscreen}
                  aria-label="全屏查看"
                >
                  <img src={imageUrl} alt={meta.alt} onError={showDecodeError} />
                </button>
              </>
            )}
          </div>
        );
      })}
      {fullscreen && imageUrl && (
        <div className="artifact-fullscreen" onClick={closeFullscreen}>
          <div
            className="artifact-fullscreen-dialog"
            role="dialog"
            aria-modal="true"
            aria-label={`全屏${meta.name}`}
            onClick={(event) => event.stopPropagation()}
          >
            <button
              ref={closeButtonRef}
              type="button"
              className="artifact-fullscreen-close"
              onClick={closeFullscreen}
            >
              关闭全屏
            </button>
            <img src={imageUrl} alt={meta.alt} onError={showDecodeError} />
          </div>
        </div>
      )}
    </div>
  );
}
