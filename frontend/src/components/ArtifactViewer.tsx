import { useEffect, useState } from "react";
import type { EventArtifact } from "../api";

const KIND_META: Record<string, { name: string; alt: string }> = {
  waterfall_raw: { name: "瀑布图", alt: "事件瀑布图" },
  waterfall_filtered: { name: "滤波瀑布图", alt: "事件滤波瀑布图" },
  waveform: { name: "波形", alt: "事件波形图" },
  psd: { name: "PSD", alt: "事件功率谱" },
  stft: { name: "STFT", alt: "事件时频图" },
};

const IMAGE_KINDS = Object.keys(KIND_META);

export function artifactUrl(id: string): string {
  return `/api/v1/artifacts/${id}`;
}

export default function ArtifactViewer({
  artifacts,
}: {
  artifacts: EventArtifact[];
}) {
  const images = artifacts.filter((item) => IMAGE_KINDS.includes(item.kind));
  const [activeKind, setActiveKind] = useState<string>(
    images[0]?.kind ?? "waterfall_raw",
  );
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    if (images.length > 0 && !images.some((item) => item.kind === activeKind)) {
      setActiveKind(images[0].kind);
    }
  }, [artifacts, activeKind, images]);

  const active = images.find((item) => item.kind === activeKind) ?? images[0];
  if (!active) {
    return <p className="loading">此事件没有图件。</p>;
  }
  const meta = KIND_META[active.kind] ?? { name: active.kind, alt: active.kind };
  return (
    <div className="artifact-viewer">
      <div className="artifact-tabs" role="tablist" aria-label="图件类型">
        {images.map((item) => (
          <button
            key={item.kind}
            type="button"
            role="tab"
            aria-selected={item.kind === active.kind}
            className={item.kind === active.kind ? "tab active" : "tab"}
            onClick={() => setActiveKind(item.kind)}
          >
            {KIND_META[item.kind]?.name ?? item.kind}
          </button>
        ))}
        <a className="tab download" href={artifactUrl(active.id)} download>
          下载
        </a>
      </div>
      <button
        type="button"
        className="artifact-image-button"
        onClick={() => setFullscreen(true)}
        aria-label="全屏查看"
      >
        <img src={artifactUrl(active.id)} alt={meta.alt} />
      </button>
      {fullscreen && (
        <div
          className="artifact-fullscreen"
          role="dialog"
          aria-label={`全屏${meta.name}`}
          onClick={() => setFullscreen(false)}
        >
          <img src={artifactUrl(active.id)} alt={meta.alt} />
        </div>
      )}
    </div>
  );
}
