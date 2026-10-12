import { useEffect, useRef, useState, type CSSProperties } from "react";
import { api } from "../api";
import { windowApi, type WindowDetail, type WindowDetectionRegion, type WindowPlotGeometry } from "../windowApi";

function message(cause: unknown) { return cause instanceof Error ? cause.message : String(cause); }

type CachedImage = { url: string; bytes: number; users: number };
const IMAGE_CACHE_BYTES = 32 * 1024 * 1024;

function useImageBlob(path: string | undefined, cache: Map<string, CachedImage>) {
    const [revision, setRevision] = useState(0);
    const [image, setImage] = useState<{ path: string; revision: number; url: string; error: string } | null>(null);
    useEffect(() => {
        setImage(null);
        if (!path) return;
        const controller = new AbortController();
        let entry = cache.get(path);
        let transientUrl = "";
        if (entry) {
            entry.users++;
            cache.delete(path);
            cache.set(path, entry);
            setImage({ path, revision, url: entry.url, error: "" });
        } else {
            void api.getBlob(path, controller.signal).then(blob => {
                if (controller.signal.aborted) return;
                const url = URL.createObjectURL(blob);
                if (blob.size <= IMAGE_CACHE_BYTES) {
                    let bytes = Array.from(cache.values()).reduce((sum, value) => sum + value.bytes, 0);
                    while (cache.size >= 4 || bytes + blob.size > IMAGE_CACHE_BYTES) {
                        const oldest = Array.from(cache.entries()).find(([, value]) => value.users === 0);
                        if (!oldest) break;
                        cache.delete(oldest[0]);
                        bytes -= oldest[1].bytes;
                        URL.revokeObjectURL(oldest[1].url);
                    }
                    if (cache.size < 4 && bytes + blob.size <= IMAGE_CACHE_BYTES) {
                        entry = { url, bytes: blob.size, users: 1 };
                        cache.set(path, entry);
                    }
                }
                if (!entry) transientUrl = url;
                setImage({ path, revision, url, error: "" });
            }).catch(cause => {
                if (!controller.signal.aborted) setImage({ path, revision, url: "", error: message(cause) });
            });
        }
        return () => {
            controller.abort();
            if (entry) entry.users--;
            if (transientUrl) URL.revokeObjectURL(transientUrl);
        };
    }, [path, revision, cache]);
    const current = image?.path === path && image?.revision === revision ? image : null;
    return {
        url: current?.url ?? "", error: current?.error ?? "",
        retry: () => setRevision(value => value + 1),
        decodeError: () => {
            const invalid = path && cache.get(path);
            if (invalid) { cache.delete(path!); URL.revokeObjectURL(invalid.url); }
            setImage(value => value && value.path === path && value.revision === revision ? { ...value, error: "图像无法解码" } : value);
        },
    };
}

function regionStyle(region: WindowDetectionRegion, geometry: WindowPlotGeometry, seconds: number): CSSProperties {
    const [left, top, width, height] = geometry.axes;
    const distanceSpan = geometry.distance_max_m - geometry.distance_min_m;
    return {
        left: `${100 * (left + width * (region.start_sample / 100) / seconds)}%`,
        top: `${100 * (top + height * (geometry.distance_max_m - region.distance_stop_m) / distanceSpan)}%`,
        width: `${100 * width * ((region.end_sample - region.start_sample) / 100) / seconds}%`,
        height: `${100 * height * (region.distance_stop_m - region.distance_start_m) / distanceSpan}%`,
    };
}

function SignalViewer({ detail }: { detail: WindowDetail }) {
    const cache = useRef(new Map<string, CachedImage>());
    useEffect(() => () => {
        for (const entry of cache.current.values()) URL.revokeObjectURL(entry.url);
        cache.current.clear();
    }, []);
    const [kind, setKind] = useState(detail.artifacts.some(artifact => artifact.kind === "waterfall") ? "waterfall" : "waterfall_filtered");
    const [selectedId, setSelectedId] = useState<number | null>(null);
    const [expanded, setExpanded] = useState(false);
    const [showRegions, setShowRegions] = useState(false);
    const [enlarged, setEnlarged] = useState<"full" | "preview" | null>(null);
    const dialog = useRef<HTMLDivElement>(null);
    const fullButton = useRef<HTMLButtonElement>(null);
    const previewButton = useRef<HTMLButtonElement>(null);
    const regionToggle = useRef<HTMLInputElement>(null);
    const current = detail.artifacts.find(artifact => artifact.kind === kind);
    const seconds = detail.window_seconds ?? (Date.parse(detail.end_utc) - Date.parse(detail.start_utc)) / 1000;
    const filtered = kind === "waterfall_filtered";
    const regions = detail.visualization_version === 2 ? detail.detection_regions ?? [] : [];
    const geometry = detail.visualization_version === 2 ? detail.plot_geometry : null;
    const selected = regions.find(region => region.id === selectedId);
    const canPreview = detail.status === "succeeded" && Boolean(detail.analysis_id && current);
    const previewPath = expanded && canPreview && selected && detail.analysis_id ? windowApi.regionPreviewUrl(detail.id, detail.analysis_id, selected.id, filtered, showRegions) : undefined;
    const fullPath = detail.waterfall_regions_baked !== false && current && detail.analysis_id && regions.length > 0 && !showRegions ? windowApi.waterfallPreviewUrl(detail.id, detail.analysis_id, filtered) : current?.url;
    const fullImage = useImageBlob(fullPath, cache.current);
    const preview = useImageBlob(previewPath, cache.current);
    const tabs = [
        { kind: "waterfall", label: "原始瀑布图" },
        { kind: "waterfall_filtered", label: "0.5–40 Hz 滤波瀑布图" },
    ];
    const needsBackfill = detail.visualization_version !== 2 || !geometry || !detail.artifacts.some(artifact => artifact.kind === "waterfall_filtered");
    const fullAlt = `${seconds} 秒全通道${filtered ? "滤波" : "原始"}瀑布图`;
    const previewAlt = `区域 ${selected?.id} ${filtered ? "滤波" : "原始"}瀑布图局部预览`;
    const enlargedImage = enlarged === "preview" ? preview : fullImage;
    const enlargedAlt = enlarged === "preview" ? previewAlt : fullAlt;
    const fullDownloadLabel = detail.waterfall_regions_baked === false && showRegions && regions.length > 0 ? "下载原图（不含疑似区域框）" : "下载图件";
    const regionBoxes = () => showRegions && geometry && geometry.distance_max_m > geometry.distance_min_m && regions.map(region => <button className="window-signal-box" key={region.id} style={regionStyle(region, geometry, seconds)} aria-label={`查看区域 ${region.id}`} aria-pressed={selectedId === region.id} disabled={!canPreview} onClick={() => { setSelectedId(region.id); setExpanded(true); }}><span>{region.id}</span></button>);

    useEffect(() => {
        if (!enlarged) return;
        const previousOverflow = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        const focusable = () => Array.from(dialog.current?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), a[href]') ?? []);
        focusable()[0]?.focus();
        const onKey = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                event.preventDefault();
                setEnlarged(null);
            }
            if (event.key !== "Tab") return;
            const elements = focusable();
            const first = elements[0];
            const last = elements[elements.length - 1];
            if (!elements.includes(document.activeElement as HTMLElement)) {
                event.preventDefault();
                (event.shiftKey ? last : first)?.focus();
            } else if (event.shiftKey && document.activeElement === first) {
                event.preventDefault(); last?.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault(); first?.focus();
            }
        };
        document.addEventListener("keydown", onKey);
        return () => {
            document.removeEventListener("keydown", onKey);
            document.body.style.overflow = previousOverflow;
            const trigger = enlarged === "preview" ? previewButton.current : fullButton.current;
            (trigger ?? regionToggle.current)?.focus();
        };
    }, [enlarged]);

    return <section className="window-image window-signal-viewer" aria-label="整窗图件">
        {detail.status === "failed" ? <p role="alert">{detail.analysis_error_code === "coverage_incomplete" ? "原始数据或前后文不连续，未生成整窗图件。需补齐数据并重新规划扫描后才能复核。" : `整窗分析失败，未生成可用图件。${detail.analysis_error_message ?? "请在任务页查看失败原因。"}`}</p> : needsBackfill && detail.artifacts.some(artifact => artifact.kind === "waterfall") && <p>滤波图与局部标框待补图；已有原始瀑布图仍可查看。</p>}
        <div className="window-toolbar" role="tablist" aria-label="整窗图类型">{tabs.map(tab => <button key={tab.kind} role="tab" disabled={!detail.artifacts.some(artifact => artifact.kind === tab.kind)} tabIndex={kind === tab.kind ? 0 : -1} aria-selected={kind === tab.kind} onClick={() => setKind(tab.kind)} onKeyDown={event => {
            if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(event.key)) return;
            event.preventDefault();
            const available = Array.from(event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>("button:not([disabled])") ?? []);
            const index = available.indexOf(event.currentTarget);
            const next = event.key === "Home" ? 0 : event.key === "End" ? available.length - 1 : (index + (event.key === "ArrowRight" ? 1 : available.length - 1)) % available.length;
            available[next]?.click();
            available[next]?.focus();
        }}>{tab.label}</button>)}</div>
        <p>100 Hz 全通道 · 原始图未带通，滤波图为 0.5–40 Hz · 图中秒数从本窗起点计算</p>
        <div className="window-toolbar"><label><input ref={regionToggle} type="checkbox" checked={showRegions} onChange={event => setShowRegions(event.target.checked)}/>显示疑似区域</label><span>单击图片放大查看。</span></div>
        {!current ? <p>整窗图件尚未生成</p> : fullImage.error ? <div role="alert">图件加载失败：{fullImage.error} <button onClick={fullImage.retry}>重试图件</button></div> : fullImage.url ? <div className="window-signal-frame">
            <button ref={fullButton} className="window-signal-image-open" aria-label={`放大${fullAlt}`} onClick={() => setEnlarged("full")}><img src={fullImage.url} alt={fullAlt} onError={fullImage.decodeError}/></button>
            {regionBoxes()}
        </div> : <p role="status">正在加载图件…</p>}
        {regions.length > 0 ? <details className="window-review-details" open={expanded} onToggle={event => setExpanded(event.currentTarget.open)}><summary>疑似区域与局部预览（{regions.length}）</summary>{expanded && <div className="window-signal-review">
            <section aria-label="疑似信号区域"><h3>疑似区域（{regions.length}）</h3><p>点击图中编号或下方区域查看局部。</p><div className="window-signal-regions">{regions.map(region => <button key={region.id} aria-pressed={selectedId === region.id} disabled={!canPreview} onClick={() => setSelectedId(region.id)}>
                <strong>区域 {region.id} · {(region.start_sample / 100).toFixed(2)}–{(region.end_sample / 100).toFixed(2)} 秒</strong>
                <span>{region.distance_start_m.toFixed(1)}–{region.distance_stop_m.toFixed(1)} 米 · 分数 {Number(region.score.toPrecision(4))}</span>
                <small>{region.detectors.join(" / ")} · {region.evidence_count} 条证据</small>
            </button>)}</div></section>
            <section className="window-signal-preview" aria-label="局部预览"><h3>{selected ? `区域 ${selected.id} 局部预览` : "局部预览"}</h3><p>标注仍保存到整个 {seconds} 秒窗口，局部框仅作判断线索。</p>
                {!selected ? <p>选择疑似区域放大查看。</p> : !previewPath ? <p>当前分析暂不支持局部预览。</p> : preview.error ? <div role="alert">局部预览加载失败：{preview.error} <button onClick={preview.retry}>重试局部预览</button></div> : preview.url ? <div className="window-signal-preview-image"><button ref={previewButton} className="window-signal-image-open" aria-label={`放大${previewAlt}`} onClick={() => setEnlarged("preview")}><img src={preview.url} alt={previewAlt} onError={preview.decodeError}/></button></div> : <p role="status">正在加载局部预览…</p>}
            </section>
        </div>}</details> : !needsBackfill && <p>本窗无合格局部区域；是否仅背景仍需人工复核。</p>}
        <div className="window-toolbar">{current && fullImage.url && !fullImage.error && <a href={fullImage.url} download={`${current.kind}.png`}>{fullDownloadLabel}</a>}</div>
        {enlarged && <div ref={dialog} className="window-image-dialog" role="dialog" aria-modal="true" aria-label={`${enlargedAlt}放大`}>
            <h3>{enlargedAlt}</h3>
            <div className="window-toolbar"><button onClick={() => setEnlarged(null)}>关闭放大图片</button><label><input type="checkbox" checked={showRegions} onChange={event => setShowRegions(event.target.checked)}/>显示疑似区域</label></div>
            {enlargedImage.error ? <div role="alert">图件加载失败：{enlargedImage.error} <button onClick={enlargedImage.retry}>重试图件</button></div> : enlargedImage.url ? <div className="window-signal-frame">
                <img src={enlargedImage.url} alt={enlargedAlt} onError={enlargedImage.decodeError}/>
                {enlarged === "full" && regionBoxes()}
            </div> : <p role="status">正在加载图件…</p>}
            {current && enlargedImage.url && !enlargedImage.error && <div className="window-toolbar"><a href={enlargedImage.url} download={`${current.kind}${enlarged === "preview" ? `-region-${selected?.id}` : ""}.png`}>{enlarged === "full" ? fullDownloadLabel : "下载图件"}</a></div>}
        </div>}
    </section>;
}

export default function WindowSignalViewer({ detail }: { detail: WindowDetail }) {
    return <SignalViewer key={`${detail.id}:${detail.analysis_id}:${detail.input_sha256}`} detail={detail}/>;
}
