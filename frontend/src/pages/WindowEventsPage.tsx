import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, ApiError, type Site } from "../api";
import { formatOccurredAt } from "../components/EventQueue";
import { windowApi, windowStatusNames, type WindowDetail, type WindowItem, type WindowPage, type WindowPresence, type WindowReviewCreate, type WindowReview } from "../windowApi";
import WindowSignalViewer from "../components/WindowSignalViewer";
import WindowPrediction from "../components/WindowPrediction";
import "../components/window-events.css";
const CLASS_NAMES: Record<Exclude<WindowPresence, "unknown">, string> = {
    present: "地震类（含爆破）",
    absent: "噪声类（含车辆）",
};
const SUMMARY_FEATURES = [
    { key: "active_channel_fraction", name: "活跃通道占比", scale: 100, digits: 1, unit: "%", hint: "识别范围内出现明显能量变化的通道" },
    { key: "active_contiguous_span_m", name: "连续活跃范围", scale: 1, digits: 1, unit: " 米", hint: "连续活跃通道覆盖的最长距离" },
    { key: "lag_correlation_median", name: "通道相关性", scale: 1, digits: 2, unit: "", hint: "时移相关中位数；正负表示同相或反相" },
    { key: "valid_channel_fraction", name: "有效通道占比", scale: 100, digits: 1, unit: "%", hint: "全窗口有效通道占全部通道的比例" },
];
function coverageScoreText(coverage: WindowItem["spatial_coverage"]): string {
    return typeof coverage?.score === "number" && Number.isFinite(coverage.score) ? `${coverage.score.toFixed(1)} / 100` : "待测量";
}
function coverageFractionText(value: number | null | undefined): string {
    return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "暂无";
}
function coveragePropagationText(coverage: NonNullable<WindowItem["spatial_coverage"]>): string {
    const velocity = coverage.apparent_velocity_mps;
    const confidence = coverage.propagation_confidence;
    if (!coverage.propagation_status) return "待测传播";
    if (coverage.propagation_status === "unresolved_zero_lag") return "时延未分辨";
    if (coverage.propagation_status === "ambiguous_periodic") return "周期多峰，传播不确定";
    if (coverage.propagation_status !== "resolved" || typeof velocity !== "number" || !Number.isFinite(velocity) || velocity <= 0
        || typeof confidence !== "number" || !Number.isFinite(confidence) || confidence < .6 || confidence > 1) return "传播未确定";
    const direction = coverage.propagation_direction === "+" ? "向距离增大方向" : coverage.propagation_direction === "-" ? "向距离减小方向" : "方向待定";
    return `视速度 ${(velocity / 1000).toFixed(2)} km/s · ${direction} · 可信度 ${(confidence * 100).toFixed(0)}%`;
}
function coverageCommonModeText(coverage: NonNullable<WindowItem["spatial_coverage"]>): string {
    let risk = coverage.common_mode_status === "not_supported" ? "未见持续共模证据" : "共模待定";
    if (coverage.algorithm_version === "borehole-coverage-v3" && coverage.common_mode_status === "not_supported") risk = "未见强同相共模证据";
    if (coverage.common_mode_status === "persistent_supported") risk = "持续共模风险";
    if (coverage.common_mode_status === "component_supported") {
        const residualScore = typeof coverage.residual_coverage_score === "number" && Number.isFinite(coverage.residual_coverage_score) ? `${coverage.residual_coverage_score.toFixed(1)} / 100` : "暂无";
        risk = `强同相共模 · 解释占比 ${coverageFractionText(coverage.common_mode_energy_fraction)} · 残余覆盖 ${residualScore}`;
    }
    if (coverage.review_protected === true) return `${risk} · 已人工确认地震，保留原复核优先级`;
    if (coverage.common_mode_status !== "persistent_supported" && coverage.common_mode_status !== "component_supported") return risk;
    const originalScore = typeof coverage.coverage_score === "number" && Number.isFinite(coverage.coverage_score) ? `${coverage.coverage_score.toFixed(1)} / 100` : "待测量";
    return typeof coverage.common_mode_penalty === "number" && Number.isFinite(coverage.common_mode_penalty) && coverage.common_mode_penalty > 0
        ? `${risk}，复核分已下调（原覆盖分 ${originalScore} → 调整分 ${coverageScoreText(coverage)}）`
        : risk;
}
function ReviewSummary({ detail }: { detail: WindowDetail }) {
    const coverage = detail.spatial_coverage;
    return <section aria-label="复核摘要" className="window-review-summary">
        <h3>复核摘要</h3>
        {detail.recognition_range_m && <p>识别范围 {detail.recognition_range_m.start === null ? "首道" : `${detail.recognition_range_m.start} 米`}至{detail.recognition_range_m.stop === null ? "末道" : `${detail.recognition_range_m.stop} 米`}（图件保留全道）</p>}
        <section aria-label="井下广覆盖">
            <p><strong>井下广覆盖复核分 {coverageScoreText(coverage)}</strong> · 500 米以后</p>
            {coverage && <p>覆盖 {coverageFractionText(coverage.coverage_fraction)} · 深部覆盖 {coverageFractionText(coverage.deep_coverage_fraction)} · 连续范围 {typeof coverage.continuous_span_m === "number" && Number.isFinite(coverage.continuous_span_m) ? `${coverage.continuous_span_m.toFixed(0)} 米` : "暂无"}</p>}
            {coverage && <p className="window-summary-note">{coveragePropagationText(coverage)} · {coverageCommonModeText(coverage)}</p>}
            <p className="window-summary-note">复核排序依据，不是地震概率。深部覆盖指 1000 米以后。</p>
        </section>
        <dl className="window-summary-metrics">{SUMMARY_FEATURES.map(field => {
            const value = detail.features[field.key];
            return <div key={field.key}><dt>{field.name}</dt><dd>{typeof value === "number" && Number.isFinite(value) ? `${(value * field.scale).toFixed(field.digits)}${field.unit}` : "暂无"}</dd><small>{field.hint}</small></div>;
        })}</dl>
        <p className="window-summary-note">辅助观察信号覆盖与数据质量，不代表地震概率。</p>
    </section>;
}
function message(cause: unknown) { return cause instanceof Error ? cause.message : String(cause); }
type ReviewIdentity = Pick<WindowItem, "id" | "analysis_id" | "input_sha256">;
function sameReviewInput(item: WindowItem, saved: ReviewIdentity) {
    return item.id === saved.id && item.analysis_id === saved.analysis_id && item.input_sha256 === saved.input_sha256;
}
function olderReview(review: WindowReview, current: WindowReview | null | undefined) {
    if (!current) return false;
    const createdAt = (value: WindowReview) => {
        const [seconds, fraction = ""] = (value.created_at ?? "").replace(/(?:Z|\+00:00)$/, "").split(".");
        return `${seconds}.${fraction.padEnd(6, "0")}`;
    };
    const time = createdAt(review), currentTime = createdAt(current);
    return time < currentTime || (time === currentTime && (review.id ?? "") < (current.id ?? ""));
}
function FavoriteButton({ detail, onToggle }: { detail: WindowDetail; onToggle: (id: string, favorite: boolean) => Promise<void> }) {
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const lock = useRef(false);
    const mounted = useRef(true);
    useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
    const seconds = detail.window_seconds ?? (Date.parse(detail.end_utc) - Date.parse(detail.start_utc)) / 1000;
    const canAdd = seconds === 30 && detail.status === "succeeded" && detail.artifacts.some(artifact => artifact.kind === "waterfall" || artifact.kind === "waterfall_filtered");
    async function toggle() {
        if (lock.current || (!detail.favorite && !canAdd)) return;
        lock.current = true;
        setBusy(true);
        setError("");
        try {
            await onToggle(detail.id, !detail.favorite);
        } catch (cause) {
            if (mounted.current) setError(message(cause));
        } finally {
            lock.current = false;
            if (mounted.current) setBusy(false);
        }
    }
    return <span className="window-favorite-control"><button aria-pressed={detail.favorite} disabled={busy || (!detail.favorite && !canAdd)} onClick={() => void toggle()}>{detail.favorite ? "★ 已收藏" : "☆ 收藏"}</button>{error && <span role="alert">{error}</span>}</span>;
}
function QueueReview({ review, attention }: { review: WindowReview | null; attention?: WindowItem["model_review_attention"] }) {
    if (!review)
        return null;
    const pendingAttention = !review.id || review.id === attention?.source_review_id || review.stale || review.labels.earthquake === "unknown";
    const state = review.stale ? "stale" : review.labels.earthquake === "unknown" ? "pending" : "saved";
    const label = review.stale ? "人工标签已失效 · 待重新复核" : review.labels.earthquake === "unknown" ? "人工：待确定" : `✓ 人工：${CLASS_NAMES[review.labels.earthquake]}`;
    return <>{attention && <span className={`window-model-attention ${pendingAttention ? "pending" : "reviewed"}`} title={attention.note}>模型{attention.kind === "FN" ? "漏检" : "误报"}{pendingAttention ? "待复核" : "已复核"}</span>}<span className={`queue-review window-queue-review window-queue-review-${state}`}>{label}</span></>;
}
function Evidence({ detail }: {
    detail: WindowDetail;
}) {
    const [offset, setOffset] = useState(0);
    const [page, setPage] = useState<WindowPage<Record<string, unknown>> | null>(null);
    const [error, setError] = useState("");
    const [revision, setRevision] = useState(0);
    useEffect(() => {
        if (!detail.analysis_id || detail.status !== "succeeded")
            return;
        const controller = new AbortController();
        setPage(null);
        setError("");
        void windowApi.evidence(detail.id, detail.analysis_id, offset, controller.signal).then(value => { if (!controller.signal.aborted)
            setPage(value); }).catch(cause => { if (!controller.signal.aborted)
            setError(message(cause)); });
        return () => controller.abort();
    }, [detail.id, detail.analysis_id, detail.status, offset, revision]);
    return <section>{error && <p role="alert">{error} <button onClick={() => setRevision(v => v + 1)}>重试证据</button></p>}{page && <><p>共 {page.total} 条证据；证据是整窗判断的辅助线索。</p><div className="window-table-scroll"><table><thead><tr><th>检测器</th><th>峰值时间</th><th>原始道索引区间</th><th>分数</th><th>通过筛选</th></tr></thead><tbody>{page.items.map((row, i) => <tr key={offset + i}><td>{String(row.detector ?? "—")}</td><td>{typeof row.peak_utc === "string" ? formatOccurredAt(row.peak_utc, detail.site_timezone) : "—"}</td><td>[{String(row.channel_start ?? "—")}, {String(row.channel_stop ?? "—")})</td><td>{typeof row.score === "number" ? Number(row.score.toPrecision(4)) : "—"}</td><td>{row.qualified === true ? "是" : "否"}</td></tr>)}</tbody></table></div><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>上一页证据</button><button disabled={offset + page.items.length >= page.total} onClick={() => setOffset(offset + 50)}>下一页证据</button></>}</section>;
}
function EvidenceDetails({ detail }: { detail: WindowDetail }) {
    const [loaded, setLoaded] = useState(false);
    return <details className="window-review-details" onToggle={event => { if (event.currentTarget.open) setLoaded(true); }}><summary>局部检测证据</summary>{loaded && <Evidence detail={detail}/>}</details>;
}
function Review({ detail, onConflict, onSaved }: {
    detail: WindowDetail;
    onConflict: () => void;
    onSaved: (review: WindowReview) => void;
}) {
    const [label, setLabel] = useState<WindowPresence>(() => detail.latest_review && !detail.latest_review.stale ? detail.latest_review.labels.earthquake : "unknown");
    const [note, setNote] = useState(detail.latest_review && !detail.latest_review.stale ? detail.latest_review.note ?? "" : "");
    const [busy, setBusy] = useState(false);
    const [notice, setNotice] = useState("");
    const [error, setError] = useState("");
    const lock = useRef(false);
    const pending = useRef<WindowReviewCreate | null>(null);
    const mounted = useRef(true);
    useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
    const seconds = detail.window_seconds ?? (Date.parse(detail.end_utc) - Date.parse(detail.start_utc)) / 1000;
    const eligible = detail.status === "succeeded" && Boolean(detail.analysis_id && detail.input_sha256) && [30, 60].includes(seconds) && detail.shape?.[1] === seconds * 100;
    async function save() {
        if (lock.current || !eligible)
            return;
        lock.current = true;
        setBusy(true);
        setError("");
        setNotice("");
        const body: WindowReviewCreate = { analysis_id: detail.analysis_id!, input_sha256: detail.input_sha256!, labels: { earthquake: label }, note: note.trim(), request_id: "" };
        const previous = pending.current;
        body.request_id = previous && JSON.stringify({ ...previous, request_id: "" }) === JSON.stringify(body) ? previous.request_id : crypto.randomUUID();
        pending.current = body;
        try {
            const review = await windowApi.review(detail.id, body);
            onSaved(review);
            if (mounted.current) {
                pending.current = null;
                setNotice("整窗标签已保存");
            }
        }
        catch (cause) {
            if (mounted.current) {
                if (cause instanceof ApiError && cause.status === 409) {
                    onConflict();
                }
                else
                    setError(message(cause));
            }
        }
        finally {
            lock.current = false;
            if (mounted.current)
                setBusy(false);
        }
    }
    return <section aria-label="整窗人工复核"><h3>整窗人工标签</h3><p>选择地震类或噪声类；待确定窗口不参与监督训练。</p>{detail.latest_review?.stale && <p>旧整窗标签已失效，请按当前输入重新复核。</p>}{!eligible && <p role="status">当前窗口未完成有效整窗分析，不能保存标签。</p>}
    <fieldset disabled={!eligible || busy}><legend>窗口类别</legend>{(["present", "absent"] as const).map(value => <label key={value}><input type="radio" name="window-class" value={value} checked={label === value} onChange={() => setLabel(value)}/>{CLASS_NAMES[value]}</label>)}</fieldset>
    <button disabled={!eligible || busy || label === "unknown"} onClick={() => setLabel("unknown")}>设为待确定</button><label className="window-note">复核备注<textarea maxLength={2000} value={note} disabled={!eligible || busy} onChange={e => setNote(e.target.value)}/></label><button disabled={!eligible || busy} onClick={() => void save()}>{busy ? "正在保存…" : "保存整窗标签"}</button>{notice && <p role="status">{notice}</p>}{error && <p role="alert">{error}</p>}
  </section>;
}
export default function WindowEventsPage() {
    const windowSeconds = 30;
    const [search] = useSearchParams();
    const [query, setQuery] = useState({ site: "", range: ["model_recommended", "favorites"].includes(search.get("range") ?? "") ? search.get("range")! : "candidates", scan: search.get("range") === "favorites" ? "" : search.get("scan_id") ?? "", review: search.get("range") === "favorites" ? "all" : "unreviewed", sort: "borehole_coverage", offset: 0 });
    const [sites, setSites] = useState<Site[]>([]);
    const [page, setPage] = useState<WindowPage<WindowItem> | null>(null);
    const queryRef = useRef(query);
    const pageRef = useRef(page);
    const favoriteRequestVersions = useRef(new Map<string, number>());
    const confirmedFavorites = useRef(new Map<string, { favorite: boolean; sequence: number }>());
    const favoriteConfirmationSequence = useRef(0);
    const confirmedReviews = useRef(new Map<string, { input: ReviewIdentity; review: WindowReview; sequence: number; spatial_coverage?: WindowItem["spatial_coverage"] }>());
    const reviewConfirmationSequence = useRef(0);
    queryRef.current = query;
    pageRef.current = page;
    const [selected, setSelected] = useState("");
    const [detail, setDetail] = useState<WindowDetail | null>(null);
    const detailRef = useRef(detail);
    detailRef.current = detail;
    const [revision, setRevision] = useState(0);
    const [listRevision, setListRevision] = useState(0);
    const [error, setError] = useState("");
    const [detailError, setDetailError] = useState("");
    const [notice, setNotice] = useState("");
    const [savedNotice, setSavedNotice] = useState("");
    useEffect(() => { const controller = new AbortController(); void api.get<Site[]>("/api/v1/sites", undefined, controller.signal).then(value => { if (!controller.signal.aborted)
        setSites(value); }).catch(cause => { if (!controller.signal.aborted)
        setError(message(cause)); }); return () => controller.abort(); }, []);
    useEffect(() => {
        const controller = new AbortController();
        const startedAt = favoriteConfirmationSequence.current;
        const reviewsStartedAt = reviewConfirmationSequence.current;
        setPage(null);
        setError("");
        void windowApi.list({ site_id: query.site, review_status: query.review, sort: query.sort, model_recommended: String(query.range === "model_recommended"), ...(query.range === "model_recommended" && query.scan ? { model_scan_id: query.scan } : {}), candidate_only: String(query.range === "candidates"), recommended: String(query.range === "recommended"), favorites_only: String(query.range === "favorites"), window_seconds: String(windowSeconds), limit: "30", offset: String(query.offset) }, controller.signal).then(value => { if (!controller.signal.aborted)
            {
                const items = value.items.map(item => {
                    const confirmed = confirmedFavorites.current.get(item.id);
                    const review = confirmedReviews.current.get(item.id);
                    const current = confirmed && confirmed.sequence > startedAt ? { ...item, favorite: confirmed.favorite } : item;
                    return review && review.sequence > reviewsStartedAt && sameReviewInput(item, review.input) && !olderReview(review.review, item.latest_review) ? { ...current, latest_review: review.review, ...(review.spatial_coverage !== undefined ? { spatial_coverage: review.spatial_coverage } : {}) } : current;
                });
                const visible = query.range === "favorites" ? items.filter(item => item.favorite) : items;
                const total = value.total - (items.length - visible.length);
                setPage({ ...value, items: visible, total });
                if (query.range === "favorites" && query.offset > 0 && total <= query.offset)
                    setQuery(previous => previous === query ? { ...previous, offset: Math.max(0, previous.offset - 30) } : previous);
            } }).catch(cause => { if (!controller.signal.aborted)
            setError(message(cause)); });
        return () => controller.abort();
    }, [query, listRevision, windowSeconds]);
    useEffect(() => {
        setDetail(null);
        setDetailError("");
        if (!selected)
            return;
        const controller = new AbortController();
        const startedAt = favoriteConfirmationSequence.current;
        const reviewsStartedAt = reviewConfirmationSequence.current;
        void windowApi.detail(selected, controller.signal, query.range === "model_recommended" ? query.scan : undefined).then(value => { if (!controller.signal.aborted)
            {
                const confirmed = confirmedFavorites.current.get(value.id);
                const review = confirmedReviews.current.get(value.id);
                const current = confirmed && confirmed.sequence > startedAt ? { ...value, favorite: confirmed.favorite } : value;
                setDetail(review && review.sequence > reviewsStartedAt && sameReviewInput(value, review.input) && !olderReview(review.review, value.latest_review) ? { ...current, latest_review: review.review, ...(review.spatial_coverage !== undefined ? { spatial_coverage: review.spatial_coverage } : {}) } : current);
            } }).catch(cause => { if (!controller.signal.aborted)
            setDetailError(message(cause)); });
        return () => controller.abort();
    }, [selected, revision, query.range, query.scan]);
    function changeQuery(next: typeof query) { setSelected(""); setNotice(""); setQuery(next); }
    function favoriteSaved(id: string, favorite: boolean) {
        const currentQuery = queryRef.current;
        const currentPage = pageRef.current;
        setDetail(previous => previous?.id === id ? { ...previous, favorite } : previous);
        setPage(previous => previous ? { ...previous, items: currentQuery.range === "favorites" && !favorite ? previous.items.filter(item => item.id !== id) : previous.items.map(item => item.id === id ? { ...item, favorite } : item), total: currentQuery.range === "favorites" && !favorite && previous.items.some(item => item.id === id) ? previous.total - 1 : previous.total } : previous);
        if (currentQuery.range === "favorites" && !favorite && currentPage?.items.some(item => item.id === id)) {
            if (currentQuery.offset > 0 && currentPage.total - 1 <= currentQuery.offset) setQuery(previous => previous.range === "favorites" && previous.offset === currentQuery.offset ? { ...previous, offset: Math.max(0, previous.offset - 30) } : previous);
            else setListRevision(value => value + 1);
        }
    }
    async function toggleFavorite(id: string, favorite: boolean) {
        const version = (favoriteRequestVersions.current.get(id) ?? 0) + 1;
        favoriteRequestVersions.current.set(id, version);
        const result = await windowApi.favorite(id, favorite);
        if (favoriteRequestVersions.current.get(id) === version) {
            confirmedFavorites.current.set(result.window_id, { favorite: result.favorite, sequence: ++favoriteConfirmationSequence.current });
            favoriteSaved(result.window_id, result.favorite);
        }
    }
    function reviewSaved(savedDetail: WindowDetail, review: WindowReview) {
        const input = { id: savedDetail.id, analysis_id: savedDetail.analysis_id, input_sha256: savedDetail.input_sha256 };
        const matches = (item: WindowItem) => sameReviewInput(item, input);
        const confirmed = confirmedReviews.current.get(savedDetail.id);
        const knownReviews = [
            confirmed?.review,
            pageRef.current?.items.find(matches)?.latest_review,
            detailRef.current && matches(detailRef.current) ? detailRef.current.latest_review : null,
        ];
        if (knownReviews.some(current => olderReview(review, current))) return;
        const pendingCoverage = savedDetail.spatial_coverage !== undefined ? { spatial_coverage: null } : {};
        const confirmation = { input, review, sequence: ++reviewConfirmationSequence.current, ...pendingCoverage };
        confirmedReviews.current.set(savedDetail.id, confirmation);
        setPage(previous => previous ? { ...previous, items: previous.items.map(item => matches(item) && !olderReview(review, item.latest_review) ? { ...item, latest_review: review, ...pendingCoverage } : item) } : previous);
        setDetail(previous => previous && matches(previous) && !olderReview(review, previous.latest_review) ? { ...previous, latest_review: review, ...pendingCoverage } : previous);
        setSavedNotice(`整窗标签已保存：${savedDetail.site_name} · ${formatOccurredAt(savedDetail.start_utc, savedDetail.site_timezone)}（窗口 ${savedDetail.id}）`);
        if (savedDetail.spatial_coverage !== undefined) {
            const controller = new AbortController();
            const currentQuery = queryRef.current;
            void windowApi.detail(savedDetail.id, controller.signal, currentQuery.range === "model_recommended" ? currentQuery.scan : undefined).then(value => {
                if (confirmedReviews.current.get(savedDetail.id) !== confirmation || !matches(value) || !value.latest_review || olderReview(value.latest_review, review)) return;
                const freshReview = value.latest_review;
                const currentReviews = [pageRef.current?.items.find(matches)?.latest_review, detailRef.current && matches(detailRef.current) ? detailRef.current.latest_review : null];
                if (currentReviews.some(current => olderReview(freshReview, current))) return;
                const refreshed = { latest_review: freshReview, spatial_coverage: value.spatial_coverage };
                confirmedReviews.current.set(savedDetail.id, { input, review: freshReview, sequence: ++reviewConfirmationSequence.current, spatial_coverage: value.spatial_coverage });
                setPage(previous => previous ? { ...previous, items: previous.items.map(item => matches(item) && !olderReview(freshReview, item.latest_review) ? { ...item, ...refreshed } : item) } : previous);
                setDetail(previous => previous && matches(previous) && !olderReview(freshReview, previous.latest_review) ? { ...previous, ...refreshed } : previous);
            }).catch(() => {
                if (confirmedReviews.current.get(savedDetail.id) === confirmation && detailRef.current && matches(detailRef.current)) setNotice("标签已保存，但复核分刷新失败；请刷新窗口重试。");
            });
        }
    }
    return <div className="window-events"><h1>{windowSeconds} 秒全通道窗口</h1><p>UTC 每分钟 00/30 秒对齐 · 100 Hz · 0.5–40 Hz</p><div className="window-toolbar">
    <label>窗口范围 <select value={query.range} onChange={e => changeQuery({ ...query, range: e.target.value, scan: "", review: e.target.value === "favorites" ? "all" : query.review, offset: 0 })}><option value="model_recommended">模型推荐</option><option value="candidates">疑似信号窗口</option><option value="recommended">推荐复核</option><option value="all">全部窗口</option><option value="favorites">只看收藏</option></select></label>
    <label>窗口站点 <select value={query.site} onChange={e => changeQuery({ ...query, site: e.target.value, offset: 0 })}><option value="">全部站点</option>{sites.map(site => <option key={site.id} value={site.id}>{site.name}</option>)}</select></label>
    <label>队列排序 <select value={query.sort} onChange={e => changeQuery({ ...query, sort: e.target.value, offset: 0 })}><option value="borehole_coverage">井下广覆盖优先</option><option value="default">原有排序</option></select></label>
    <label>复核状态 <select value={query.review} onChange={e => changeQuery({ ...query, review: e.target.value, offset: 0 })}><option value="unreviewed">待复核</option><option value="reviewed">已复核</option><option value="reviewed_earthquake">已复核·地震类</option><option value="reviewed_noise">已复核·噪声类</option><option value="all">全部状态</option></select></label><button onClick={() => { setListRevision(v => v + 1); setRevision(v => v + 1); }}>刷新窗口</button>
  </div>{error && <p role="alert">{error}</p>}{notice && <p role="alert">{notice}</p>}{savedNotice && <p role="status">{savedNotice}</p>}<div className="window-layout"><aside aria-label="整窗队列"><p>{query.sort === "borehole_coverage" ? "按井下广覆盖复核分从高到低排序" : query.range === "model_recommended" ? "按模型得分从高到低排序" : "按目录批次优先级、筛选分数排序"}</p>{!page && !error && <p>正在加载窗口…</p>}{page && <><p>共 {page.total} 个窗口</p>{page.items.map(item => <button className="window-queue-item" aria-pressed={selected === item.id} key={item.id} onClick={() => { setSelected(item.id); setNotice(""); }}><strong>{item.site_name} · {formatOccurredAt(item.start_utc, item.site_timezone)}</strong><span>井下广覆盖 {coverageScoreText(item.spatial_coverage)}</span><span>{windowStatusNames[item.status]} · {query.range === "model_recommended" ? `模型得分 ${item.model_recommendation?.score.toFixed(4) ?? "—"}` : `优先级 ${item.priority} · 分数 ${item.score == null ? "—" : Number(item.score.toPrecision(4))}`}</span><span>目录关联 {item.catalog_matches.length}</span>{item.favorite && <span className="window-queue-favorite">★ 已收藏</span>}<QueueReview review={item.latest_review} attention={item.model_review_attention}/></button>)}{page.items.length === 0 && <p>没有符合条件的窗口</p>}<div className="window-toolbar"><button aria-label="上一页窗口" disabled={query.offset === 0} onClick={() => changeQuery({ ...query, offset: Math.max(0, query.offset - 30) })}>上一页</button><button aria-label="下一页窗口" disabled={query.offset + page.items.length >= page.total} onClick={() => changeQuery({ ...query, offset: query.offset + 30 })}>下一页</button></div></>}</aside>
    <article>{detailError && <p role="alert">{detailError} <button onClick={() => setRevision(v => v + 1)}>重试详情</button></p>}{!selected && <p>选择一个窗口开始复核</p>}{selected && (!detail || detail.id !== selected) && !detailError && <p>正在加载窗口详情…</p>}{detail && detail.id === selected && <div key={`${detail.id}:${detail.analysis_id}:${detail.input_sha256}:${revision}`}><div className="window-detail-title"><h2>{detail.site_name} · {formatOccurredAt(detail.start_utc, detail.site_timezone)}</h2><FavoriteButton detail={detail} onToggle={toggleFavorite}/></div><p>站点时区：{detail.site_timezone} · {detail.shape ? `${detail.shape[0]} 道 × ${detail.shape[1]} 样本` : "样本待生成"} · 单位 {detail.unit ?? "待定"}</p><p>传统规则证据：{detail.candidate === true ? "通过疑似信号筛选" : detail.candidate === false ? "未触发合格信号" : "筛选状态待补齐"}；类别由人工按整窗判断。</p>
      {detail.model_recommendation && <section aria-label="扫描模型判定"><h3>模型推荐</h3><p>{detail.model_recommendation.model_name} · 版本 {detail.model_recommendation.model_id.slice(0, 8)}</p><p>模型得分 {detail.model_recommendation.score.toFixed(4)} · 入选阈值 {detail.model_recommendation.threshold.toFixed(4)}</p><p>扫描时固定的模型判定；得分不是地震概率，最终类别由人工复核。</p></section>}
      {!detail.model_recommendation && detail.status === "succeeded" && detail.analysis_id && detail.input_sha256 && <WindowPrediction windowId={detail.id} analysisId={detail.analysis_id} inputSha256={detail.input_sha256} />}
      {detail.model_review_attention && <section aria-label="模型误判复核说明"><h3>模型误判复核说明</h3><p style={{ whiteSpace: "pre-wrap" }}>{detail.model_review_attention.note}</p></section>}<WindowSignalViewer detail={detail}/><Review detail={detail} onConflict={() => { setNotice("输入已变更或请求冲突，已重新加载；请重新核对整窗标签。"); setRevision(v => v + 1); }} onSaved={review => reviewSaved(detail, review)}/><ReviewSummary detail={detail}/>
      <EvidenceDetails detail={detail}/>
    </div>}</article></div></div>;
}
