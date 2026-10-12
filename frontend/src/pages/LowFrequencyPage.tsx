import { useCallback, useEffect, useLayoutEffect, useRef, useState, type FormEvent } from "react";
import { api, ApiError, type Site } from "../api";
import { formatOccurredAt } from "../components/EventQueue";
import {
  lowFrequencyApi, type LowFrequencyCandidate, type LowFrequencyPlot,
  type LowFrequencyResult, type LowFrequencyReview, type LowFrequencyScanDetail,
  type LowFrequencyScanSummary, type MachineLabel,
} from "../lowFrequencyApi";

const STATUS = { queued: "排队中", running: "运行中", succeeded: "已完成", failed: "失败" };
const LABELS: Record<MachineLabel, string> = { earthquake: "地震", blast: "爆破", traffic: "车辆", noise: "噪声", unknown: "待定" };
const PLOTS: { key: LowFrequencyPlot; name: string; alt: string }[] = [
  { key: "waterfall", name: "瀑布图", alt: "低频瀑布图" },
  { key: "waveform", name: "波形", alt: "低频波形图" },
  { key: "psd", name: "PSD", alt: "低频功率谱" },
  { key: "stft", name: "STFT", alt: "低频时频图" },
];
const MEASUREMENTS: Record<string, string> = {
  band_rms: "频带均方根幅值",
  snr_db: "信噪比（dB）", active_channel_fraction: "活跃通道占比", coherence: "相干性",
  dominant_frequency_hz: "主频（Hz）", frequency_resolution_hz: "频率采样间隔（Hz）",
  energy_duration_s: "能量持续时间（s）", sample_rate_hz: "采样率（Hz）",
  band_fraction_0_01_0_05: "0.01–0.05 Hz 能量占比", band_fraction_0_05_0_1: "0.05–0.1 Hz 能量占比",
  band_fraction_0_1_0_4: "0.1–0.4 Hz 能量占比", valid_channel_count: "有效通道数",
  valid_channel_fraction: "有效通道占比", duration_seconds: "有效窗口时长（s）", background_seconds: "背景时长（s）",
  background_start_seconds: "背景起点（相对秒）", background_end_seconds: "背景终点（相对秒）",
};
const WARNINGS: Record<string, string> = {
  insufficient_filter_context: "滤波所需的真实前后数据不足。",
  insufficient_background: "真实背景数据不足，信噪比仅供参考。",
  insufficient_low_frequency_resolution: "低频分辨率不足，请结合实际窗口时长复核。",
  candidate_insufficient_low_frequency_resolution: "候选时长不足，无法可靠给出低频主频。",
  no_valid_channels: "没有可用的有效通道。",
  insufficient_spatial_support_channels: "支持候选的空间通道不足，请结合波形复核。",
  candidate_clipped_start: "候选在窗口起点截断。",
  candidate_clipped_end: "候选在窗口末端截断。",
  zero_background_energy: "背景能量为零，无法给出可靠的信噪比。",
};
const running = (scan: LowFrequencyScanSummary) => scan.status === "queued" || scan.status === "running";
const localTime = (value: string) => {
  const milliseconds = new Date(value).getUTCMilliseconds();
  return formatOccurredAt(value, "Asia/Shanghai") + (milliseconds ? `.${String(milliseconds).padStart(3, "0")}` : "");
};
const message = (error: unknown, fallback: string) => error instanceof ApiError ? error.message : fallback;
const metric = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? Number(value.toPrecision(5)).toString() : "—";

export default function LowFrequencyPage() {
  const [sites, setSites] = useState<Site[]>([]);
  const [siteId, setSiteId] = useState("");
  const [directory, setDirectory] = useState("jjk2025_2026/20260616");
  const [start, setStart] = useState("2026-06-16T17:05");
  const [stop, setStop] = useState("2026-06-16T17:15");
  const [distanceStart, setDistanceStart] = useState("499");
  const [distanceStop, setDistanceStop] = useState("");
  const [scans, setScans] = useState<LowFrequencyScanSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");
  const [formError, setFormError] = useState("");
  const [siteError, setSiteError] = useState("");
  const [busy, setBusy] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<LowFrequencyScanDetail | null>(null);
  const [detailError, setDetailError] = useState("");
  const [detailVersion, setDetailVersion] = useState(0);
  const mounted = useRef(true);
  const busyRef = useRef(false);
  const scansRef = useRef(scans);
  const historyController = useRef<AbortController | null>(null);
  const createController = useRef<AbortController | null>(null);
  scansRef.current = scans;

  const refreshHistory = useCallback(async () => {
    historyController.current?.abort();
    const controller = new AbortController();
    historyController.current = controller;
    try {
      const value = await lowFrequencyApi.list(controller.signal);
      if (!controller.signal.aborted && mounted.current) { setScans(value); setHistoryError(""); }
    } catch (error) {
      if (!controller.signal.aborted && mounted.current) setHistoryError(message(error, "无法加载低频扫描历史"));
    } finally {
      if (!controller.signal.aborted && mounted.current) setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    const controller = new AbortController();
    api.get<Site[]>("/api/v1/sites", undefined, controller.signal).then((value) => {
      if (controller.signal.aborted) return;
      setSites(value);
      setSiteId(value.find((site) => site.enabled)?.id ?? "");
    }).catch((error) => {
      if (!controller.signal.aborted) setSiteError(message(error, "无法加载站点"));
    });
    void refreshHistory();
    return () => {
      mounted.current = false;
      controller.abort();
      historyController.current?.abort();
      createController.current?.abort();
    };
  }, [refreshHistory]);

  const hasRunning = scans.some(running);
  useEffect(() => {
    if (!hasRunning) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      await refreshHistory();
      if (!cancelled) timer = setTimeout(() => void poll(), 3000);
    };
    timer = setTimeout(() => void poll(), 3000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [hasRunning, refreshHistory]);

  useEffect(() => {
    if (!selectedId) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let poll = scansRef.current.some((scan) => scan.id === selectedId && running(scan));
    setDetail(null);
    setDetailError("");
    const load = async () => {
      try {
        const value = await lowFrequencyApi.detail(selectedId, controller.signal);
        if (controller.signal.aborted) return;
        setDetail(value);
        setDetailError("");
        setScans((current) => current.map((scan) => scan.id === value.id ? value : scan));
        poll = running(value);
      } catch (error) {
        if (!controller.signal.aborted) setDetailError(message(error, "无法加载低频扫描"));
      } finally {
        if (!controller.signal.aborted && poll) timer = setTimeout(() => void load(), 3000);
      }
    };
    void load();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [selectedId, detailVersion]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (busyRef.current) return;
    const startDate = new Date(`${start}+08:00`);
    const stopDate = new Date(`${stop}+08:00`);
    const seconds = (stopDate.getTime() - startDate.getTime()) / 1000;
    const from = distanceStart.trim() === "" ? NaN : Number(distanceStart);
    const to = distanceStop.trim() === "" ? null : Number(distanceStop);
    if (!siteId || !directory.trim()) { setFormError("请选择站点并填写相对目录"); return; }
    if (!Number.isFinite(seconds) || seconds < 600 || seconds > 1200) { setFormError("连续有效窗必须为 600–1200 秒"); return; }
    if (!Number.isFinite(from) || from < 0 || (to !== null && (!Number.isFinite(to) || to < 0))) { setFormError("请输入有限且非负的距离"); return; }
    if (to !== null && to <= from) { setFormError("终点距离必须大于起点距离"); return; }
    busyRef.current = true;
    setBusy(true);
    setFormError("");
    const controller = new AbortController();
    createController.current = controller;
    try {
      const value = await lowFrequencyApi.create({
        site_id: siteId, relative_path: directory.trim(), start_utc: startDate.toISOString(), end_utc: stopDate.toISOString(),
        distance_start_m: from, distance_stop_m: to,
      }, controller.signal);
      if (controller.signal.aborted || !mounted.current) return;
      historyController.current?.abort();
      setHistoryLoading(false);
      setScans((current) => [value, ...current.filter((scan) => scan.id !== value.id)]);
      setSelectedId(value.id);
    } catch (error) {
      if (!controller.signal.aborted && mounted.current) setFormError(message(error, "无法开始低频扫描"));
    } finally {
      if (!controller.signal.aborted && mounted.current) { busyRef.current = false; setBusy(false); }
    }
  }

  const currentDetail = detail?.id === selectedId ? detail : null;
  return (
    <section className="lf-page">
      <h1>低频连续扫描</h1>
      <p>固定频带 0.01–0.4 Hz；连续有效窗 600–1200 秒，两侧各约 5 分钟连续数据用于滤波。</p>
      <p className="lf-help">光缆 0–299 m 悬空，299–499 m 地表，499 m 后为井中；以源文件实际距离为准，光缆距离不等于井深。范围可随时修改，只用于新扫描。</p>
      <form className="lf-form" aria-label="低频扫描设置" onSubmit={(event) => void submit(event)} noValidate>
        <label>站点<select value={siteId} disabled={busy} onChange={(event) => setSiteId(event.target.value)}>
          <option value="">请选择站点</option>
          {sites.map((site) => <option key={site.id} value={site.id} disabled={!site.enabled}>{site.name}{site.enabled ? "" : "（已停用）"}</option>)}
        </select></label>
        <label>相对目录<input value={directory} disabled={busy} onChange={(event) => setDirectory(event.target.value)} /></label>
        <label>开始时间（UTC+8）<input type="datetime-local" step={1} value={start} disabled={busy} onChange={(event) => setStart(event.target.value)} /></label>
        <label>结束时间（UTC+8）<input type="datetime-local" step={1} value={stop} disabled={busy} onChange={(event) => setStop(event.target.value)} /></label>
        <label>起点距离（m）<input inputMode="decimal" value={distanceStart} disabled={busy} onChange={(event) => setDistanceStart(event.target.value)} /></label>
        <label>终点距离（m，留空为文件末端）<input inputMode="decimal" value={distanceStop} disabled={busy} onChange={(event) => setDistanceStop(event.target.value)} /></label>
        <div className="lf-form-actions"><button className="primary" type="submit" disabled={busy || !siteId}>{busy ? "正在提交…" : "开始低频扫描"}</button></div>
        {(siteError || formError) && <p className="lf-error" role="alert">{siteError || formError}</p>}
      </form>
      <div className="lf-workspace">
        <aside className="lf-history" aria-label="低频扫描历史">
          <div className="lf-section-heading"><h2>扫描历史</h2><button type="button" className="subtle" onClick={() => void refreshHistory()}>刷新历史</button></div>
          {historyLoading && <p role="status">正在加载扫描历史…</p>}
          {historyError && <p className="lf-error" role="alert">{historyError}</p>}
          {!historyLoading && !scans.length && <p>暂无低频扫描</p>}
          <ul className="lf-list">{scans.map((scan) => <li key={scan.id}>
            <button type="button" aria-label={`扫描 ${scan.id}`} aria-pressed={scan.id === selectedId} className="lf-select" onClick={() => setSelectedId(scan.id)}>
              <span>{localTime(scan.start_utc)}</span><strong>{STATUS[scan.status]}</strong>
              <span>{sites.find((site) => site.id === scan.site_id)?.name ?? scan.site_id} · {scan.distance_start_m}–{scan.distance_stop_m ?? "末端"} m</span>
              <span className="lf-path">{scan.relative_path}</span>
            </button>
          </li>)}</ul>
        </aside>
        <section className="lf-detail" aria-label="低频扫描结果">
          {!selectedId && <p>选择历史扫描查看结果，或开始新的低频扫描。</p>}
          {selectedId && !currentDetail && !detailError && <p role="status">正在加载扫描结果…</p>}
          {detailError && <div><p className="lf-error" role="alert">{detailError}</p><button type="button" onClick={() => setDetailVersion((value) => value + 1)}>重试加载扫描</button></div>}
          {currentDetail && <>
            <h2>{STATUS[currentDetail.status]} · {localTime(currentDetail.start_utc)}</h2>
            <p>有效时间：{localTime(currentDetail.start_utc)} – {localTime(currentDetail.end_utc)}（UTC+8）</p>
            {currentDetail.error_message && <p className="lf-error" role="alert">{currentDetail.error_message}</p>}
            {running(currentDetail) && <p role="status">扫描进行中，每 3 秒更新进度。</p>}
            {currentDetail.result && <ScanResult key={currentDetail.id} scanId={currentDetail.id} result={currentDetail.result} onReviewed={(candidateId, review) => setDetail((current) =>
              current?.id === currentDetail.id && current.result ? { ...current, result: { ...current.result, candidates: current.result.candidates.map((candidate) => candidate.id === candidateId ? { ...candidate, latest_review: review } : candidate) } } : current,
            )} />}
          </>}
        </section>
      </div>
    </section>
  );
}

function ScanResult({ scanId, result, onReviewed }: { scanId: string; result: LowFrequencyResult; onReviewed: (candidateId: string, review: LowFrequencyReview) => void }) {
  const [plot, setPlot] = useState<LowFrequencyPlot>("waterfall");
  const [selected, setSelected] = useState<string | null>(null);
  const [imageError, setImageError] = useState(false);
  const candidate = result.candidates.find((item) => item.id === selected);
  const currentPlot = PLOTS.find((item) => item.key === plot)!;
  const inputUnit = result.measurements.input_unit;
  return <>
    <p>实际距离 {result.distance_start_m}–{result.distance_stop_m} m · {result.channel_count} 通道 · {result.source_count} 源文件 · 采样率 {result.sample_rate_hz} Hz</p>
    <p className="lf-help">背景固定为有效窗起点前 200–30 秒；前置或持续波包可能抬高阈值，未检出候选也不能排除信号。</p>
    {result.warnings.length > 0 && <details className="lf-diagnostics">
      <summary>诊断记录（{result.warnings.length} 条{result.warnings.some((warning) => /时钟|接缝/.test(warning)) ? "，含时钟接缝" : ""}）</summary>
      <ul className="lf-warnings">{result.warnings.map((warning, index) => <li key={index}>{WARNINGS[warning] ?? warning}</li>)}</ul>
    </details>}
    <div className="lf-tabs" role="tablist" aria-label="低频图件">{PLOTS.map((item, index) => <button
      key={item.key} type="button" role="tab" id={`lf-tab-${item.key}`} aria-controls="lf-plot-panel" aria-selected={plot === item.key} tabIndex={plot === item.key ? 0 : -1}
      onClick={() => { setPlot(item.key); setImageError(false); }}
      onKeyDown={(event) => {
        const next = event.key === "ArrowRight" ? (index + 1) % PLOTS.length : event.key === "ArrowLeft" ? (index + PLOTS.length - 1) % PLOTS.length : event.key === "Home" ? 0 : event.key === "End" ? PLOTS.length - 1 : null;
        if (next === null) return;
        event.preventDefault(); setPlot(PLOTS[next].key); setImageError(false);
        document.getElementById(`lf-tab-${PLOTS[next].key}`)?.focus();
      }}
    >{item.name}</button>)}</div>
    <div id="lf-plot-panel" role="tabpanel" aria-labelledby={`lf-tab-${plot}`} className="lf-plot">
      {imageError ? <p className="lf-error" role="alert">图件加载失败，请刷新扫描结果。</p> : <img key={result.plots[plot]} src={result.plots[plot]} alt={currentPlot.alt} onError={() => setImageError(true)} />}
    </div>
    {plot !== "psd" && <p className="lf-help">虚线（如有）标记存在时间偏差的文件接缝；微小时间对齐也可能影响低频信号。</p>}
    {plot === "stft" && <p className="lf-help">STFT 使用 200 秒时间窗，颜色变化不能作为精确到时。</p>}
    {Object.keys(result.measurements).length > 0 && <details className="lf-measurements"><summary>窗口测量</summary>
      <p>幅值单位：{typeof inputUnit === "string" && inputUnit.trim() && inputUnit !== "unknown" ? inputUnit : "未知"}</p>
      <MeasurementTable values={result.measurements} />
    </details>}
    <h3>候选信号 · 按评分从高到低</h3>
    <p className="lf-help">评分为检测证据强度，候选仍需人工判断。</p>
    {!result.candidates.length && <p>此窗口未检出低频候选。</p>}
    <div className="lf-candidates">
      <ul className="lf-list">{[...result.candidates].sort((left, right) => right.score - left.score).map((item) => <li key={item.id}>
        <button type="button" className="lf-select" aria-label={`候选 ${item.id}`} aria-pressed={selected === item.id} onClick={() => setSelected(item.id)}>
          <span>{localTime(item.peak_utc)}</span><strong>评分 {item.score.toFixed(3)}</strong>
          <span>{item.latest_review ? LABELS[item.latest_review.label] : "待复核"}</span>
        </button>
      </li>)}</ul>
      {candidate && <section aria-label="低频候选复核">
        <p>候选时间：{localTime(candidate.start_utc)} – {localTime(candidate.end_utc)}</p>
        <MeasurementTable values={{ snr_db: candidate.snr_db, active_channel_fraction: candidate.active_channel_fraction, coherence: candidate.coherence, dominant_frequency_hz: candidate.dominant_frequency_hz, frequency_resolution_hz: candidate.frequency_resolution_hz, energy_duration_s: candidate.energy_duration_s }} />
        <CandidateReview key={candidate.id} scanId={scanId} candidate={candidate} onSaved={(review) => onReviewed(candidate.id, review)} />
      </section>}
    </div>
  </>;
}

function MeasurementTable({ values }: { values: Record<string, unknown> }) {
  return <table className="lf-metrics"><tbody>{Object.entries(MEASUREMENTS).filter(([key]) => key in values).map(([key, name]) => <tr key={key}><th scope="row">{name}</th><td>{metric(values[key])}</td></tr>)}</tbody></table>;
}

function CandidateReview({ scanId, candidate, onSaved }: { scanId: string; candidate: LowFrequencyCandidate; onSaved: (review: LowFrequencyReview) => void }) {
  const [label, setLabel] = useState<MachineLabel | "">(candidate.latest_review?.label ?? "");
  const [notes, setNotes] = useState(candidate.latest_review?.notes ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const active = useRef(true);
  const busyRef = useRef(false);
  const requestId = useRef<string | null>(null);
  useLayoutEffect(() => { active.current = true; return () => { active.current = false; }; }, []);
  const edited = () => { requestId.current = null; setSaved(false); setError(""); };
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!label || busyRef.current) return;
    busyRef.current = true; setBusy(true); setError(""); setSaved(false);
    requestId.current ??= crypto.randomUUID();
    try {
      const review = await lowFrequencyApi.review(scanId, candidate.id, { label, notes, request_id: requestId.current });
      if (!active.current) return;
      setSaved(true); onSaved(review);
    } catch (cause) {
      if (active.current) setError(message(cause, "无法保存低频复核"));
    } finally {
      if (active.current) { busyRef.current = false; setBusy(false); }
    }
  }
  return <form className="lf-review" onSubmit={(event) => void submit(event)}>
    <label>低频人工标签<select value={label} disabled={busy} onChange={(event) => { setLabel(event.target.value as MachineLabel | ""); edited(); }}>
      <option value="">请选择人工标签</option>{Object.entries(LABELS).map(([value, text]) => <option key={value} value={value}>{text}</option>)}
    </select></label>
    <label>低频复核备注<textarea rows={3} value={notes} disabled={busy} onChange={(event) => { setNotes(event.target.value); edited(); }} /></label>
    <button className="primary" type="submit" disabled={!label || busy}>{busy ? "正在保存…" : "保存低频复核"}</button>
    {error && <p className="lf-error" role="alert">{error}</p>}
    {saved && <p role="status">复核已保存</p>}
  </form>;
}
