import { useEffect, useState } from "react";
import { api, type Site } from "../api";
import { scanCoverageApi, type ScanCoverage, type ScanCoverageDay } from "../scanCoverageApi";
import "./scan-coverage-calendar.css";

function duration(seconds: number) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor(seconds % 3600 / 60);
    const remainder = Math.floor(seconds % 60);
    return [hours ? `${hours} 小时` : "", minutes ? `${minutes} 分钟` : "", remainder ? `${remainder} 秒` : ""].filter(Boolean).join(" ") || "0 秒";
}

function level(seconds: number) {
    return seconds === 0 ? 0 : seconds <= 21600 ? 1 : seconds <= 43200 ? 2 : seconds <= 64800 ? 3 : 4;
}

function beijingTime(value: string) {
    const parts = new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" }).formatToParts(new Date(value));
    const part = (type: Intl.DateTimeFormatPartTypes) => parts.find(item => item.type === type)?.value;
    return `${part("year")}-${part("month")}-${part("day")} ${part("hour")}:${part("minute")}:${part("second")}`;
}

export default function ScanCoverageCalendar({ revision = 0 }: { revision?: number }) {
    const [sites, setSites] = useState<Site[]>([]);
    const [siteId, setSiteId] = useState("");
    const [year, setYear] = useState<number>();
    const [resolvedYear, setResolvedYear] = useState<number>();
    const [availableYears, setAvailableYears] = useState<number[]>([]);
    const [coverage, setCoverage] = useState<ScanCoverage | null>(null);
    const [error, setError] = useState("");
    const [siteError, setSiteError] = useState("");
    const [siteRetry, setSiteRetry] = useState(0);
    const [refresh, setRefresh] = useState(0);
    const [busy, setBusy] = useState(true);
    const [selectedDate, setSelectedDate] = useState("");
    const [detail, setDetail] = useState<ScanCoverageDay | null>(null);
    const [detailError, setDetailError] = useState("");
    const [detailRetry, setDetailRetry] = useState(0);
    useEffect(() => {
        const controller = new AbortController();
        setSiteError("");
        void api.get<Site[]>("/api/v1/sites", undefined, controller.signal).then(value => {
            if (!controller.signal.aborted) setSites(value);
        }).catch(cause => { if (!controller.signal.aborted) setSiteError(String(cause)); });
        return () => controller.abort();
    }, [siteRetry]);
    useEffect(() => {
        let stopped = false;
        let inFlight = false;
        let controller: AbortController | undefined;
        let timer: number | undefined;
        const schedule = () => {
            window.clearTimeout(timer);
            if (document.visibilityState === "visible") timer = window.setTimeout(() => void load(), 60000);
        };
        async function load() {
            if (stopped || inFlight) return;
            window.clearTimeout(timer);
            inFlight = true;
            controller = new AbortController();
            const signal = controller.signal;
            setBusy(true); setError("");
            try {
                const value = await scanCoverageApi.year(year, siteId, signal);
                if (!stopped && !signal.aborted) {
                    setCoverage(value); setResolvedYear(value.year); setAvailableYears(value.available_years);
                    setSelectedDate(date => date && !date.startsWith(`${value.year}-`) ? "" : date);
                }
            } catch (cause) {
                if (!stopped && !signal.aborted) setError(String(cause));
            } finally {
                inFlight = false;
                if (!stopped) { setBusy(false); schedule(); }
            }
        }
        const onVisibility = () => {
            window.clearTimeout(timer);
            if (document.visibilityState === "visible") void load();
        };
        document.addEventListener("visibilitychange", onVisibility);
        void load();
        return () => { stopped = true; controller?.abort(); window.clearTimeout(timer); document.removeEventListener("visibilitychange", onVisibility); };
    }, [year, siteId, revision, refresh]);
    useEffect(() => {
        const controller = new AbortController();
        setDetail(null); setDetailError("");
        if (selectedDate) void scanCoverageApi.day(selectedDate, siteId, controller.signal).then(value => {
            if (!controller.signal.aborted) setDetail(value);
        }).catch(cause => { if (!controller.signal.aborted) setDetailError(String(cause)); });
        return () => controller.abort();
    }, [selectedDate, siteId, year, coverage, detailRetry]);
    const displayedYear = year ?? resolvedYear;
    const years = [...new Set([...availableYears, ...(displayedYear === undefined ? [] : [displayedYear])])].sort((a, b) => b - a);
    function changeFilter(nextYear: number | undefined, nextSite: string) {
        setCoverage(null); setSelectedDate(""); setDetail(null); setDetailError(""); setError("");
        setYear(nextYear); setSiteId(nextSite);
    }
    const days: { date: string; seconds: number; row: number; column: number }[] = [];
    let offset = 0;
    let weeks = 53;
    if (coverage) {
        const start = Date.UTC(coverage.year, 0, 1);
        offset = new Date(start).getUTCDay();
        const byDate = new Map(coverage.days.map(day => [day.date, day.scanned_seconds]));
        for (let time = start, index = 0; time < Date.UTC(coverage.year + 1, 0, 1); time += 86400000, index++) {
            const date = new Date(time).toISOString().slice(0, 10);
            days.push({ date, seconds: byDate.get(date) ?? 0, row: (index + offset) % 7 + 2, column: Math.floor((index + offset) / 7) + 2 });
        }
        weeks = Math.ceil((days.length + offset) / 7);
    }
    return <section className="scan-coverage" aria-label="已扫描数据日历">
        <div className="scan-coverage-heading"><h2>已扫描数据日历</h2><div className="scan-coverage-filters">
            <label>日历年份 <select value={displayedYear ?? ""} onChange={event => changeFilter(Number(event.target.value), siteId)}>{years.map(value => <option key={value} value={value}>{value}</option>)}</select></label>
            <label>日历站点 <select value={siteId} onChange={event => changeFilter(displayedYear, event.target.value)}><option value="">全部站点</option>{sites.map(site => <option key={site.id} value={site.id}>{site.name}</option>)}</select></label>
            <button className="subtle" type="button" disabled={busy} onClick={() => setRefresh(value => value + 1)}>刷新日历</button>
        </div></div>
        <p className="scan-coverage-help">按北京时间的数据采集日期统计，每天以去重扫描时长着色（0–24 小时），含模型判为地震、噪声和规则跳过的窗口。</p>
        <p className="scan-coverage-help">全部站点按时间并集去重，计入不同模型的历史成功扫描。灰色表示暂无成功扫描记录，不代表没有源数据。</p>
        {error && <p role="alert" className="error">{error}</p>}
        {siteError && <p role="alert" className="error">{siteError} <button className="subtle" onClick={() => setSiteRetry(value => value + 1)}>重试站点列表</button></p>}
        {!coverage && !error && <p role="status">正在加载已扫描数据日历…</p>}
        {coverage && <>
            <p className="scan-coverage-summary">{coverage.year} 年 · 已扫描 {coverage.scanned_days} 天 · 去重扫描 {duration(coverage.total_scanned_seconds)}</p>
            <div className="scan-coverage-scroll"><div className="scan-coverage-grid" style={{ gridTemplateColumns: `1.5rem repeat(${weeks}, 12px)` }} aria-label={`${coverage.year} 年扫描覆盖`}>
                {Array.from({ length: 12 }, (_, month) => <span className="scan-coverage-month" key={month} style={{ gridRow: 1, gridColumn: `${Math.floor(((Date.UTC(coverage.year, month, 1) - Date.UTC(coverage.year, 0, 1)) / 86400000 + offset) / 7) + 2} / span 3` }}>{month + 1}月</span>)}
                {["日", "一", "二", "三", "四", "五", "六"].map((day, index) => <span className="scan-coverage-weekday" key={day} aria-hidden="true" style={{ gridRow: index + 2, gridColumn: 1 }}>{day}</span>)}
                {days.map(day => {
                    const label = `${day.date}（北京时间）：去重扫描 ${duration(day.seconds)}${day.seconds === 0 ? "，暂无成功扫描记录" : ""}`;
                    return <button key={day.date} type="button" data-date={day.date} className={`scan-coverage-day scan-coverage-level-${level(day.seconds)}`} style={{ gridRow: day.row, gridColumn: day.column }} title={label} aria-label={label} aria-pressed={selectedDate === day.date} onClick={() => setSelectedDate(day.date)} />;
                })}
            </div></div>
            <div className="scan-coverage-legend" aria-label="每天扫描时长图例"><span>少</span>{["暂无成功扫描记录", "大于 0 至 6 小时", "大于 6 至 12 小时", "大于 12 至 18 小时", "大于 18 至 24 小时"].map((label, index) => <span key={label} className={`scan-coverage-swatch scan-coverage-level-${index}`} title={label} aria-label={label} />)}<span>多</span></div>
        </>}
        {selectedDate && <section className="scan-coverage-detail" aria-label={`${selectedDate} 扫描覆盖详情`}>
            <h3>{selectedDate} · 北京时间</h3>
            {detailError && <p role="alert" className="error">{detailError} <button className="subtle" onClick={() => setDetailRetry(value => value + 1)}>重试当天详情</button></p>}
            {!detail && !detailError && <p role="status">正在加载当天扫描区间…</p>}
            {detail && <><p>去重扫描 {duration(detail.scanned_seconds)}</p>
                {detail.intervals.length === 0 ? <p>当天暂无已扫描记录。</p> : <><p className="scan-coverage-help">以下为已合并的扫描时间区间，右端不含。</p><ul>{detail.intervals.map(interval => <li key={interval.start_at}><time dateTime={interval.start_at}>{beijingTime(interval.start_at)}</time> — <time dateTime={interval.end_at}>{beijingTime(interval.end_at)}</time></li>)}</ul></>}
            </>}
        </section>}
    </section>;
}
