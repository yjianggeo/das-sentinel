import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "../api";
import ScanCoverageCalendar from "./ScanCoverageCalendar";
vi.mock("../api", () => ({ api: { get: vi.fn() } }));

function coverage(year: number, seconds: Record<string, number> = {}) {
    const days = [];
    for (let time = Date.UTC(year, 0, 1); time < Date.UTC(year + 1, 0, 1); time += 86400000) {
        const date = new Date(time).toISOString().slice(0, 10);
        days.push({ date, scanned_seconds: seconds[date] ?? 0 });
    }
    return { year, timezone: "Asia/Shanghai", available_years: [2023, 2024], days,
        total_scanned_seconds: Object.values(seconds).reduce((sum, value) => sum + value, 0),
        scanned_days: Object.values(seconds).filter(value => value > 0).length };
}
const sites = [{ id: "alpha", name: "甲站", data_path: "/alpha", timezone: "Asia/Shanghai",
    cable_type: "downhole", enabled: true, created_at: "2024-01-01T00:00:00Z",
    current_profile: { id: "profile", version: 1, payload: {}, is_current: true, created_at: "2024-01-01T00:00:00Z" } }];
function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>(done => { resolve = done; });
    return { promise, resolve };
}
beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
    vi.mocked(api.get).mockImplementation(async (path, params) => path === "/api/v1/sites" ? sites : coverage(Number(params?.year ?? 2024)));
});
afterEach(() => { vi.useRealTimers(); });

it("defaults to the latest year with records and renders all 366 leap-year days", async () => {
    render(<ScanCoverageCalendar />);
    const region = await screen.findByRole("region", { name: "已扫描数据日历" });
    await screen.findByRole("button", { name: /2024-02-29/ });
    expect(within(region).getByRole("combobox", { name: "日历年份" })).toHaveValue("2024");
    expect(within(region).getByRole("combobox", { name: "日历站点" })).toHaveValue("");
    expect(region.querySelectorAll("button[data-date]")).toHaveLength(366);
    expect(within(region).getByRole("button", { name: /2024-01-01/ })).toHaveStyle({ gridRow: "3", gridColumn: "2" });
    expect(within(region).getByRole("button", { name: /2024-12-31/ })).toBeInTheDocument();
    for (const month of ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]) {
        expect(within(region).getByText(month)).toBeInTheDocument();
    }
    expect(within(region).getByText("2月")).toHaveStyle({ gridColumn: "6 / span 3" });
    expect(within(region).getByText("3月")).toHaveStyle({ gridColumn: "10 / span 3" });
});

it("switches to a non-leap year with exactly 365 day cells", async () => {
    render(<ScanCoverageCalendar />);
    await screen.findByRole("button", { name: /2024-02-29/ });
    fireEvent.change(screen.getByRole("combobox", { name: "日历年份" }), { target: { value: "2023" } });
    await screen.findByRole("button", { name: /2023-12-31/ });
    expect(document.querySelectorAll("button[data-date]")).toHaveLength(365);
    expect(screen.queryByRole("button", { name: /2024-02-29/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /2023-02-29/ })).not.toBeInTheDocument();
});

it("uses fixed daily duration bands and explains zero coverage and deduplication", async () => {
    vi.mocked(api.get).mockImplementation(async path => path === "/api/v1/sites" ? sites : coverage(2024, {
        "2024-01-02": 21600, "2024-01-03": 21601, "2024-01-04": 43201, "2024-01-05": 64801,
        "2024-01-06": 43200, "2024-01-07": 64800, "2024-01-08": 86400, "2024-01-09": 1,
    }));
    render(<ScanCoverageCalendar />);
    const zero = await screen.findByRole("button", { name: /2024-01-01.*暂无成功扫描记录/ });
    expect(zero).toHaveClass("scan-coverage-level-0");
    expect(zero).toHaveAttribute("title", expect.stringContaining("去重扫描 0 秒"));
    for (const [date, level] of [["2024-01-02", 1], ["2024-01-03", 2], ["2024-01-04", 3], ["2024-01-05", 4], ["2024-01-06", 2], ["2024-01-07", 3], ["2024-01-08", 4], ["2024-01-09", 1]] as const) {
        const day = screen.getByRole("button", { name: new RegExp(date) });
        expect(day).toHaveClass(`scan-coverage-level-${level}`);
        expect(day).toHaveAttribute("title", expect.stringContaining("北京时间"));
    }
    expect(screen.getByRole("button", { name: /2024-01-02/ })).toHaveAttribute("title", expect.stringContaining("6 小时"));
    expect(screen.getByText(/含模型判为地震、噪声和规则跳过的窗口/)).toBeInTheDocument();
    expect(screen.getByText(/全部站点.*并集去重.*不同模型.*历史成功/)).toBeInTheDocument();
    expect(screen.getByText(/灰色.*暂无成功扫描记录.*不代表.*源数据/)).toBeInTheDocument();
    expect(screen.getByText("少")).toBeInTheDocument();
    expect(screen.getByText("多")).toBeInTheDocument();
});

it("opens a focused day with the keyboard and shows the next-midnight exclusive boundary in Beijing time", async () => {
    vi.mocked(api.get).mockImplementation(async path => path === "/api/v1/sites" ? sites : path === "/api/v1/scan-coverage/day" ? {
        date: "2024-02-07", timezone: "Asia/Shanghai", scanned_seconds: 86400,
        intervals: [{ start_at: "2024-02-07T00:00:00+08:00", end_at: "2024-02-08T00:00:00+08:00" }],
    } : coverage(2024, { "2024-02-07": 86400 }));
    render(<ScanCoverageCalendar />);
    const day = await screen.findByRole("button", { name: /2024-02-07/ });
    day.focus();
    expect(day).toHaveFocus();
    await userEvent.keyboard("{Enter}");
    const detail = await screen.findByRole("region", { name: "2024-02-07 扫描覆盖详情" });
    expect(within(detail).getByText(/去重扫描 24 小时/)).toBeInTheDocument();
    expect(within(detail).getByText("2024-02-07 00:00:00")).toHaveAttribute("datetime", "2024-02-07T00:00:00+08:00");
    expect(within(detail).getByText("2024-02-08 00:00:00")).toBeInTheDocument();
    expect(within(detail).getByText(/右端不含/)).toBeInTheDocument();
});

it("shows an empty day's absence of scan records without claiming source data is absent", async () => {
    vi.mocked(api.get).mockImplementation(async path => path === "/api/v1/sites" ? [] : path === "/api/v1/scan-coverage/day" ? {
        date: "2024-01-01", timezone: "Asia/Shanghai", scanned_seconds: 0, intervals: [],
    } : { ...coverage(2024), available_years: [], days: [] });
    render(<ScanCoverageCalendar />);
    fireEvent.click(await screen.findByRole("button", { name: /2024-01-01/ }));
    const detail = await screen.findByRole("region", { name: "2024-01-01 扫描覆盖详情" });
    expect(await within(detail).findByText("当天暂无已扫描记录。")).toBeInTheDocument();
    expect(document.querySelectorAll("button[data-date]")).toHaveLength(366);
    expect(screen.getByRole("combobox", { name: "日历年份" })).toHaveValue("2024");
});

it("retries a failed day detail and replaces the error with real coverage", async () => {
    let failed = true;
    vi.mocked(api.get).mockImplementation(async path => {
        if (path === "/api/v1/sites") return sites;
        if (path !== "/api/v1/scan-coverage/day") return coverage(2024);
        if (failed) throw new Error("详情读取失败");
        return { date: "2024-01-01", timezone: "Asia/Shanghai", scanned_seconds: 3600,
            intervals: [{ start_at: "2024-01-01T08:00:00+08:00", end_at: "2024-01-01T09:00:00+08:00" }] };
    });
    render(<ScanCoverageCalendar />);
    fireEvent.click(await screen.findByRole("button", { name: /2024-01-01/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("详情读取失败");
    failed = false;
    fireEvent.click(screen.getByRole("button", { name: "重试当天详情" }));
    expect(await screen.findByText("2024-01-01 08:00:00")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("clears old coverage and details on filters, retains the chosen year, and discards an aborted site response", async () => {
    const oldSite = deferred<ReturnType<typeof coverage>>();
    let oldSignal: AbortSignal | undefined;
    vi.mocked(api.get).mockImplementation(async (path, params, signal) => {
        if (path === "/api/v1/sites") return sites;
        if (path === "/api/v1/scan-coverage/day") return { date: params?.date, timezone: "Asia/Shanghai", scanned_seconds: 0, intervals: [] };
        if (params?.site_id === "alpha" && params.year === "2024") { oldSignal = signal; return oldSite.promise; }
        if (params?.year === "2023") return coverage(2023, { "2023-01-01": 10800 });
        return coverage(2024, { "2024-01-01": 3600 });
    });
    render(<ScanCoverageCalendar />);
    fireEvent.click(await screen.findByRole("button", { name: /2024-01-01/ }));
    await screen.findByText("当天暂无已扫描记录。");
    fireEvent.change(screen.getByRole("combobox", { name: "日历站点" }), { target: { value: "alpha" } });
    expect(screen.queryByRole("region", { name: "2024-01-01 扫描覆盖详情" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /2024-01-01/ })).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "日历年份" })).toHaveValue("2024");
    fireEvent.change(screen.getByRole("combobox", { name: "日历年份" }), { target: { value: "2023" } });
    expect(await screen.findByRole("button", { name: /2023-01-01.*3 小时/ })).toBeInTheDocument();
    expect(oldSignal?.aborted).toBe(true);
    await act(async () => { oldSite.resolve(coverage(2024, { "2024-01-01": 43200 })); });
    expect(screen.getByRole("button", { name: /2023-01-01.*3 小时/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /2024-01-01/ })).not.toBeInTheDocument();
});

it("keeps only the newest clicked day's detail when a previous response arrives late", async () => {
    const previous = deferred<{ date: string; timezone: string; scanned_seconds: number; intervals: { start_at: string; end_at: string }[] }>();
    let previousSignal: AbortSignal | undefined;
    vi.mocked(api.get).mockImplementation(async (path, params, signal) => {
        if (path === "/api/v1/sites") return sites;
        if (path !== "/api/v1/scan-coverage/day") return coverage(2024);
        if (params?.date === "2024-01-01") { previousSignal = signal; return previous.promise; }
        return { date: "2024-01-02", timezone: "Asia/Shanghai", scanned_seconds: 0, intervals: [] };
    });
    render(<ScanCoverageCalendar />);
    fireEvent.click(await screen.findByRole("button", { name: /2024-01-01/ }));
    fireEvent.click(screen.getByRole("button", { name: /2024-01-02/ }));
    const detail = await screen.findByRole("region", { name: "2024-01-02 扫描覆盖详情" });
    expect(await within(detail).findByText("当天暂无已扫描记录。")).toBeInTheDocument();
    expect(previousSignal?.aborted).toBe(true);
    await act(async () => { previous.resolve({ date: "2024-01-01", timezone: "Asia/Shanghai", scanned_seconds: 86400,
        intervals: [{ start_at: "2024-01-01T00:00:00+08:00", end_at: "2024-01-02T00:00:00+08:00" }] }); });
    expect(within(detail).getByText("当天暂无已扫描记录。")).toBeInTheDocument();
    expect(screen.queryByText("2024-01-01 00:00:00")).not.toBeInTheDocument();
});

it("shows an annual query failure and retries without showing another filter's old cells", async () => {
    let failed = true;
    vi.mocked(api.get).mockImplementation(async (path, params) => {
        if (path === "/api/v1/sites") return sites;
        if (params?.year === "2023" && failed) throw new Error("年度读取失败");
        return coverage(Number(params?.year ?? 2024), { "2023-01-01": 7200 });
    });
    render(<ScanCoverageCalendar />);
    await screen.findByRole("button", { name: /2024-01-01/ });
    fireEvent.change(screen.getByRole("combobox", { name: "日历年份" }), { target: { value: "2023" } });
    expect(await screen.findByRole("alert")).toHaveTextContent("年度读取失败");
    expect(screen.queryByRole("button", { name: /2024-01-01/ })).not.toBeInTheDocument();
    failed = false;
    fireEvent.click(screen.getByRole("button", { name: "刷新日历" }));
    expect(await screen.findByRole("button", { name: /2023-01-01.*2 小时/ })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("keeps a site-list failure visible alongside a successful calendar and retries the site list", async () => {
    let failed = true;
    vi.mocked(api.get).mockImplementation(async path => {
        if (path !== "/api/v1/sites") return coverage(2024);
        if (failed) throw new Error("站点列表读取失败");
        return sites;
    });
    render(<ScanCoverageCalendar />);
    await screen.findByRole("button", { name: /2024-01-01/ });
    expect(screen.getByRole("alert")).toHaveTextContent("站点列表读取失败");
    failed = false;
    fireEvent.click(screen.getByRole("button", { name: "重试站点列表" }));
    expect(await screen.findByRole("option", { name: "甲站" })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("refreshes visible coverage at sixty seconds, pauses while hidden, and refreshes immediately when visible again", async () => {
    vi.useFakeTimers();
    let seconds = 3600;
    vi.mocked(api.get).mockImplementation(async path => path === "/api/v1/sites" ? sites : coverage(2024, { "2024-01-01": seconds }));
    await act(async () => { render(<ScanCoverageCalendar />); });
    expect(screen.getByRole("button", { name: /2024-01-01.*1 小时/ })).toBeInTheDocument();
    seconds = 7200;
    await act(async () => { await vi.advanceTimersByTimeAsync(59000); });
    expect(screen.getByRole("button", { name: /2024-01-01.*1 小时/ })).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByRole("button", { name: /2024-01-01.*2 小时/ })).toBeInTheDocument();
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "hidden" });
    seconds = 10800;
    await act(async () => { document.dispatchEvent(new Event("visibilitychange")); await vi.advanceTimersByTimeAsync(180000); });
    expect(screen.getByRole("button", { name: /2024-01-01.*2 小时/ })).toBeInTheDocument();
    expect(vi.mocked(api.get).mock.calls.filter(([path]) => path === "/api/v1/scan-coverage")).toHaveLength(2);
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
    await act(async () => { document.dispatchEvent(new Event("visibilitychange")); });
    expect(screen.getByRole("button", { name: /2024-01-01.*3 小时/ })).toBeInTheDocument();
    expect(vi.mocked(api.get).mock.calls.filter(([path]) => path === "/api/v1/scan-coverage")).toHaveLength(3);
    expect(vi.mocked(api.get).mock.calls.filter(([path]) => path === "/api/v1/sites")).toHaveLength(1);
});

it("retains current cells during a slow refresh and never overlaps annual reads", async () => {
    vi.useFakeTimers();
    const slow = deferred<ReturnType<typeof coverage>>();
    let reads = 0;
    vi.mocked(api.get).mockImplementation(async path => {
        if (path === "/api/v1/sites") return sites;
        reads++;
        return reads === 2 ? slow.promise : coverage(2024, { "2024-01-01": reads * 3600 });
    });
    await act(async () => { render(<ScanCoverageCalendar />); });
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
    expect(screen.getByRole("button", { name: /2024-01-01.*1 小时/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "刷新日历" })).toBeDisabled();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "刷新日历" })); document.dispatchEvent(new Event("visibilitychange")); await vi.advanceTimersByTimeAsync(180000); });
    expect(reads).toBe(2);
    await act(async () => { slow.resolve(coverage(2024, { "2024-01-01": 7200 })); });
    expect(screen.getByRole("button", { name: /2024-01-01.*2 小时/ })).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
    expect(screen.getByRole("button", { name: /2024-01-01.*3 小时/ })).toBeInTheDocument();
    expect(reads).toBe(3);
});

it("refreshes on revision and the refresh button and ignores the aborted earlier revision", async () => {
    vi.useFakeTimers();
    const first = deferred<ReturnType<typeof coverage>>();
    let firstSignal: AbortSignal | undefined;
    let reads = 0;
    vi.mocked(api.get).mockImplementation(async (path, _params, signal) => {
        if (path === "/api/v1/sites") return sites;
        reads++;
        if (reads === 1) { firstSignal = signal; return first.promise; }
        return coverage(2024, { "2024-01-01": reads * 3600 });
    });
    const view = render(<ScanCoverageCalendar revision={0} />);
    await act(async () => { view.rerender(<ScanCoverageCalendar revision={1} />); });
    expect(firstSignal?.aborted).toBe(true);
    expect(screen.getByRole("button", { name: /2024-01-01.*2 小时/ })).toBeInTheDocument();
    await act(async () => { first.resolve(coverage(2024, { "2024-01-01": 3600 })); });
    expect(screen.getByRole("button", { name: /2024-01-01.*2 小时/ })).toBeInTheDocument();
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "刷新日历" })); });
    expect(screen.getByRole("button", { name: /2024-01-01.*3 小时/ })).toBeInTheDocument();
    expect(reads).toBe(3);
});

it("refreshes a selected day's intervals when annual polling updates its coverage", async () => {
    vi.useFakeTimers();
    let hours = 1;
    vi.mocked(api.get).mockImplementation(async path => path === "/api/v1/sites" ? sites : path === "/api/v1/scan-coverage/day" ? {
        date: "2024-01-01", timezone: "Asia/Shanghai", scanned_seconds: hours * 3600,
        intervals: [{ start_at: "2024-01-01T00:00:00+08:00", end_at: `2024-01-01T0${hours}:00:00+08:00` }],
    } : coverage(2024, { "2024-01-01": hours * 3600 }));
    await act(async () => { render(<ScanCoverageCalendar />); });
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: /2024-01-01/ })); });
    const detail = screen.getByRole("region", { name: "2024-01-01 扫描覆盖详情" });
    expect(within(detail).getByText("2024-01-01 01:00:00")).toBeInTheDocument();
    hours = 2;
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
    expect(within(detail).getByText("2024-01-01 02:00:00")).toBeInTheDocument();
    expect(within(detail).queryByText("2024-01-01 01:00:00")).not.toBeInTheDocument();
    expect(vi.mocked(api.get).mock.calls.filter(([path]) => path === "/api/v1/scan-coverage/day")).toHaveLength(2);
});

it("aborts site and day reads on unmount and leaves no polling timer", async () => {
    vi.useFakeTimers();
    const signals: AbortSignal[] = [];
    vi.mocked(api.get).mockImplementation(async (path, _params, signal) => {
        if (signal) signals.push(signal);
        return path === "/api/v1/scan-coverage" ? coverage(2024) : new Promise(() => {});
    });
    let view!: ReturnType<typeof render>;
    await act(async () => { view = render(<ScanCoverageCalendar />); });
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: /2024-01-01/ })); });
    expect(screen.getByText("正在加载当天扫描区间…")).toBeInTheDocument();
    view.unmount();
    expect(signals).toHaveLength(3);
    expect(signals.every(signal => signal.aborted)).toBe(true);
    await act(async () => { await vi.advanceTimersByTimeAsync(180000); });
    expect(vi.mocked(api.get).mock.calls).toHaveLength(3);
});

it("clears a selected old-year detail if refreshing the default latest year returns a newer year", async () => {
    let latestYear = 2024;
    vi.mocked(api.get).mockImplementation(async (path, params) => path === "/api/v1/sites" ? sites : path === "/api/v1/scan-coverage/day" ? {
        date: params?.date, timezone: "Asia/Shanghai", scanned_seconds: 0, intervals: [],
    } : { ...coverage(latestYear), available_years: [2024, 2025] });
    render(<ScanCoverageCalendar />);
    fireEvent.click(await screen.findByRole("button", { name: /2024-01-01/ }));
    await screen.findByText("当天暂无已扫描记录。");
    latestYear = 2025;
    fireEvent.click(screen.getByRole("button", { name: "刷新日历" }));
    await screen.findByRole("button", { name: /2025-01-01/ });
    expect(screen.queryByRole("region", { name: "2024-01-01 扫描覆盖详情" })).not.toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "日历年份" })).toHaveValue("2025");
});
