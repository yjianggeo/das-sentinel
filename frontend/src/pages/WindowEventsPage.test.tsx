import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "../api";
import WindowEventsPage from "./WindowEventsPage";
import App from "../App";
vi.mock("../auth", () => ({ useAuth: () => ({ authenticated: true, loading: false, logout: vi.fn() }) }));
vi.mock("../api", async (original) => ({ ...await original<typeof import("../api")>(), api: { get: vi.fn(), post: vi.fn(), put: vi.fn(), getBlob: vi.fn() } }));
const item = { id: "w1", site_id: "s1", site_name: "站点一", site_timezone: "Asia/Shanghai", start_utc: "2024-01-01T00:00:00Z", end_utc: "2024-01-01T00:00:30Z", status: "succeeded", favorite: false, score: 4, priority: 100, legacy_count: 2, catalog_matches: [], analysis_id: "a1", input_sha256: "hash1", latest_review: null, features: {}, recognition_range_m: { start: 0 as number | null, stop: 100 as number | null }, shape: [20, 3000], unit: "strain", machine_labels: [], artifacts: [] as {
        kind: string;
        url: string;
    }[], legacy_events: [] };
let detail: typeof item;
beforeEach(() => {
    vi.clearAllMocks();
    detail = { ...item };
    vi.mocked(api.get).mockImplementation(async (url) => {
        if (url === "/api/v1/events")
            return [];
        if (url === "/api/v1/sites")
            return [{ id: "s1", name: "站点一" }];
        if (url.endsWith("/evidence"))
            return { items: [], total: 0, limit: 50, offset: 0 };
        if (url === "/api/v1/event-windows")
            return { items: [item], total: 31, limit: 30, offset: 0 };
        return detail;
    });
    vi.mocked(api.post).mockResolvedValue({ id: "saved-review", labels: { earthquake: "present" }, note: null, stale: false, created_at: "2026-10-09T10:00:00Z" });
    vi.mocked(api.put).mockResolvedValue({ window_id: "w1", favorite: true, created_at: item.start_utc });
});
function show() { render(<MemoryRouter><WindowEventsPage /></MemoryRouter>); }
it.each([[12345.6789, "12350"], [0.000123456, "0.0001235"], [0, "0"], [null, "—"]])("formats rule scores to four significant digits while preserving zero: %j", async (score, expected) => {
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [{ ...item, score }], total: 1, limit: 30, offset: 0 } : detail);
    show();
    expect(await screen.findByRole("button", { name: /站点一/ })).toHaveTextContent(`优先级 100 · 分数 ${expected}`);
});
it.each([[12345.6789, "12350"], [0.000123456, "0.0001235"], [0, "0"], [null, "—"]])("formats evidence rule scores while preserving zero: %j", async (score, expected) => {
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [item], total: 1, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [{ detector: "stalta", score }], total: 1, limit: 50, offset: 0 } : detail);
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByText("局部检测证据", { selector: "summary" }));
    const row = (await screen.findByText("stalta")).closest("tr")!;
    expect(within(row).getAllByRole("cell")[3]).toHaveTextContent(expected);
});
it("places review immediately after images and keeps only the four summary metrics", async () => {
    Object.assign(detail, { features: { active_channel_fraction: .25, active_contiguous_span_m: 500, lag_correlation_median: -.25, valid_channel_fraction: 0, peak_absolute: 99 } });
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    const summary = await screen.findByRole("region", { name: "复核摘要" });
    const images = screen.getByRole("region", { name: "整窗图件" });
    expect(images.nextElementSibling).toContainElement(screen.getByRole("button", { name: "保存整窗标签" }));
    expect(images.nextElementSibling?.nextElementSibling).toBe(summary);
    expect(within(summary).getAllByRole("term")).toHaveLength(4);
    expect(summary).toHaveTextContent("25.0%");
    expect(summary).toHaveTextContent("0.0%");
    expect(within(summary).queryByText(/更多测量参数/)).not.toBeInTheDocument();
});
it("defaults to borehole coverage sorting without changing the model recommendation scope", async () => {
    render(<MemoryRouter initialEntries={["/events?range=model_recommended&scan_id=batch"]}><WindowEventsPage /></MemoryRouter>);
    await screen.findByRole("button", { name: /站点一/ });
    expect(screen.getByLabelText("队列排序")).toHaveValue("borehole_coverage");
    expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows", expect.objectContaining({
        sort: "borehole_coverage", model_recommended: "true", model_scan_id: "batch", review_status: "unreviewed", offset: "0",
    }), expect.any(AbortSignal));
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByText("按井下广覆盖复核分从高到低排序")).toBeInTheDocument();
});
it("resets the page when changing sorting and retains the selected station and review filter", async () => {
    show();
    await screen.findByRole("option", { name: "站点一" });
    fireEvent.change(screen.getByLabelText("窗口站点"), { target: { value: "s1" } });
    fireEvent.change(screen.getByLabelText("复核状态"), { target: { value: "reviewed_earthquake" } });
    fireEvent.click(await screen.findByRole("button", { name: "下一页窗口" }));
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows", expect.objectContaining({ offset: "30" }), expect.any(AbortSignal)));
    fireEvent.change(screen.getByLabelText("队列排序"), { target: { value: "default" } });
    await waitFor(() => expect(api.get).toHaveBeenLastCalledWith("/api/v1/event-windows", expect.objectContaining({
        sort: "default", site_id: "s1", review_status: "reviewed_earthquake", candidate_only: "true", offset: "0",
    }), expect.any(AbortSignal)));
    expect(screen.getByRole("button", { name: "上一页窗口" })).toBeDisabled();
    expect(screen.getByLabelText("窗口站点")).toHaveValue("s1");
    expect(screen.getByLabelText("复核状态")).toHaveValue("reviewed_earthquake");
});
it("shows the independent coverage score without replacing the frozen model score", async () => {
    Object.assign(detail, { spatial_coverage: { algorithm_version: "borehole-coverage-v1", score: 86.4,
        coverage_fraction: .9, deep_coverage_fraction: .8, continuous_span_m: 1800,
        median_correlation: .7, valid_coverage_fraction: 1, analysis_id: "a1", input_sha256: "hash1" },
        model_recommendation: { scan_id: "batch", model_id: "frozen", model_name: "冻结模型", score: .72,
            threshold: .04, analysis_id: "a1", input_sha256: "hash1" } });
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [detail], total: 1, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : detail);
    render(<MemoryRouter initialEntries={["/events?range=model_recommended"]}><WindowEventsPage /></MemoryRouter>);
    const queue = screen.getByRole("complementary", { name: "整窗队列" });
    const window = await within(queue).findByRole("button", { name: /站点一/ });
    expect(window).toHaveTextContent("井下广覆盖 86.4 / 100");
    expect(window).toHaveTextContent("模型得分 0.7200");
    fireEvent.click(window);
    const coverage = await screen.findByRole("region", { name: "井下广覆盖" });
    expect(coverage).toHaveTextContent("井下广覆盖复核分 86.4 / 100");
    expect(coverage).toHaveTextContent("覆盖 90.0% · 深部覆盖 80.0% · 连续范围 1800 米");
    expect(coverage).toHaveTextContent("复核排序依据，不是地震概率");
    expect(screen.getByRole("region", { name: "扫描模型判定" })).toHaveTextContent("模型得分 0.7200");
});
const coverageMeasurement = { algorithm_version: "borehole-coverage-v1", score: 0,
    coverage_fraction: 0, deep_coverage_fraction: 0, continuous_span_m: 0, median_correlation: 0,
    valid_coverage_fraction: 1, analysis_id: "a1", input_sha256: "hash1" };
it.each([
    [undefined, "待测量"],
    [null, "待测量"],
    [{ ...coverageMeasurement, score: null }, "待测量"],
    [coverageMeasurement, "0.0 / 100"],
])("distinguishes missing coverage measurements from a measured zero score: %j", async (coverage, expected) => {
    Object.assign(detail, { spatial_coverage: coverage });
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [detail], total: 1, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : detail);
    show();
    const queue = screen.getByRole("complementary", { name: "整窗队列" });
    const window = await within(queue).findByRole("button", { name: /站点一/ });
    expect(window).toHaveTextContent(`井下广覆盖 ${expected}`);
    fireEvent.click(window);
    expect(await screen.findByRole("region", { name: "井下广覆盖" })).toHaveTextContent(`井下广覆盖复核分 ${expected}`);
});
const propagationMeasurement = { ...coverageMeasurement, algorithm_version: "borehole-coverage-v2", coverage_score: 80,
    score: 80, propagation_slowness_s_per_m: .0004, apparent_velocity_mps: 2500,
    propagation_residual_s: .002, propagation_direction: "+", propagation_confidence: .9,
    propagation_status: "resolved", common_mode_status: "unknown", common_mode_penalty: 0 };
async function showCoverageSummary(coverage: object) {
    Object.assign(detail, { spatial_coverage: coverage });
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [detail], total: 1, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : detail);
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    return screen.findByRole("region", { name: "井下广覆盖" });
}
it.each([
    ["旧载荷", coverageMeasurement, "待测传播"],
    ["缺少状态", { ...propagationMeasurement, propagation_status: undefined }, "待测传播"],
    ["零时延", { ...propagationMeasurement, propagation_status: "unresolved_zero_lag" }, "时延未分辨"],
    ["周期多峰", { ...propagationMeasurement, propagation_status: "ambiguous_periodic" }, "周期多峰，传播不确定"],
    ["低相关", { ...propagationMeasurement, propagation_status: "low_correlation" }, "传播未确定"],
    ["支持不足", { ...propagationMeasurement, propagation_status: "insufficient_support" }, "传播未确定"],
    ["边缘或边界", { ...propagationMeasurement, propagation_status: "edge_or_boundary" }, "传播未确定"],
    ["未知可信度", { ...propagationMeasurement, propagation_confidence: null }, "传播未确定"],
    ["低可信度", { ...propagationMeasurement, propagation_confidence: .59 }, "传播未确定"],
    ["非有限可信度", { ...propagationMeasurement, propagation_confidence: NaN }, "传播未确定"],
    ["超范围可信度", { ...propagationMeasurement, propagation_confidence: 1.1 }, "传播未确定"],
    ["零速度", { ...propagationMeasurement, apparent_velocity_mps: 0 }, "传播未确定"],
    ["负速度", { ...propagationMeasurement, apparent_velocity_mps: -2500 }, "传播未确定"],
    ["非有限速度", { ...propagationMeasurement, apparent_velocity_mps: Infinity }, "传播未确定"],
])("does not invent a precise velocity for %s", async (_case, coverage, expected) => {
    const summary = await showCoverageSummary(coverage);
    expect(summary).toHaveTextContent(expected);
    expect(summary).not.toHaveTextContent("km/s");
    expect(summary).not.toHaveTextContent("复核分已下调");
});
it.each([
    ["+", "向距离增大方向", .9, "90%"],
    ["-", "向距离减小方向", .6, "60%"],
])("shows resolved apparent velocity and measured confidence for direction %s", async (direction, expected, confidence, confidenceText) => {
    const summary = await showCoverageSummary({ ...propagationMeasurement, propagation_direction: direction, propagation_confidence: confidence });
    expect(summary).toHaveTextContent(`视速度 2.50 km/s · ${expected} · 可信度 ${confidenceText}`);
    expect(summary).toHaveTextContent("共模待定");
});
it("explains the confirmed common mode penalty while preserving the original coverage score", async () => {
    const summary = await showCoverageSummary({ ...propagationMeasurement, score: 68,
        common_mode_status: "persistent_supported", common_mode_penalty: 12 });
    expect(summary).toHaveTextContent("井下广覆盖复核分 68.0 / 100");
    expect(summary).toHaveTextContent("持续共模风险，复核分已下调（原覆盖分 80.0 / 100 → 调整分 68.0 / 100）");
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("井下广覆盖 68.0 / 100");
});
it.each([
    ["not_supported", 12, "未见持续共模证据"],
    ["unknown", 12, "共模待定"],
    ["persistent_supported", 0, "持续共模风险"],
])("does not claim a score reduction for common mode %s with penalty %s", async (status, penalty, expected) => {
    const summary = await showCoverageSummary({ ...propagationMeasurement,
        common_mode_status: status, common_mode_penalty: penalty });
    expect(summary).toHaveTextContent(expected);
    expect(summary).not.toHaveTextContent("复核分已下调");
    expect(summary).not.toHaveTextContent("原覆盖分");
    expect(summary).not.toHaveTextContent("无共模");
});
const componentMeasurement = { ...propagationMeasurement, algorithm_version: "borehole-coverage-v3",
    common_mode_status: "component_supported", common_mode_energy_fraction: .85,
    common_mode_channel_fraction: .9, residual_coverage_score: 28.5 };
it("uses the v3 component evidence wording when no strong same-phase component is supported", async () => {
    const summary = await showCoverageSummary({ ...componentMeasurement, common_mode_status: "not_supported", common_mode_penalty: 12 });
    expect(summary).toHaveTextContent("未见强同相共模证据");
    expect(summary).not.toHaveTextContent("未见持续共模证据");
    expect(summary).not.toHaveTextContent("复核分已下调");
});
it("shows the strong same-phase component and residual coverage behind the applied review penalty", async () => {
    const summary = await showCoverageSummary({ ...componentMeasurement, score: 68, common_mode_penalty: 12 });
    expect(summary).toHaveTextContent("强同相共模");
    expect(summary).toHaveTextContent("解释占比 85.0% · 残余覆盖 28.5 / 100");
    expect(summary).toHaveTextContent("复核分已下调（原覆盖分 80.0 / 100 → 调整分 68.0 / 100）");
    expect(summary).not.toHaveTextContent("已人工确认地震");
});
it.each([
    ["缺失", undefined, undefined, "解释占比 暂无 · 残余覆盖 暂无"],
    ["未知", null, null, "解释占比 暂无 · 残余覆盖 暂无"],
    ["非有限", NaN, Infinity, "解释占比 暂无 · 残余覆盖 暂无"],
    ["实测零值", 0, 0, "解释占比 0.0% · 残余覆盖 0.0 / 100"],
])("distinguishes %s common component measurements from a measured zero", async (_case, energyFraction, residualScore, expected) => {
    const summary = await showCoverageSummary({ ...componentMeasurement,
        common_mode_energy_fraction: energyFraction, residual_coverage_score: residualScore });
    expect(summary).toHaveTextContent("强同相共模");
    expect(summary).toHaveTextContent(expected);
    expect(summary).not.toHaveTextContent("复核分已下调");
});
it.each([
    ["component_supported", 0, "强同相共模"],
    ["component_supported", 12, "强同相共模"],
    ["persistent_supported", 12, "持续共模风险"],
    ["unknown", 0, "共模待定"],
])("keeps manually confirmed earthquake review priority for common mode %s with penalty %s", async (status, penalty, expected) => {
    const summary = await showCoverageSummary({ ...componentMeasurement,
        common_mode_status: status, common_mode_penalty: penalty, review_protected: true });
    expect(summary).toHaveTextContent(expected);
    expect(summary).toHaveTextContent("已人工确认地震，保留原复核优先级");
    expect(summary).toHaveTextContent("井下广覆盖复核分 80.0 / 100");
    expect(summary).not.toHaveTextContent("复核分已下调");
    expect(summary).not.toHaveTextContent("调整分");
    if (status === "component_supported")
        expect(summary).toHaveTextContent("解释占比 85.0% · 残余覆盖 28.5 / 100");
});
it.each([
    [false, "present", "地震类（含爆破）", 80, 0, true],
    [true, "absent", "噪声类（含车辆）", 68, 12, false],
])("refreshes server coverage after changing review protection from %s to %s", async (protectedBefore, label, className, score, penalty, protectedAfter) => {
    await showCoverageSummary({ ...componentMeasurement, score: protectedBefore ? 80 : 68,
        common_mode_penalty: protectedBefore ? 0 : 12, review_protected: protectedBefore });
    const review = { id: "new-review", labels: { earthquake: label }, note: null, stale: false, created_at: "2026-10-09T10:00:00Z" };
    const refreshed = { ...detail, latest_review: review, spatial_coverage: { ...componentMeasurement,
        score, common_mode_penalty: penalty, review_protected: protectedAfter } };
    vi.mocked(api.post).mockResolvedValueOnce(review);
    vi.mocked(api.get).mockImplementation(async url => url.endsWith("/w1") ? refreshed : detail);
    fireEvent.click(screen.getByLabelText(className));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await waitFor(() => expect(screen.getByRole("region", { name: "井下广覆盖" })).toHaveTextContent(`井下广覆盖复核分 ${score.toFixed(1)} / 100`));
    const summary = screen.getByRole("region", { name: "井下广覆盖" });
    expect(summary).toHaveTextContent("解释占比 85.0% · 残余覆盖 28.5 / 100");
    if (protectedAfter) {
        expect(summary).toHaveTextContent("已人工确认地震，保留原复核优先级");
        expect(summary).not.toHaveTextContent("复核分已下调");
    } else {
        expect(summary).toHaveTextContent("复核分已下调");
        expect(summary).not.toHaveTextContent("已人工确认地震");
    }
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent(`井下广覆盖 ${score.toFixed(1)} / 100`);
});
it("keeps a saved label but clears obsolete protection claims when coverage refresh fails", async () => {
    await showCoverageSummary({ ...componentMeasurement, score: 68, common_mode_penalty: 12, review_protected: false });
    vi.mocked(api.get).mockImplementation(async url => { if (url.endsWith("/w1")) throw new Error("coverage-offline"); return detail; });
    fireEvent.click(screen.getByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await screen.findByText("标签已保存，但复核分刷新失败；请刷新窗口重试。");
    expect(screen.getByText("整窗标签已保存")).toBeInTheDocument();
    const summary = screen.getByRole("region", { name: "井下广覆盖" });
    expect(summary).toHaveTextContent("井下广覆盖复核分 待测量");
    expect(summary).not.toHaveTextContent("复核分已下调");
    expect(summary).not.toHaveTextContent("已人工确认地震");
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("✓ 人工：地震类（含爆破）");
});
it("ignores a coverage refresh for an analysis that changed after saving", async () => {
    await showCoverageSummary({ ...componentMeasurement, score: 68, common_mode_penalty: 12, review_protected: false });
    let finishCoverage: (value: unknown) => void = () => {};
    vi.mocked(api.get).mockImplementation(async url => url.endsWith("/w1") ? new Promise(resolve => { finishCoverage = resolve; }) : detail);
    fireEvent.click(screen.getByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await screen.findByText("整窗标签已保存");
    expect(screen.getByRole("region", { name: "井下广覆盖" })).toHaveTextContent("井下广覆盖复核分 待测量");
    const next = { ...detail, analysis_id: "a2", input_sha256: "hash2", spatial_coverage: { ...componentMeasurement, score: 37, review_protected: false } };
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/event-windows" ? { items: [next], total: 1, limit: 30, offset: 0 } : next);
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    await waitFor(() => expect(screen.getByRole("region", { name: "井下广覆盖" })).toHaveTextContent("37.0 / 100"));
    await act(async () => { finishCoverage({ ...detail, latest_review: { id: "saved-review", labels: { earthquake: "present" }, note: null, stale: false, created_at: "2026-10-09T10:00:00Z" }, spatial_coverage: { ...componentMeasurement, score: 80, review_protected: true } }); });
    expect(screen.getByRole("region", { name: "井下广覆盖" })).toHaveTextContent("井下广覆盖复核分 37.0 / 100");
    expect(screen.getByRole("region", { name: "井下广覆盖" })).not.toHaveTextContent("已人工确认地震");
});
it("preserves refreshed coverage when earlier list and detail responses complete later", async () => {
    const before = { ...item, spatial_coverage: { ...componentMeasurement, score: 68, common_mode_penalty: 12, review_protected: false } };
    const review = { id: "saved-review", labels: { earthquake: "present" }, note: null, stale: false, created_at: "2026-10-09T10:00:00Z" };
    const after = { ...before, latest_review: review, spatial_coverage: { ...componentMeasurement, score: 80, common_mode_penalty: 0, review_protected: true } };
    let listReads = 0, detailReads = 0;
    let finishList: (value: unknown) => void = () => {};
    let finishDetail: (value: unknown) => void = () => {};
    let finishSave: (value: unknown) => void = () => {};
    vi.mocked(api.get).mockImplementation(async url => {
        if (url === "/api/v1/sites") return [];
        if (url === "/api/v1/event-windows") return ++listReads === 1 ? { items: [before], total: 1, limit: 30, offset: 0 } : new Promise(resolve => { finishList = resolve; });
        if (url.endsWith("/w1")) return ++detailReads === 1 ? before : detailReads === 2 ? new Promise(resolve => { finishDetail = resolve; }) : after;
        return null;
    });
    vi.mocked(api.post).mockImplementationOnce(() => new Promise(resolve => { finishSave = resolve; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    await waitFor(() => expect([listReads, detailReads]).toEqual([2, 2]));
    await act(async () => { finishSave(review); });
    await act(async () => { finishList({ items: [before], total: 1, limit: 30, offset: 0 }); finishDetail(before); });
    const summary = await screen.findByRole("region", { name: "井下广覆盖" });
    expect(summary).toHaveTextContent("井下广覆盖复核分 80.0 / 100");
    expect(summary).toHaveTextContent("已人工确认地震，保留原复核优先级");
    expect(summary).not.toHaveTextContent("复核分已下调");
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("井下广覆盖 80.0 / 100");
});
it("does not replace a newer review's coverage with an earlier save refresh", async () => {
    await showCoverageSummary({ ...componentMeasurement, score: 68, common_mode_penalty: 12, review_protected: false });
    const earlier = { id: "first-review", labels: { earthquake: "present" }, note: null, stale: false, created_at: "2026-10-09T10:00:00Z" };
    const latest = { id: "latest-review", labels: { earthquake: "absent" }, note: null, stale: false, created_at: "2026-10-09T10:00:01Z" };
    let finishFirstCoverage: (value: unknown) => void = () => {};
    let coverageReads = 0;
    vi.mocked(api.get).mockImplementation(async url => url.endsWith("/w1")
        ? ++coverageReads === 1 ? new Promise(resolve => { finishFirstCoverage = resolve; })
            : { ...detail, latest_review: latest, spatial_coverage: { ...componentMeasurement, score: 68, common_mode_penalty: 12, review_protected: false } }
        : detail);
    let saves = 0;
    vi.mocked(api.post).mockImplementation(async () => ++saves === 1 ? earlier : latest);
    fireEvent.click(screen.getByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await screen.findByText("整窗标签已保存");
    expect(screen.getByRole("region", { name: "井下广覆盖" })).toHaveTextContent("井下广覆盖复核分 待测量");
    fireEvent.click(screen.getByLabelText("噪声类（含车辆）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await waitFor(() => expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("✓ 人工：噪声类（含车辆）"));
    await act(async () => { finishFirstCoverage({ ...detail, latest_review: earlier, spatial_coverage: { ...componentMeasurement, score: 80, common_mode_penalty: 0, review_protected: true } }); });
    const summary = screen.getByRole("region", { name: "井下广覆盖" });
    expect(summary).toHaveTextContent("井下广覆盖复核分 68.0 / 100");
    expect(summary).toHaveTextContent("复核分已下调");
    expect(summary).not.toHaveTextContent("已人工确认地震");
});
it("opens model recommendations with the frozen score and without current-model inference", async () => {
    Object.assign(detail, { candidate: false, model_recommendation: { scan_id: "batch", model_id: "frozen",
        model_name: "冻结的 300 m 模型", score: .72, threshold: .04, analysis_id: "a1", input_sha256: "hash1" } });
    render(<MemoryRouter initialEntries={["/events?range=model_recommended&scan_id=batch"]}><WindowEventsPage /></MemoryRouter>);
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows", expect.objectContaining({
        model_recommended: "true", model_scan_id: "batch", candidate_only: "false", review_status: "unreviewed",
    }), expect.any(AbortSignal)));
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    expect(await screen.findByText(/冻结的 300 m 模型/)).toBeInTheDocument();
    expect(screen.getByText(/模型得分 0.7200/)).toBeInTheDocument();
    expect(screen.getByText(/未触发合格信号/)).toBeInTheDocument();
    expect(vi.mocked(api.get).mock.calls.some(([url]) => url.endsWith("/prediction"))).toBe(false);
});
it.each([
    ["漏检", "present", "地震类（含爆破）"],
    ["误报", "absent", "噪声类（含车辆）"],
])("shows model %s attention separately from the retained human label", async (kind, label, className) => {
    const review = { id: "original-review", labels: { earthquake: label }, stale: false,
        note: "原备注", created_at: item.start_utc };
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : {
        items: [{ ...item, latest_review: review, model_review_attention: { kind: kind === "漏检" ? "FN" : "FP", source_review_id: review.id, note: "评估说明" } }], total: 1, limit: 30, offset: 0,
    });
    show();
    const queue = screen.getByRole("complementary", { name: "整窗队列" });
    expect(await within(queue).findByText(`模型${kind}待复核`)).toBeInTheDocument();
    expect(within(queue).getByText(`✓ 人工：${className}`)).toBeInTheDocument();
});
it("resolves a model attention marker only when the user saves a determined label", async () => {
    const note = "原人工备注保留。";
    Object.assign(detail, { latest_review: { id: "original-review", labels: { earthquake: "absent" }, stale: false, note },
        model_review_attention: { kind: "FP", source_review_id: "original-review", note: "评估时批量标签为噪声，模型判断为地震。" } });
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [detail], total: 1, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : detail);
    vi.mocked(api.post).mockImplementation(async (_url, body) => ({ ...(body as object), id: "new-review", stale: false, created_at: "2026-10-09T10:00:00Z" }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    expect(await screen.findByRole("textbox", { name: "复核备注" })).toHaveValue(note);
    fireEvent.click(screen.getByRole("button", { name: "设为待确定" }));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/v1/event-windows/w1/reviews", expect.objectContaining({ note, labels: { earthquake: "unknown" } })));
    await screen.findByText("整窗标签已保存");
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByText("模型误报待复核")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("噪声类（含车辆）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await waitFor(() => expect(api.post).toHaveBeenLastCalledWith("/api/v1/event-windows/w1/reviews", expect.objectContaining({ note, labels: { earthquake: "absent" } })));
    expect(await within(screen.getByRole("complementary", { name: "整窗队列" })).findByText("模型误报已复核")).toBeInTheDocument();
    expect(screen.getByText("评估时批量标签为噪声，模型判断为地震。")).toBeInTheDocument();
});
it("uses thirty seconds even for an old history bookmark and removes its entry", async () => {
    render(<MemoryRouter initialEntries={["/events?window_seconds=60"]}><WindowEventsPage /></MemoryRouter>);
    expect(await screen.findByRole("heading", { name: "30 秒全通道窗口" })).toBeInTheDocument();
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows", expect.objectContaining({ window_seconds: "30" }), expect.any(AbortSignal)));
    expect(screen.queryByRole("link", { name: "历史 60 秒窗口" })).not.toBeInTheDocument();
});
it("uses one binary choice and sends the single-key review contract", async () => {
    detail = { ...item };
    Object.assign(detail, { legacy_label_import: { state: "needs_review", history_count: 8, event_count: 6,
        labels: { earthquake: "unknown", blast: "unknown", traffic: "unknown" }, evidence: [] } });
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    const classification = await screen.findByRole("group", { name: "窗口类别" });
    expect(within(classification).getByLabelText("地震类（含爆破）")).not.toBeChecked();
    expect(within(classification).getByLabelText("噪声类（含车辆）")).not.toBeChecked();
    expect(within(classification).queryByRole("radio", { name: "待确定" })).not.toBeInTheDocument();
    fireEvent.click(within(classification).getByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/v1/event-windows/w1/reviews", expect.objectContaining({ analysis_id: "a1", input_sha256: "hash1", labels: { earthquake: "present" }, note: "", request_id: expect.any(String) })));
    expect(screen.getByText(/传统规则证据/)).toBeInTheDocument();
    expect(screen.getByText(/类别由人工按整窗判断/)).toBeInTheDocument();
    const queue = screen.getByRole("complementary", { name: "整窗队列" });
    expect(await within(queue).findByText("✓ 人工：地震类（含爆破）")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "旧标签转换" })).not.toBeInTheDocument();
    expect(screen.queryByText(/历史局部线索/)).not.toBeInTheDocument();
});
it("can return a selected class to the pending state without adding a third radio option", async () => {
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "设为待确定" }));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/v1/event-windows/w1/reviews", expect.objectContaining({ labels: { earthquake: "unknown" } })));
});
it.each<[{ earthquake: "present" | "absent" | "unknown" }, boolean, string]>([
    [{ earthquake: "present" }, false, "✓ 人工：地震类（含爆破）"],
    [{ earthquake: "absent" }, false, "✓ 人工：噪声类（含车辆）"],
    [{ earthquake: "unknown" }, false, "人工：待确定"],
    [{ earthquake: "present" }, true, "人工标签已失效 · 待重新复核"],
])("shows saved queue labels without treating unknown or stale labels as complete: %j", async (labels, stale, expected) => {
    vi.mocked(api.get).mockImplementation(async (url) => url === "/api/v1/sites" ? [] : {
        items: [{ ...item, latest_review: { labels, stale, note: null, created_at: item.start_utc } }], total: 1, limit: 30, offset: 0,
    });
    show();
    const queue = screen.getByRole("complementary", { name: "整窗队列" });
    expect(await within(queue).findByText(expected)).toBeInTheDocument();
});
it("uses server pagination and resets all/site filters", async () => {
    show();
    await screen.findByRole("button", { name: /站点一/ });
    fireEvent.click(screen.getByRole("button", { name: "下一页窗口" }));
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows", expect.objectContaining({ offset: "30", candidate_only: "true", recommended: "false" }), expect.any(AbortSignal)));
    fireEvent.change(screen.getByLabelText("窗口范围"), { target: { value: "all" } });
    fireEvent.change(screen.getByLabelText("窗口站点"), { target: { value: "s1" } });
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows", expect.objectContaining({ offset: "0", candidate_only: "false", recommended: "false", site_id: "s1" }), expect.any(AbortSignal)));
    fireEvent.change(screen.getByLabelText("窗口范围"), { target: { value: "recommended" } });
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows", expect.objectContaining({ candidate_only: "false", recommended: "true" }), expect.any(AbortSignal)));
});
it.each([
    ["reviewed_earthquake", "已复核·地震类", "model_recommended", "unreviewed"],
    ["reviewed_noise", "已复核·噪声类", "model_recommended", "unreviewed"],
    ["reviewed_earthquake", "已复核·地震类", "favorites", "all"],
    ["reviewed_noise", "已复核·噪声类", "favorites", "all"],
])("requests %s (%s) within %s while retaining scope and resetting pagination", async (review, label, range, initialReview) => {
    vi.mocked(api.get).mockImplementation(async (url, params) => url === "/api/v1/sites" ? [{ id: "s1", name: "站点一" }] : {
        items: [{ ...item, favorite: true }], total: 61, limit: 30, offset: Number((params as Record<string, string>).offset),
    });
    render(<MemoryRouter initialEntries={[`/events?range=${range}&scan_id=batch`]}><WindowEventsPage /></MemoryRouter>);
    await screen.findByRole("option", { name: "站点一" });
    fireEvent.change(screen.getByLabelText("窗口站点"), { target: { value: "s1" } });
    fireEvent.click(await screen.findByRole("button", { name: "下一页窗口" }));
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows", expect.objectContaining({
        site_id: "s1", review_status: initialReview, offset: "30",
    }), expect.any(AbortSignal)));
    fireEvent.change(screen.getByLabelText("复核状态"), { target: { value: review } });
    await waitFor(() => expect(api.get).toHaveBeenLastCalledWith("/api/v1/event-windows", {
        site_id: "s1", review_status: review, sort: "borehole_coverage", model_recommended: String(range === "model_recommended"),
        ...(range === "model_recommended" ? { model_scan_id: "batch" } : {}), candidate_only: "false",
        recommended: "false", favorites_only: String(range === "favorites"), window_seconds: "30", limit: "30", offset: "0",
    }, expect.any(AbortSignal)));
    expect(screen.getByLabelText("窗口站点")).toHaveValue("s1");
    expect(screen.getByLabelText("窗口范围")).toHaveValue(range);
    expect(screen.getByRole("option", { name: label, selected: true })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "上一页窗口" })).toBeDisabled();
});
it("reloads conflicting inputs and resets draft labels", async () => {
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("噪声类（含车辆）"));
    detail = { ...item, input_sha256: "hash2" };
    vi.mocked(api.post).mockRejectedValueOnce(new ApiError(409, "input changed"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await screen.findByText(/输入已变更/);
    await waitFor(() => {
        const classification = screen.getByRole("group", { name: "窗口类别" });
        expect(within(classification).getByLabelText("地震类（含爆破）")).not.toBeChecked();
        expect(within(classification).getByLabelText("噪声类（含车辆）")).not.toBeChecked();
    });
});
it("blocks failed windows from saving", async () => {
    detail = { ...item, status: "failed" };
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    expect(await screen.findByRole("button", { name: "保存整窗标签" })).toBeDisabled();
});
it("keeps request ids for identical network retries", async () => {
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    await screen.findByRole("group", { name: "窗口类别" });
    vi.mocked(api.post).mockRejectedValueOnce(new Error("网络断开"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await screen.findByText("网络断开");
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).queryByText(/✓ 人工/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await screen.findByText("整窗标签已保存");
    expect(vi.mocked(api.post).mock.calls[0][1]).toEqual(vi.mocked(api.post).mock.calls[1][1]);
});
it("loads registered full-window images without legacy preview calls and revokes URLs", async () => {
    const create = vi.fn(() => "blob:window");
    const revoke = vi.fn();
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: create });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revoke });
    detail = { ...item, artifacts: [{ kind: "waterfall", url: "/api/v1/event-windows/w1/artifacts/waterfall?analysis_id=a1" }] } as typeof item;
    vi.mocked(api.getBlob).mockResolvedValue(new Blob(["png"]));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    expect(await screen.findByAltText("30 秒全通道原始瀑布图")).toHaveAttribute("src", "blob:window");
    expect(screen.getByText(/待补图/)).toBeInTheDocument();
    expect(api.getBlob).toHaveBeenCalledWith(detail.artifacts[0].url, expect.any(AbortSignal));
    expect(api.get).not.toHaveBeenCalledWith(expect.stringContaining("waterfall-options"), expect.anything(), expect.anything());
    fireEvent.click(screen.getByRole("button", { name: "放大30 秒全通道原始瀑布图" }));
    const dialog = screen.getByRole("dialog");
    expect(dialog).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "关闭放大图片" })).toHaveFocus();
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(within(dialog).getByRole("link", { name: "下载图件" })).toHaveFocus();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("窗口范围"), { target: { value: "all" } });
    await waitFor(() => expect(revoke).toHaveBeenCalledWith("blob:window"));
    create.mockRestore();
    revoke.mockRestore();
});
it("routes /events to the whole-window page", async () => {
    render(<MemoryRouter initialEntries={["/events"]}><App /></MemoryRouter>);
    expect(await screen.findByRole("heading", { name: "30 秒全通道窗口" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "历史局部事件复核" })).not.toBeInTheDocument();
});
it("requests all review states for favorites and clears the model scan while retaining the site", async () => {
    render(<MemoryRouter initialEntries={["/events?range=model_recommended&scan_id=batch"]}><WindowEventsPage /></MemoryRouter>);
    await screen.findByRole("button", { name: /站点一/ });
    fireEvent.change(screen.getByLabelText("窗口站点"), { target: { value: "s1" } });
    fireEvent.change(screen.getByLabelText("窗口范围"), { target: { value: "favorites" } });
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows", expect.objectContaining({
        site_id: "s1", favorites_only: "true", review_status: "all", model_recommended: "false", candidate_only: "false", recommended: "false", offset: "0",
    }), expect.any(AbortSignal)));
    const favoritesCall = vi.mocked(api.get).mock.calls.find(([url, params]) => url === "/api/v1/event-windows" && (params as Record<string, string>).favorites_only === "true");
    expect(favoritesCall?.[1]).not.toHaveProperty("model_scan_id");
    fireEvent.change(screen.getByLabelText("窗口范围"), { target: { value: "candidates" } });
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows", expect.objectContaining({ favorites_only: "false", site_id: "s1" }), expect.any(AbortSignal)));
});
it("only enables a new favorite when a successful thirty-second window has a registered waterfall", async () => {
    detail = { ...item, artifacts: [{ kind: "waterfall", url: "/waterfall" }] };
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    const favorite = await screen.findByRole("button", { name: "☆ 收藏" });
    expect(favorite).toBeEnabled();
    fireEvent.click(favorite);
    await waitFor(() => expect(api.put).toHaveBeenCalledWith("/api/v1/event-windows/w1/favorite", { favorite: true }));
    expect(await screen.findByRole("button", { name: "★ 已收藏" })).toBeInTheDocument();
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByText("★ 已收藏")).toBeInTheDocument();
    expect(api.post).not.toHaveBeenCalled();
});
it("allows a registered filtered waterfall to qualify for a new favorite", async () => {
    detail = { ...item, artifacts: [{ kind: "waterfall_filtered", url: "/filtered-waterfall" }] };
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    expect(await screen.findByRole("button", { name: "☆ 收藏" })).toBeEnabled();
});
it("keeps a favorite removable during reanalysis and preserves the review draft", async () => {
    detail = { ...item, status: "running", favorite: true };
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    const note = await screen.findByRole("textbox", { name: "复核备注" });
    fireEvent.change(note, { target: { value: "正在填写的备注" } });
    fireEvent.click(screen.getByRole("button", { name: "★ 已收藏" }));
    await waitFor(() => expect(api.put).toHaveBeenCalledWith("/api/v1/event-windows/w1/favorite", { favorite: false }));
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("正在填写的备注");
    expect(api.post).not.toHaveBeenCalled();
});
it("ignores a late favorite response after switching to another window", async () => {
    const first = { ...item, artifacts: [{ kind: "waterfall", url: "/waterfall" }] };
    const second = { ...first, id: "w2", site_name: "站点二", favorite: false };
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [first, second], total: 2, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : url.endsWith("w2") ? second : first);
    let resolveFavorite: (value: unknown) => void = () => {};
    vi.mocked(api.put).mockImplementationOnce(() => new Promise(resolve => { resolveFavorite = resolve; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByRole("button", { name: "☆ 收藏" }));
    fireEvent.click(screen.getByRole("button", { name: /站点二/ }));
    await screen.findByRole("heading", { name: /站点二/ });
    fireEvent.change(screen.getByRole("textbox", { name: "复核备注" }), { target: { value: "第二个窗口草稿" } });
    resolveFavorite({ window_id: "w1", favorite: true, created_at: item.start_utc });
    await waitFor(() => expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("★ 已收藏"));
    expect(screen.getByRole("heading", { name: /站点二/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "☆ 收藏" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("第二个窗口草稿");
});
it("keeps a window in the current all-windows queue when an earlier favorites removal finishes late", async () => {
    const first = { ...item, favorite: true };
    const second = { ...item, id: "w2", site_name: "站点二" };
    vi.mocked(api.get).mockImplementation(async (url, params) => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? (params as Record<string, string>).favorites_only === "true" ? { items: [first], total: 1, limit: 30, offset: 0 } : { items: [first, second], total: 2, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : url.endsWith("w2") ? second : first);
    let resolveFavorite: (value: unknown) => void = () => {};
    vi.mocked(api.put).mockImplementationOnce(() => new Promise(resolve => { resolveFavorite = resolve; }));
    render(<MemoryRouter initialEntries={["/events?range=favorites"]}><WindowEventsPage /></MemoryRouter>);
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByRole("button", { name: "★ 已收藏" }));
    fireEvent.change(screen.getByLabelText("窗口范围"), { target: { value: "all" } });
    fireEvent.click(await screen.findByRole("button", { name: /站点二/ }));
    fireEvent.change(await screen.findByRole("textbox", { name: "复核备注" }), { target: { value: "第二窗口草稿" } });
    resolveFavorite({ window_id: "w1", favorite: false, created_at: null });
    await waitFor(() => expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).not.toHaveTextContent("★ 已收藏"));
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByText("共 2 个窗口")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /站点二/ })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("第二窗口草稿");
});
it("does not rewind a newly selected page for a late removal from the old favorites page", async () => {
    const first = { ...item, favorite: true };
    const second = { ...item, id: "w2", site_name: "站点二" };
    vi.mocked(api.get).mockImplementation(async (url, params) => {
        if (url === "/api/v1/sites") return [];
        if (url === "/api/v1/event-windows") {
            const filter = params as Record<string, string>;
            return { items: filter.offset === "30" ? [filter.favorites_only === "true" ? first : second] : [], total: 31, limit: 30, offset: Number(filter.offset) };
        }
        if (url.endsWith("/evidence")) return { items: [], total: 0, limit: 50, offset: 0 };
        return url.endsWith("w2") ? second : first;
    });
    let resolveFavorite: (value: unknown) => void = () => {};
    vi.mocked(api.put).mockImplementationOnce(() => new Promise(resolve => { resolveFavorite = resolve; }));
    render(<MemoryRouter initialEntries={["/events?range=favorites"]}><WindowEventsPage /></MemoryRouter>);
    fireEvent.click(await screen.findByRole("button", { name: "下一页窗口" }));
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByRole("button", { name: "★ 已收藏" }));
    fireEvent.change(screen.getByLabelText("窗口范围"), { target: { value: "all" } });
    fireEvent.click(await screen.findByRole("button", { name: "下一页窗口" }));
    fireEvent.click(await screen.findByRole("button", { name: /站点二/ }));
    fireEvent.change(await screen.findByRole("textbox", { name: "复核备注" }), { target: { value: "第二页草稿" } });
    await act(async () => { resolveFavorite({ window_id: "w1", favorite: false, created_at: null }); });
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("第二页草稿");
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点二/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "上一页窗口" })).toBeEnabled();
});
it("ignores an older add response after the same window has been removed", async () => {
    const first = { ...item, artifacts: [{ kind: "waterfall", url: "/waterfall" }] };
    const second = { ...item, id: "w2", site_name: "站点二" };
    vi.mocked(api.getBlob).mockResolvedValue(new Blob(["png"]));
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:window") });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
    let firstReads = 0;
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [first, second], total: 2, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : url.endsWith("w2") ? second : { ...first, favorite: ++firstReads > 1 });
    let resolveAdd: (value: unknown) => void = () => {};
    vi.mocked(api.put).mockImplementation((_url, body) => (body as { favorite: boolean }).favorite ? new Promise(resolve => { resolveAdd = resolve; }) : Promise.resolve({ window_id: "w1", favorite: false, created_at: null }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByRole("button", { name: "☆ 收藏" }));
    fireEvent.click(screen.getByRole("button", { name: /站点二/ }));
    fireEvent.click(screen.getByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByRole("button", { name: "★ 已收藏" }));
    expect(await screen.findByRole("button", { name: "☆ 收藏" })).toBeInTheDocument();
    await act(async () => { resolveAdd({ window_id: "w1", favorite: true, created_at: item.start_utc }); });
    expect(screen.getByRole("button", { name: "☆ 收藏" })).toBeInTheDocument();
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).not.toHaveTextContent("★ 已收藏");
});
it("keeps a confirmed favorite when earlier refresh responses arrive later", async () => {
    const first = { ...item, artifacts: [{ kind: "waterfall", url: "/waterfall" }] };
    vi.mocked(api.getBlob).mockResolvedValue(new Blob(["png"]));
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: vi.fn(() => "blob:window") });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: vi.fn() });
    let listReads = 0;
    let detailReads = 0;
    let resolveList: (value: unknown) => void = () => {};
    let resolveDetail: (value: unknown) => void = () => {};
    vi.mocked(api.get).mockImplementation(async url => {
        if (url === "/api/v1/sites") return [];
        if (url === "/api/v1/event-windows") return ++listReads === 1 ? { items: [first], total: 1, limit: 30, offset: 0 } : new Promise(resolve => { resolveList = resolve; });
        if (url.endsWith("/evidence")) return { items: [], total: 0, limit: 50, offset: 0 };
        if (url.endsWith("/w1")) return ++detailReads === 1 ? first : new Promise(resolve => { resolveDetail = resolve; });
        return first;
    });
    let resolveFavorite: (value: unknown) => void = () => {};
    vi.mocked(api.put).mockImplementationOnce(() => new Promise(resolve => { resolveFavorite = resolve; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByRole("button", { name: "☆ 收藏" }));
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    await waitFor(() => expect([listReads, detailReads]).toEqual([2, 2]));
    await act(async () => { resolveFavorite({ window_id: "w1", favorite: true, created_at: item.start_utc }); });
    await act(async () => {
        resolveList({ items: [first], total: 1, limit: 30, offset: 0 });
        resolveDetail(first);
    });
    expect(screen.getByRole("button", { name: "★ 已收藏" })).toBeInTheDocument();
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("★ 已收藏");
});
it("returns to the previous favorites page when a stale refresh loses its last item", async () => {
    const first = { ...item, favorite: true };
    let offsetThirtyReads = 0;
    let offsetZeroReads = 0;
    let resolveList: (value: unknown) => void = () => {};
    vi.mocked(api.get).mockImplementation(async (url, params) => {
        if (url === "/api/v1/sites") return [];
        if (url === "/api/v1/event-windows") {
            if ((params as Record<string, string>).offset === "0") {
                offsetZeroReads++;
                return { items: [], total: offsetZeroReads === 1 ? 31 : 30, limit: 30, offset: 0 };
            }
            return ++offsetThirtyReads === 1 ? { items: [first], total: 31, limit: 30, offset: 30 } : new Promise(resolve => { resolveList = resolve; });
        }
        if (url.endsWith("/evidence")) return { items: [], total: 0, limit: 50, offset: 0 };
        return first;
    });
    let resolveFavorite: (value: unknown) => void = () => {};
    vi.mocked(api.put).mockImplementationOnce(() => new Promise(resolve => { resolveFavorite = resolve; }));
    render(<MemoryRouter initialEntries={["/events?range=favorites"]}><WindowEventsPage /></MemoryRouter>);
    fireEvent.click(await screen.findByRole("button", { name: "下一页窗口" }));
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByRole("button", { name: "★ 已收藏" }));
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    await waitFor(() => expect(offsetThirtyReads).toBe(2));
    await act(async () => { resolveFavorite({ window_id: "w1", favorite: false, created_at: null }); });
    await act(async () => { resolveList({ items: [first], total: 31, limit: 30, offset: 30 }); });
    await waitFor(() => expect(offsetZeroReads).toBe(2));
    expect(screen.getByRole("button", { name: "上一页窗口" })).toBeDisabled();
});
it("removes an unfavorited item from favorites, adjusts the page, and keeps its open draft", async () => {
    detail = { ...item, favorite: true };
    vi.mocked(api.get).mockImplementation(async (url, params) => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: (params as Record<string, string>).offset === "30" ? [detail] : [], total: (params as Record<string, string>).offset === "30" ? 31 : 30, limit: 30, offset: Number((params as Record<string, string>).offset) } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : detail);
    vi.mocked(api.put).mockResolvedValue({ window_id: "w1", favorite: false, created_at: null });
    render(<MemoryRouter initialEntries={["/events?range=favorites"]}><WindowEventsPage /></MemoryRouter>);
    fireEvent.click(await screen.findByRole("button", { name: "下一页窗口" }));
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.change(await screen.findByRole("textbox", { name: "复核备注" }), { target: { value: "保留草稿" } });
    const firstPageRequests = vi.mocked(api.get).mock.calls.filter(([url, params]) => url === "/api/v1/event-windows" && (params as Record<string, string>).offset === "0").length;
    fireEvent.click(screen.getByRole("button", { name: "★ 已收藏" }));
    await waitFor(() => expect(vi.mocked(api.get).mock.calls.filter(([url, params]) => url === "/api/v1/event-windows" && (params as Record<string, string>).offset === "0")).toHaveLength(firstPageRequests + 1));
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByText("共 30 个窗口")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("保留草稿");
});
it("redirects old local-event bookmarks to the whole-window page", async () => {
    render(<MemoryRouter initialEntries={["/events/legacy"]}><App /></MemoryRouter>);
    expect(await screen.findByRole("heading", { name: "30 秒全通道窗口" })).toBeInTheDocument();
});
it("ignores a late conflict from a previously selected window", async () => {
    const second = { ...item, id: "w2", site_name: "站点二", analysis_id: "a2", input_sha256: "hash2" };
    vi.mocked(api.get).mockImplementation(async (url) => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [item, second], total: 2, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : url.endsWith("w2") ? second : item);
    let rejectSave: (error: Error) => void = () => { };
    vi.mocked(api.post).mockImplementationOnce(() => new Promise((_, reject) => { rejectSave = reject; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByRole("button", { name: "保存整窗标签" }));
    fireEvent.click(screen.getByRole("button", { name: /站点二/ }));
    await screen.findByRole("heading", { name: /站点二/ });
    fireEvent.click(within(screen.getByRole("group", { name: "窗口类别" })).getByLabelText("地震类（含爆破）"));
    rejectSave(new ApiError(409, "changed"));
    await waitFor(() => expect(screen.queryByText(/输入已变更/)).not.toBeInTheDocument());
    expect(within(screen.getByRole("group", { name: "窗口类别" })).getByLabelText("地震类（含爆破）")).toBeChecked();
});
it("pages evidence against the opened analysis version", async () => {
    vi.mocked(api.get).mockImplementation(async (url) => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [item], total: 1, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [{ detector: "stalta", peak_utc: item.start_utc, channel_start: 0, channel_stop: 4, score: 2, qualified: true }], total: 51, limit: 50, offset: 0 } : detail);
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByText("局部检测证据", { selector: "summary" }));
    fireEvent.click(await screen.findByRole("button", { name: "下一页证据" }));
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows/w1/evidence", { analysis_id: "a1", limit: "50", offset: "50" }, expect.any(AbortSignal)));
});

it("loads evidence only on first expansion and keeps its page when collapsed", async () => {
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [item], total: 1, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [{ detector: "stalta" }], total: 51, limit: 50, offset: 0 } : detail);
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    const summary = await screen.findByText("局部检测证据", { selector: "summary" });
    expect(vi.mocked(api.get).mock.calls.filter(([url]) => url.endsWith("/evidence"))).toHaveLength(0);
    fireEvent.click(summary);
    fireEvent.click(await screen.findByRole("button", { name: "下一页证据" }));
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/event-windows/w1/evidence", { analysis_id: "a1", limit: "50", offset: "50" }, expect.any(AbortSignal)));
    fireEvent.click(summary);
    fireEvent.click(summary);
    await act(async () => {});
    expect(vi.mocked(api.get).mock.calls.filter(([url]) => url.endsWith("/evidence"))).toHaveLength(2);
    expect(screen.getByRole("button", { name: "上一页证据" })).toBeEnabled();
});

it("isolates lazy evidence after an input change and retries its new request", async () => {
    let finishOld: (value: unknown) => void = () => {};
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [item], total: 1, limit: 30, offset: 0 } : url.endsWith("/evidence") ? new Promise(resolve => { finishOld = resolve; }) : detail);
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByText("局部检测证据", { selector: "summary" }));
    await waitFor(() => expect(vi.mocked(api.get).mock.calls.filter(([url]) => url.endsWith("/evidence"))).toHaveLength(1));
    const signal = vi.mocked(api.get).mock.calls.find(([url]) => url.endsWith("/evidence"))![2]!;
    const completeOld = finishOld;
    detail = { ...item, input_sha256: "hash2" };
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    await screen.findByRole("group", { name: "窗口类别" });
    await act(async () => { completeOld({ items: [{ detector: "old-evidence" }], total: 1, limit: 50, offset: 0 }); });
    expect(signal.aborted).toBe(true);
    expect(screen.queryByText("old-evidence")).not.toBeInTheDocument();
    expect(vi.mocked(api.get).mock.calls.filter(([url]) => url.endsWith("/evidence"))).toHaveLength(1);
    vi.mocked(api.get).mockImplementation(async url => url.endsWith("/evidence") ? Promise.reject(new Error("evidence-offline")) : detail);
    fireEvent.click(screen.getByText("局部检测证据", { selector: "summary" }));
    fireEvent.click(await screen.findByRole("button", { name: "重试证据" }));
    vi.mocked(api.get).mockResolvedValue({ items: [{ detector: "new-evidence" }], total: 1, limit: 50, offset: 0 });
    fireEvent.click(await screen.findByRole("button", { name: "重试证据" }));
    expect(await screen.findByText("new-evidence")).toBeInTheDocument();
});

it("confirms a successful save after switching windows without changing the new draft", async () => {
    const second = { ...item, id: "w2", site_name: "站点二", analysis_id: "a2", input_sha256: "hash2" };
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [item, second], total: 2, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : url.endsWith("w2") ? second : item);
    let finishSave: (value: unknown) => void = () => {};
    vi.mocked(api.post).mockImplementationOnce(() => new Promise(resolve => { finishSave = resolve; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    fireEvent.click(screen.getByRole("button", { name: /站点二/ }));
    fireEvent.click(await screen.findByLabelText("噪声类（含车辆）"));
    fireEvent.change(screen.getByRole("textbox", { name: "复核备注" }), { target: { value: "第二窗草稿" } });
    await act(async () => { finishSave({ id: "review1", labels: { earthquake: "present" }, note: "", stale: false, created_at: "2026-10-09T10:00:00Z" }); });
    const queue = screen.getByRole("complementary", { name: "整窗队列" });
    expect(within(queue).getByRole("button", { name: /站点一/ })).toHaveTextContent("✓ 人工：地震类（含爆破）");
    expect(within(queue).getByRole("button", { name: /站点二/ })).not.toHaveTextContent("✓ 人工");
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("第二窗草稿");
    expect(screen.getByLabelText("噪声类（含车辆）")).toBeChecked();
    expect(screen.getByRole("status")).toHaveTextContent(/整窗标签已保存.*站点一.*w1/);
    expect(vi.mocked(api.get).mock.calls.filter(([url]) => url === "/api/v1/event-windows")).toHaveLength(1);
});

it("clears the obsolete label warning after saving the matching current input", async () => {
    Object.assign(detail, { latest_review: { id: "stale-review", labels: { earthquake: "absent" }, note: null, stale: true, created_at: item.start_utc } });
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    expect(await screen.findByText("旧整窗标签已失效，请按当前输入重新复核。")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    await screen.findByText("整窗标签已保存");
    expect(screen.queryByText("旧整窗标签已失效，请按当前输入重新复核。")).not.toBeInTheDocument();
});

it("does not attach a late saved label to a different analysis of the same window", async () => {
    let finishSave: (value: unknown) => void = () => {};
    vi.mocked(api.post).mockImplementationOnce(() => new Promise(resolve => { finishSave = resolve; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    detail = { ...item, analysis_id: "a2", input_sha256: "hash2" };
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/event-windows" ? { items: [detail], total: 1, limit: 30, offset: 0 } : url.endsWith("/evidence") ? { items: [], total: 0, limit: 50, offset: 0 } : detail);
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    fireEvent.click(await screen.findByLabelText("噪声类（含车辆）"));
    await act(async () => { finishSave({ id: "review1", labels: { earthquake: "present" }, stale: false, note: "", created_at: "2026-10-09T10:00:00Z" }); });
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).queryByText(/✓ 人工/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("噪声类（含车辆）")).toBeChecked();
    expect(screen.getByRole("status")).toHaveTextContent(/整窗标签已保存.*w1/);
});

it("preserves a confirmed review when an earlier list and detail refresh complete later", async () => {
    const stale = { ...item, latest_review: { id: "old-review", labels: { earthquake: "absent" }, note: null, stale: true, created_at: item.start_utc } };
    let listReads = 0;
    let detailReads = 0;
    let finishList: (value: unknown) => void = () => {};
    let finishDetail: (value: unknown) => void = () => {};
    let finishSave: (value: unknown) => void = () => {};
    vi.mocked(api.get).mockImplementation(async url => {
        if (url === "/api/v1/sites") return [];
        if (url === "/api/v1/event-windows") return ++listReads === 1 ? { items: [stale], total: 1, limit: 30, offset: 0 } : new Promise(resolve => { finishList = resolve; });
        if (url.endsWith("/w1")) return ++detailReads === 1 ? stale : new Promise(resolve => { finishDetail = resolve; });
        return { items: [], total: 0, limit: 50, offset: 0 };
    });
    vi.mocked(api.post).mockImplementationOnce(() => new Promise(resolve => { finishSave = resolve; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    await waitFor(() => expect([listReads, detailReads]).toEqual([2, 2]));
    await act(async () => { finishSave({ id: "new-review", labels: { earthquake: "present" }, note: "已复核", stale: false, created_at: "2026-10-09T10:00:00Z" }); });
    await act(async () => { finishList({ items: [stale], total: 1, limit: 30, offset: 0 }); finishDetail(stale); });
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("✓ 人工：地震类（含爆破）");
    expect(screen.queryByText("旧整窗标签已失效，请按当前输入重新复核。")).not.toBeInTheDocument();
    expect(screen.getByLabelText("地震类（含爆破）")).toBeChecked();
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("已复核");
});

it.each([
    ["microsecond creation time", "ffffffff-ffff-ffff-ffff-ffffffffffff", "2026-10-09T10:00:00.000001Z", "00000000-0000-0000-0000-000000000001", "2026-10-09T10:00:00.000002+00:00"],
    ["review id when creation times are equal", "00000000-0000-0000-0000-000000000001", "2026-10-09T10:00:00Z", "00000000-0000-0000-0000-000000000002", "2026-10-09T10:00:00.000000+00:00"],
])("keeps the newer review by %s when two save responses arrive in reverse order", async (_ordering, firstId, firstTime, latestId, latestTime) => {
    const second = { ...item, id: "w2", site_name: "站点二", analysis_id: "a2", input_sha256: "hash2" };
    const earlier = { id: firstId, labels: { earthquake: "absent" }, note: "第一次保存", stale: false, created_at: firstTime };
    const latest = { id: latestId, labels: { earthquake: "present" }, note: "第二次保存", stale: false, created_at: latestTime };
    let refreshing = false;
    let finishList: (value: unknown) => void = () => {};
    let finishDetail: (value: unknown) => void = () => {};
    vi.mocked(api.get).mockImplementation(async url => {
        if (url === "/api/v1/sites") return [];
        if (url === "/api/v1/event-windows") return refreshing ? new Promise(resolve => { finishList = resolve; }) : { items: [item, second], total: 2, limit: 30, offset: 0 };
        if (url.endsWith("/w1")) return refreshing ? new Promise(resolve => { finishDetail = resolve; }) : item;
        if (url.endsWith("/w2")) return second;
        return null;
    });
    let finishFirst: (value: unknown) => void = () => {};
    let finishLatest: (value: unknown) => void = () => {};
    vi.mocked(api.post).mockImplementationOnce(() => new Promise(resolve => { finishFirst = resolve; }))
        .mockImplementationOnce(() => new Promise(resolve => { finishLatest = resolve; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("噪声类（含车辆）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    fireEvent.click(screen.getByRole("button", { name: /站点二/ }));
    await screen.findByRole("heading", { name: /站点二/ });
    fireEvent.click(screen.getByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("地震类（含爆破）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    expect(api.post).toHaveBeenCalledTimes(2);
    refreshing = true;
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    await waitFor(() => expect(screen.getByText("正在加载窗口详情…")).toBeInTheDocument());
    await act(async () => { finishLatest(latest); });
    await act(async () => { finishFirst(earlier); });
    await act(async () => { finishList({ items: [item, second], total: 2, limit: 30, offset: 0 }); finishDetail(item); });
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("✓ 人工：地震类（含爆破）");
    expect(screen.getByLabelText("地震类（含爆破）")).toBeChecked();
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("第二次保存");
});

it("does not downgrade a newer server review or show an old save as the current page confirmation", async () => {
    const second = { ...item, id: "w2", site_name: "站点二", analysis_id: "a2", input_sha256: "hash2" };
    const latest = { ...item, latest_review: { id: "review-newer", labels: { earthquake: "present" }, note: "较新服务端复核", stale: false, created_at: "2026-10-09T10:00:02Z" } };
    let fresh = false;
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] : url === "/api/v1/event-windows" ? { items: [fresh ? latest : item, second], total: 2, limit: 30, offset: 0 } : url.endsWith("/w2") ? second : url.endsWith("/w1") ? fresh ? latest : item : null);
    let finishSave: (value: unknown) => void = () => {};
    vi.mocked(api.post).mockImplementationOnce(() => new Promise(resolve => { finishSave = resolve; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("噪声类（含车辆）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    fireEvent.click(screen.getByRole("button", { name: /站点二/ }));
    await screen.findByRole("heading", { name: /站点二/ });
    fresh = true;
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    expect(await screen.findByRole("textbox", { name: "复核备注" })).toHaveValue("较新服务端复核");
    await act(async () => { finishSave({ id: "review-older", labels: { earthquake: "absent" }, note: "旧保存", stale: false, created_at: "2026-10-09T10:00:01Z" }); });
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("✓ 人工：地震类（含爆破）");
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("较新服务端复核");
    expect(screen.queryByText(/整窗标签已保存：/)).not.toBeInTheDocument();
});

it("does not replace a newer in-flight server review with an older local confirmation", async () => {
    const latest = { ...item, latest_review: { id: "review-newer", labels: { earthquake: "present" }, note: "较新服务端复核", stale: false, created_at: "2026-10-09T10:00:02Z" } };
    let refreshing = false;
    let finishList: (value: unknown) => void = () => {};
    let finishDetail: (value: unknown) => void = () => {};
    let finishSave: (value: unknown) => void = () => {};
    vi.mocked(api.get).mockImplementation(async url => {
        if (url === "/api/v1/sites") return [];
        if (url === "/api/v1/event-windows") return refreshing ? new Promise(resolve => { finishList = resolve; }) : { items: [item], total: 1, limit: 30, offset: 0 };
        if (url.endsWith("/w1")) return refreshing ? new Promise(resolve => { finishDetail = resolve; }) : item;
        return null;
    });
    vi.mocked(api.post).mockImplementationOnce(() => new Promise(resolve => { finishSave = resolve; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("噪声类（含车辆）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    refreshing = true;
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    await waitFor(() => expect(screen.getByText("正在加载窗口详情…")).toBeInTheDocument());
    await act(async () => { finishSave({ id: "review-older", labels: { earthquake: "absent" }, note: "旧保存", stale: false, created_at: "2026-10-09T10:00:01Z" }); });
    await act(async () => { finishList({ items: [latest], total: 1, limit: 30, offset: 0 }); finishDetail(latest); });
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("✓ 人工：地震类（含爆破）");
    expect(screen.getByLabelText("地震类（含爆破）")).toBeChecked();
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("较新服务端复核");
});

it("keeps the new input confirmation when an older input response precedes late new input reads", async () => {
    const nextInput = { ...item, analysis_id: "a2", input_sha256: "hash2" };
    let currentInput = item;
    let refreshing = false;
    let finishList: (value: unknown) => void = () => {};
    let finishDetail: (value: unknown) => void = () => {};
    vi.mocked(api.get).mockImplementation(async url => {
        if (url === "/api/v1/sites") return [];
        if (url === "/api/v1/event-windows") return refreshing ? new Promise(resolve => { finishList = resolve; }) : { items: [currentInput], total: 1, limit: 30, offset: 0 };
        if (url.endsWith("/w1")) return refreshing ? new Promise(resolve => { finishDetail = resolve; }) : currentInput;
        return null;
    });
    let finishOldInput: (value: unknown) => void = () => {};
    let finishNewInput: (value: unknown) => void = () => {};
    vi.mocked(api.post).mockImplementationOnce(() => new Promise(resolve => { finishOldInput = resolve; }))
        .mockImplementationOnce(() => new Promise(resolve => { finishNewInput = resolve; }));
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    fireEvent.click(await screen.findByLabelText("噪声类（含车辆）"));
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    currentInput = nextInput;
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    fireEvent.click(await screen.findByLabelText("地震类（含爆破）"));
    fireEvent.change(screen.getByRole("textbox", { name: "复核备注" }), { target: { value: "新输入已复核" } });
    fireEvent.click(screen.getByRole("button", { name: "保存整窗标签" }));
    expect(api.post).toHaveBeenNthCalledWith(1, "/api/v1/event-windows/w1/reviews", expect.objectContaining({ analysis_id: "a1", input_sha256: "hash1" }));
    expect(api.post).toHaveBeenNthCalledWith(2, "/api/v1/event-windows/w1/reviews", expect.objectContaining({ analysis_id: "a2", input_sha256: "hash2" }));
    refreshing = true;
    fireEvent.click(screen.getByRole("button", { name: "刷新窗口" }));
    await waitFor(() => expect(screen.getByText("正在加载窗口详情…")).toBeInTheDocument());
    await act(async () => { finishNewInput({ id: "review-new-input", labels: { earthquake: "present" }, note: "新输入已复核", stale: false, created_at: "2026-10-09T10:00:02Z" }); });
    await act(async () => { finishOldInput({ id: "review-old-input", labels: { earthquake: "absent" }, note: "旧输入", stale: false, created_at: "2026-10-09T10:00:01Z" }); });
    await act(async () => { finishList({ items: [nextInput], total: 1, limit: 30, offset: 0 }); finishDetail(nextInput); });
    expect(within(screen.getByRole("complementary", { name: "整窗队列" })).getByRole("button", { name: /站点一/ })).toHaveTextContent("✓ 人工：地震类（含爆破）");
    expect(screen.getByLabelText("地震类（含爆破）")).toBeChecked();
    expect(screen.getByRole("textbox", { name: "复核备注" })).toHaveValue("新输入已复核");
});


it.each([
    [{ start: 300, stop: null }, "识别范围 300 米至末道（图件保留全道）"],
    [{ start: null, stop: 500 }, "识别范围 首道至500 米（图件保留全道）"],
    [{ start: null, stop: null }, "识别范围 首道至末道（图件保留全道）"],
    [{ start: 0, stop: 100 }, "识别范围 0 米至100 米（图件保留全道）"],
])("shows open recognition boundaries explicitly: %j", async (range, expected) => {
    detail = { ...item, recognition_range_m: range };
    show();
    fireEvent.click(await screen.findByRole("button", { name: /站点一/ }));
    expect(await screen.findByText(expected)).toBeInTheDocument();
});
