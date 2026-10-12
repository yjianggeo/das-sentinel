import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { api } from "../api";
import { windowApi, type WindowDetail } from "../windowApi";
import WindowSignalViewer from "./WindowSignalViewer";

vi.mock("../api", async original => ({ ...await original<typeof import("../api")>(), api: { getBlob: vi.fn() } }));
const rawPath = "/api/v1/event-windows/w1/artifacts/waterfall?analysis_id=a1";
const filteredPath = "/api/v1/event-windows/w1/artifacts/waterfall_filtered?analysis_id=a1";
const rawPreviewPath = "/api/v1/event-windows/w1/region-preview?analysis_id=a1&region_id=1&filtered=false";
const filteredPreviewPath = "/api/v1/event-windows/w1/region-preview?analysis_id=a1&region_id=1&filtered=true";
const detail: WindowDetail = {
    id: "w1", site_id: "s1", site_name: "站点一", site_timezone: "Asia/Shanghai", window_seconds: 30,
    start_utc: "2024-01-01T00:00:00Z", end_utc: "2024-01-01T00:00:30Z", status: "succeeded", favorite: false,
    score: 4, priority: 100, legacy_count: 0, catalog_matches: [], analysis_id: "a1", input_sha256: "hash1",
    latest_review: null, features: {}, recognition_range_m: { start: 0, stop: 1000 }, shape: [200, 3000],
    unit: "strain", machine_labels: [], legacy_total: 0, legacy_events: [], candidate: true, visualization_version: 2,
    artifacts: [{ kind: "waterfall", url: rawPath }, { kind: "waterfall_filtered", url: filteredPath }, { kind: "waveform_plot", url: "/old-waveform" }],
    detection_regions: [
        { id: 1, start_sample: 600, end_sample: 1200, peak_sample: 900, channel_start: 40, channel_stop: 80, distance_start_m: 200, distance_stop_m: 400, score: 4, detectors: ["stalta", "band_energy"], evidence_count: 2 },
        { id: 2, start_sample: 1500, end_sample: 2100, peak_sample: 1800, channel_start: 100, channel_stop: 120, distance_start_m: 500, distance_stop_m: 600, score: 3, detectors: ["stalta"], evidence_count: 1 },
    ],
    plot_geometry: { axes: [0.1, 0.1, 0.8, 0.8], distance_min_m: 0, distance_max_m: 1000 },
};
const createUrl = vi.fn();
const revokeUrl = vi.fn();
beforeEach(() => {
    vi.resetAllMocks();
    let next = 0;
    createUrl.mockImplementation(() => `blob:signal-${++next}`);
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createUrl });
    Object.defineProperty(URL, "revokeObjectURL", { configurable: true, value: revokeUrl });
    vi.mocked(api.getBlob).mockResolvedValue(new Blob(["png"], { type: "image/png" }));
});

async function expandRegions() {
    const summary = screen.getByText(/疑似区域与局部预览/, { selector: "summary" });
    if (!(summary.parentElement as HTMLDetailsElement).open) fireEvent.click(summary);
    await screen.findByRole("region", { name: "疑似信号区域" });
}

it("mounts region controls only while expanded and aborts a preview when collapsed", async () => {
    let finishPreview!: (blob: Blob) => void;
    render(<WindowSignalViewer detail={detail}/>);
    await screen.findByAltText("30 秒全通道原始瀑布图");
    expect(screen.queryByRole("region", { name: "疑似信号区域" })).not.toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "局部预览" })).not.toBeInTheDocument();
    expect(api.getBlob).toHaveBeenCalledTimes(1);
    await expandRegions();
    vi.mocked(api.getBlob).mockImplementationOnce(() => new Promise(resolve => { finishPreview = resolve; }));
    fireEvent.click(screen.getByRole("button", { name: /^区域 1 ·/ }));
    const signal = vi.mocked(api.getBlob).mock.calls.at(-1)![1]!;
    fireEvent.click(screen.getByText(/疑似区域与局部预览/, { selector: "summary" }));
    await waitFor(() => expect(screen.queryByRole("region", { name: "局部预览" })).not.toBeInTheDocument());
    expect(signal.aborted).toBe(true);
    await act(async () => { finishPreview(new Blob(["late preview"])); });
    expect(createUrl).toHaveBeenCalledTimes(1);
    await expandRegions();
    expect(screen.getByRole("button", { name: /^区域 1 ·/ })).toHaveAttribute("aria-pressed", "true");
    await screen.findByAltText("区域 1 原始瀑布图局部预览");
});

it.each([false, true])("uses plain artifact PNGs across region toggles without redrawing (filtered=%s)", async filtered => {
    render(<WindowSignalViewer detail={Object.assign({ ...detail }, { waterfall_regions_baked: false })}/>);
    await screen.findByAltText("30 秒全通道原始瀑布图");
    expect(api.getBlob).toHaveBeenCalledWith(rawPath, expect.any(AbortSignal));
    if (filtered) {
        fireEvent.click(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" }));
        await screen.findByAltText("30 秒全通道滤波瀑布图");
        expect(api.getBlob).toHaveBeenCalledWith(filteredPath, expect.any(AbortSignal));
    }
    const calls = vi.mocked(api.getBlob).mock.calls.length;
    const fullUrl = screen.getByAltText(filtered ? "30 秒全通道滤波瀑布图" : "30 秒全通道原始瀑布图").getAttribute("src");
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    expect(screen.getByRole("button", { name: "查看区域 1" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "下载原图（不含疑似区域框）" })).toHaveAttribute("href", fullUrl);
    expect(api.getBlob).toHaveBeenCalledTimes(calls);
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    expect(screen.queryByRole("button", { name: "查看区域 1" })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "下载图件" })).toHaveAttribute("href", fullUrl);
    expect(api.getBlob).toHaveBeenCalledTimes(calls);
});

it.each([undefined, true])("retains legacy baked-image toggling (baked=%s)", async baked => {
    render(<WindowSignalViewer detail={Object.assign({ ...detail }, { waterfall_regions_baked: baked })}/>);
    await screen.findByAltText("30 秒全通道原始瀑布图");
    expect(api.getBlob).toHaveBeenCalledWith("/api/v1/event-windows/w1/waterfall-preview?analysis_id=a1&filtered=false", expect.any(AbortSignal));
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    await screen.findByAltText("30 秒全通道原始瀑布图");
    expect(api.getBlob).toHaveBeenCalledWith(rawPath, expect.any(AbortSignal));
});

it("maps numbered regions to the image axes on both waterfall tabs", async () => {
    render(<WindowSignalViewer detail={detail}/>);
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    await screen.findByAltText("30 秒全通道原始瀑布图");
    expect(screen.getAllByRole("tab")).toHaveLength(2);
    expect(api.getBlob).not.toHaveBeenCalledWith("/old-waveform", expect.anything());
    const checkBox = () => {
        const box = screen.getByRole("button", { name: "查看区域 1" });
        expect(parseFloat(box.style.left)).toBeCloseTo(26);
        expect(parseFloat(box.style.top)).toBeCloseTo(58);
        expect(parseFloat(box.style.width)).toBeCloseTo(16);
        expect(parseFloat(box.style.height)).toBeCloseTo(16);
    };
    checkBox();
    fireEvent.click(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" }));
    await screen.findByAltText("30 秒全通道滤波瀑布图");
    checkBox();
    expect(api.getBlob).toHaveBeenCalledWith(filteredPath, expect.any(AbortSignal));
});

it("explains incomplete raw coverage instead of claiming existing images are available", () => {
    render(<WindowSignalViewer detail={{ ...detail, status: "failed", artifacts: [], analysis_error_code: "coverage_incomplete" }}/>);
    expect(screen.getByText(/原始数据或前后文不连续/)).toBeInTheDocument();
    expect(screen.queryByText(/已有原始瀑布图仍可查看/)).not.toBeInTheDocument();
    expect(api.getBlob).not.toHaveBeenCalled();
});

it.each([undefined, false])("opens a region from its frame with the current analysis and reloads it when filtering changes (baked=%s)", async baked => {
    const { unmount } = render(<WindowSignalViewer detail={{ ...detail, waterfall_regions_baked: baked }}/>);
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    const rawFull = await screen.findByAltText("30 秒全通道原始瀑布图");
    const rawFullUrl = rawFull.getAttribute("src");
    fireEvent.click(screen.getByRole("button", { name: "查看区域 1" }));
    const preview = await screen.findByAltText("区域 1 原始瀑布图局部预览");
    const oldPreviewUrl = preview.getAttribute("src");
    expect(api.getBlob).toHaveBeenCalledWith(rawPreviewPath, expect.any(AbortSignal));
    expect(screen.getByRole("button", { name: /^区域 1 ·/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText(/标注仍保存到整个 30 秒窗口/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" }));
    expect(screen.queryByAltText("区域 1 原始瀑布图局部预览")).not.toBeInTheDocument();
    await screen.findByAltText("区域 1 滤波瀑布图局部预览");
    expect(api.getBlob).toHaveBeenCalledWith(filteredPreviewPath, expect.any(AbortSignal));
    expect(revokeUrl).not.toHaveBeenCalledWith(rawFullUrl);
    expect(revokeUrl).not.toHaveBeenCalledWith(oldPreviewUrl);
    unmount();
    expect(revokeUrl).toHaveBeenCalledTimes(4);
});

it("discards a late region response after keyboard selection of another region", async () => {
    let finishFirst: (blob: Blob) => void = () => {};
    vi.mocked(api.getBlob).mockImplementation(path => path === rawPreviewPath ? new Promise(resolve => { finishFirst = resolve; }) : Promise.resolve(new Blob(["png"])));
    render(<WindowSignalViewer detail={detail}/>);
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    await expandRegions();
    const first = await screen.findByRole("button", { name: /^区域 1 ·/ });
    first.focus();
    await userEvent.keyboard("{Enter}");
    const pendingSignal = vi.mocked(api.getBlob).mock.calls.find(([path]) => path === rawPreviewPath)![1]!;
    screen.getByRole("button", { name: /^区域 2 ·/ }).focus();
    await userEvent.keyboard(" ");
    const second = await screen.findByAltText("区域 2 原始瀑布图局部预览");
    const secondUrl = second.getAttribute("src");
    expect(pendingSignal.aborted).toBe(true);
    await act(async () => { finishFirst(new Blob(["old"])); });
    expect(second).toHaveAttribute("src", secondUrl);
    expect(createUrl).toHaveBeenCalledTimes(2);
});

it.each([undefined, false])("aborts old full images and clears region selection when the analysis changes (baked=%s)", async baked => {
    let finishOld: (blob: Blob) => void = () => {};
    vi.mocked(api.getBlob).mockImplementation(path => path === rawPath ? new Promise(resolve => { finishOld = resolve; }) : Promise.resolve(new Blob(["new"])));
    const { rerender } = render(<WindowSignalViewer detail={{ ...detail, waterfall_regions_baked: baked }}/>);
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    await expandRegions();
    fireEvent.click(await screen.findByRole("button", { name: /^区域 1 ·/ }));
    const previousPreview = await screen.findByAltText("区域 1 原始瀑布图局部预览");
    const previousUrl = previousPreview.getAttribute("src");
    const previousSignal = vi.mocked(api.getBlob).mock.calls.find(([path]) => path === rawPath)![1]!;
    rerender(<WindowSignalViewer detail={{ ...detail, waterfall_regions_baked: baked, analysis_id: "a2", artifacts: [{ kind: "waterfall", url: "/new-raw?analysis_id=a2" }, { kind: "waterfall_filtered", url: "/new-filtered?analysis_id=a2" }] }}/>);
    const current = await screen.findByAltText("30 秒全通道原始瀑布图");
    const currentUrl = current.getAttribute("src");
    expect(previousSignal.aborted).toBe(true);
    expect(revokeUrl).toHaveBeenCalledWith(previousUrl);
    expect(screen.queryByAltText("区域 1 原始瀑布图局部预览")).not.toBeInTheDocument();
    await act(async () => { finishOld(new Blob(["old"])); });
    expect(current).toHaveAttribute("src", currentUrl);
    expect(createUrl).toHaveBeenCalledTimes(2);
});

it("reuses the original image when returning before the other tab loads", async () => {
    render(<WindowSignalViewer detail={detail}/>);
    const original = await screen.findByAltText("30 秒全通道原始瀑布图");
    const originalUrl = original.getAttribute("src");
    vi.mocked(api.getBlob).mockImplementation(() => new Promise(() => {}));
    fireEvent.click(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" }));
    fireEvent.click(screen.getByRole("tab", { name: "原始瀑布图" }));
    expect(await screen.findByAltText("30 秒全通道原始瀑布图")).toHaveAttribute("src", originalUrl);
    expect(revokeUrl).not.toHaveBeenCalledWith(originalUrl);
    expect(api.getBlob).toHaveBeenCalledTimes(2);
});

it("retries failed full images and local previews including decode errors", async () => {
    vi.mocked(api.getBlob).mockRejectedValueOnce(new Error("网络断开"));
    render(<WindowSignalViewer detail={detail}/>);
    expect(await screen.findByRole("alert")).toHaveTextContent("网络断开");
    fireEvent.click(screen.getByRole("button", { name: "重试图件" }));
    const full = await screen.findByAltText("30 秒全通道原始瀑布图");
    const badUrl = full.getAttribute("src");
    fireEvent.error(full);
    expect(screen.getByRole("alert")).toHaveTextContent("图像无法解码");
    fireEvent.click(screen.getByRole("button", { name: "重试图件" }));
    await screen.findByAltText("30 秒全通道原始瀑布图");
    expect(revokeUrl).toHaveBeenCalledWith(badUrl);
    vi.mocked(api.getBlob).mockRejectedValueOnce(new Error("局部暂不可用"));
    await expandRegions();
    fireEvent.click(screen.getByRole("button", { name: /^区域 1 ·/ }));
    expect(await screen.findByRole("alert")).toHaveTextContent("局部暂不可用");
    fireEvent.click(screen.getByRole("button", { name: "重试局部预览" }));
    const local = await screen.findByAltText("区域 1 原始瀑布图局部预览");
    fireEvent.error(local);
    fireEvent.click(screen.getByRole("button", { name: "重试局部预览" }));
    await screen.findByAltText("区域 1 原始瀑布图局部预览");
});

it.each([false, true])("enlarges just the clicked full image and restores trapped focus on Escape (filtered=%s)", async filtered => {
    render(<><button>外部按钮</button><WindowSignalViewer detail={detail}/></>);
    await screen.findByAltText("30 秒全通道原始瀑布图");
    if (filtered) {
        const raw = screen.getByRole("tab", { name: "原始瀑布图" });
        raw.focus();
        fireEvent.keyDown(raw, { key: "ArrowRight" });
        expect(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" })).toHaveFocus();
        await screen.findByAltText("30 秒全通道滤波瀑布图");
    }
    const alt = `30 秒全通道${filtered ? "滤波" : "原始"}瀑布图`;
    const open = screen.getByRole("button", { name: `放大${alt}` });
    expect(screen.queryByRole("button", { name: "全屏查看" })).not.toBeInTheDocument();
    open.focus();
    fireEvent.click(within(open).getByRole("img"));
    const dialog = screen.getByRole("dialog", { name: `${alt}放大` });
    expect(within(dialog).getByRole("img")).toHaveAttribute("src", within(open).getByRole("img").getAttribute("src"));
    expect(within(dialog).queryByRole("tab")).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("region", { name: "疑似信号区域" })).not.toBeInTheDocument();
    const close = within(dialog).getByRole("button", { name: "关闭放大图片" });
    expect(close).toHaveFocus();
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });
    expect(within(dialog).getByRole("link", { name: "下载图件" })).toHaveFocus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(close).toHaveFocus();
    screen.getByRole("button", { name: "外部按钮" }).focus();
    fireEvent.keyDown(document, { key: "Tab" });
    expect(close).toHaveFocus();
    expect(document.body.style.overflow).toBe("hidden");
    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(open).toHaveFocus();
    expect(document.body.style.overflow).toBe("");
});

it("selects numbered regions without enlarging and opens the local image from the keyboard", async () => {
    render(<WindowSignalViewer detail={detail}/>);
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    await screen.findByAltText("30 秒全通道原始瀑布图");
    fireEvent.click(screen.getByRole("button", { name: "查看区域 1" }));
    await screen.findByAltText("区域 1 原始瀑布图局部预览");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    const open = screen.getByRole("button", { name: "放大区域 1 原始瀑布图局部预览" });
    open.focus();
    await userEvent.keyboard("{Enter}");
    const dialog = screen.getByRole("dialog", { name: "区域 1 原始瀑布图局部预览放大" });
    expect(within(dialog).getAllByRole("img")).toHaveLength(1);
    expect(within(dialog).getByRole("img")).toHaveAttribute("src", within(open).getByRole("img").getAttribute("src"));
    expect(within(dialog).getByRole("link", { name: "下载图件" })).toHaveAttribute("download", "waterfall-region-1.png");
    fireEvent.click(within(dialog).getByRole("button", { name: "关闭放大图片" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(open).toHaveFocus();
});

it.each([false, true])("uses clean full and local images while preserving the region list (filtered=%s)", async filtered => {
    render(<WindowSignalViewer detail={detail}/>);
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    await screen.findByAltText("30 秒全通道原始瀑布图");
    if (filtered) {
        fireEvent.click(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" }));
        await screen.findByAltText("30 秒全通道滤波瀑布图");
    }
    fireEvent.click(screen.getByRole("button", { name: "查看区域 1" }));
    const fullAlt = `30 秒全通道${filtered ? "滤波" : "原始"}瀑布图`;
    const localAlt = `区域 1 ${filtered ? "滤波" : "原始"}瀑布图局部预览`;
    const oldLocal = await screen.findByAltText(localAlt);
    const oldUrls = [screen.getByAltText(fullAlt).getAttribute("src"), oldLocal.getAttribute("src")];
    const toggle = screen.getByRole("checkbox", { name: "显示疑似区域" });
    expect(toggle).toBeChecked();
    fireEvent.click(toggle);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "查看区域 1" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^区域 1 ·/ })).toHaveAttribute("aria-pressed", "true");
    await screen.findByAltText(fullAlt);
    await screen.findByAltText(localAlt);
    expect(api.getBlob).toHaveBeenCalledWith(`/api/v1/event-windows/w1/waterfall-preview?analysis_id=a1&filtered=${filtered}`, expect.any(AbortSignal));
    expect(api.getBlob).toHaveBeenCalledWith(`${filtered ? filteredPreviewPath : rawPreviewPath}&show_regions=false`, expect.any(AbortSignal));
    for (const url of oldUrls) {
        expect(revokeUrl).not.toHaveBeenCalledWith(url);
        expect(screen.getAllByRole("img").every(image => image.getAttribute("src") !== url)).toBe(true);
    }
    fireEvent.click(screen.getByRole("button", { name: `放大${fullAlt}` }));
    const dialog = screen.getByRole("dialog");
    const dialogToggle = within(dialog).getByRole("checkbox", { name: "显示疑似区域" });
    expect(dialogToggle).not.toBeChecked();
    expect(within(dialog).queryByRole("button", { name: "查看区域 1" })).not.toBeInTheDocument();
    fireEvent.click(dialogToggle);
    expect(toggle).toBeChecked();
    await within(dialog).findByAltText(fullAlt);
    expect(within(dialog).getByRole("button", { name: "查看区域 1" })).toBeInTheDocument();
    expect(within(dialog).getByAltText(fullAlt)).toHaveAttribute("src", oldUrls[0]);
    expect(screen.getByAltText(localAlt)).toHaveAttribute("src", oldUrls[1]);
    fireEvent.click(within(dialog).getByRole("button", { name: "关闭放大图片" }));
    expect(screen.getByRole("button", { name: `放大${fullAlt}` })).toHaveFocus();
});

it("discards late clean images when the shared switch returns to marked images", async () => {
    render(<WindowSignalViewer detail={detail}/>);
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    const full = await screen.findByAltText("30 秒全通道原始瀑布图");
    const oldUrl = full.getAttribute("src");
    fireEvent.click(screen.getByRole("button", { name: "放大30 秒全通道原始瀑布图" }));
    let finishClean: (blob: Blob) => void = () => {};
    vi.mocked(api.getBlob).mockImplementation(path => path.includes("waterfall-preview") ? new Promise(resolve => { finishClean = resolve; }) : Promise.resolve(new Blob(["new"])));
    const dialog = screen.getByRole("dialog");
    const toggle = within(dialog).getByRole("checkbox", { name: "显示疑似区域" });
    fireEvent.click(toggle);
    const cleanSignal = vi.mocked(api.getBlob).mock.calls.at(-1)![1]!;
    expect(within(dialog).queryByRole("img")).not.toBeInTheDocument();
    fireEvent.click(toggle);
    const next = await within(dialog).findByRole("img");
    expect(next).toHaveAttribute("src", oldUrl);
    expect(cleanSignal.aborted).toBe(true);
    const newUrl = next.getAttribute("src");
    await act(async () => { finishClean(new Blob(["late clean"])); });
    expect(next).toHaveAttribute("src", newUrl);
    expect(createUrl).toHaveBeenCalledTimes(1);
});

it.each([undefined, false])("reuses raw/filter/raw images until the viewer exits (baked=%s)", async baked => {
    const { unmount } = render(<WindowSignalViewer detail={{ ...detail, waterfall_regions_baked: baked }}/>);
    const rawUrl = (await screen.findByAltText("30 秒全通道原始瀑布图")).getAttribute("src");
    fireEvent.click(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" }));
    const filteredUrl = (await screen.findByAltText("30 秒全通道滤波瀑布图")).getAttribute("src");
    fireEvent.click(screen.getByRole("tab", { name: "原始瀑布图" }));
    expect(await screen.findByAltText("30 秒全通道原始瀑布图")).toHaveAttribute("src", rawUrl);
    expect(api.getBlob).toHaveBeenCalledTimes(2);
    expect(revokeUrl).not.toHaveBeenCalled();
    unmount();
    expect(revokeUrl).toHaveBeenCalledWith(rawUrl);
    expect(revokeUrl).toHaveBeenCalledWith(filteredUrl);
});

it.each([undefined, false])("reuses region A/B/A images while keeping the full image available (baked=%s)", async baked => {
    render(<WindowSignalViewer detail={{ ...detail, waterfall_regions_baked: baked }}/>);
    const fullUrl = (await screen.findByAltText("30 秒全通道原始瀑布图")).getAttribute("src");
    await expandRegions();
    fireEvent.click(screen.getByRole("button", { name: /^区域 1 ·/ }));
    const firstUrl = (await screen.findByAltText("区域 1 原始瀑布图局部预览")).getAttribute("src");
    await expandRegions();
    fireEvent.click(screen.getByRole("button", { name: /^区域 2 ·/ }));
    await screen.findByAltText("区域 2 原始瀑布图局部预览");
    await expandRegions();
    fireEvent.click(screen.getByRole("button", { name: /^区域 1 ·/ }));
    expect(await screen.findByAltText("区域 1 原始瀑布图局部预览")).toHaveAttribute("src", firstUrl);
    expect(screen.getByAltText("30 秒全通道原始瀑布图")).toHaveAttribute("src", fullUrl);
    expect(api.getBlob).toHaveBeenCalledTimes(3);
});

it.each([undefined, false])("evicts the oldest unused image before retaining a fifth image (baked=%s)", async baked => {
    const regions = [1, 2, 3, 4].map(id => ({ ...detail.detection_regions![0], id }));
    render(<WindowSignalViewer detail={{ ...detail, detection_regions: regions, waterfall_regions_baked: baked }}/>);
    const fullUrl = (await screen.findByAltText("30 秒全通道原始瀑布图")).getAttribute("src");
    let firstUrl = "";
    for (const id of [1, 2, 3, 4]) {
        await expandRegions();
        fireEvent.click(screen.getByRole("button", { name: new RegExp(`^区域 ${id} ·`) }));
        const image = await screen.findByAltText(`区域 ${id} 原始瀑布图局部预览`);
        if (id === 1) firstUrl = image.getAttribute("src")!;
    }
    expect(revokeUrl).toHaveBeenCalledWith(firstUrl);
    expect(revokeUrl).not.toHaveBeenCalledWith(fullUrl);
    expect(createUrl.mock.calls.length - revokeUrl.mock.calls.length).toBe(4);
    await expandRegions();
    fireEvent.click(screen.getByRole("button", { name: /^区域 1 ·/ }));
    expect(await screen.findByAltText("区域 1 原始瀑布图局部预览")).not.toHaveAttribute("src", firstUrl);
    expect(api.getBlob).toHaveBeenCalledTimes(6);
});

it.each([undefined, false])("evicts unused images when their combined blobs would exceed thirty-two MiB (baked=%s)", async baked => {
    vi.mocked(api.getBlob).mockResolvedValueOnce(new Blob([new Uint8Array(8 * 1024 * 1024)], { type: "image/png" }))
        .mockResolvedValue(new Blob([new Uint8Array(12 * 1024 * 1024)], { type: "image/png" }));
    const regions = [1, 2, 3].map(id => ({ ...detail.detection_regions![0], id }));
    render(<WindowSignalViewer detail={{ ...detail, detection_regions: regions, waterfall_regions_baked: baked }}/>);
    const fullUrl = (await screen.findByAltText("30 秒全通道原始瀑布图")).getAttribute("src");
    await expandRegions();
    fireEvent.click(screen.getByRole("button", { name: /^区域 1 ·/ }));
    const firstUrl = (await screen.findByAltText("区域 1 原始瀑布图局部预览")).getAttribute("src");
    await expandRegions();
    fireEvent.click(screen.getByRole("button", { name: /^区域 2 ·/ }));
    await screen.findByAltText("区域 2 原始瀑布图局部预览");
    expect(revokeUrl).not.toHaveBeenCalledWith(firstUrl);
    expect(createUrl.mock.calls.length - revokeUrl.mock.calls.length).toBe(3);
    await expandRegions();
    fireEvent.click(screen.getByRole("button", { name: /^区域 3 ·/ }));
    await screen.findByAltText("区域 3 原始瀑布图局部预览");
    expect(revokeUrl).toHaveBeenCalledWith(firstUrl);
    expect(revokeUrl).not.toHaveBeenCalledWith(fullUrl);
    expect(createUrl.mock.calls.length - revokeUrl.mock.calls.length).toBe(3);
    await expandRegions();
    fireEvent.click(screen.getByRole("button", { name: /^区域 1 ·/ }));
    expect(await screen.findByAltText("区域 1 原始瀑布图局部预览")).not.toHaveAttribute("src", firstUrl);
    expect(api.getBlob).toHaveBeenCalledTimes(5);
});

it.each([undefined, false])("shows an oversized image but releases it on switching rather than retaining it (baked=%s)", async baked => {
    vi.mocked(api.getBlob).mockResolvedValueOnce(new Blob([new Uint8Array(33 * 1024 * 1024)], { type: "image/png" }));
    render(<WindowSignalViewer detail={{ ...detail, waterfall_regions_baked: baked }}/>);
    const oversizedUrl = (await screen.findByAltText("30 秒全通道原始瀑布图")).getAttribute("src");
    fireEvent.click(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" }));
    await screen.findByAltText("30 秒全通道滤波瀑布图");
    expect(revokeUrl).toHaveBeenCalledWith(oversizedUrl);
    fireEvent.click(screen.getByRole("tab", { name: "原始瀑布图" }));
    expect(await screen.findByAltText("30 秒全通道原始瀑布图")).not.toHaveAttribute("src", oversizedUrl);
    expect(api.getBlob).toHaveBeenCalledTimes(3);
});

it.each([undefined, false])("releases all cached images on input identity changes even if the URLs remain the same (baked=%s)", async baked => {
    const { rerender } = render(<WindowSignalViewer detail={{ ...detail, waterfall_regions_baked: baked }}/>);
    const rawUrl = (await screen.findByAltText("30 秒全通道原始瀑布图")).getAttribute("src");
    fireEvent.click(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" }));
    const filteredUrl = (await screen.findByAltText("30 秒全通道滤波瀑布图")).getAttribute("src");
    rerender(<WindowSignalViewer detail={{ ...detail, input_sha256: "changed-input", waterfall_regions_baked: baked }}/>);
    expect(await screen.findByAltText("30 秒全通道原始瀑布图")).not.toHaveAttribute("src", rawUrl);
    expect(revokeUrl).toHaveBeenCalledWith(rawUrl);
    expect(revokeUrl).toHaveBeenCalledWith(filteredUrl);
    expect(api.getBlob).toHaveBeenCalledTimes(3);
});

it("defaults to clean images and keeps regions hidden when changing waterfall tabs", async () => {
    render(<WindowSignalViewer detail={detail}/>);
    await screen.findByAltText("30 秒全通道原始瀑布图");
    expect(screen.getByRole("checkbox", { name: "显示疑似区域" })).not.toBeChecked();
    expect(api.getBlob).toHaveBeenCalledWith("/api/v1/event-windows/w1/waterfall-preview?analysis_id=a1&filtered=false", expect.any(AbortSignal));
    expect(screen.queryByRole("button", { name: "查看区域 1" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" }));
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    await screen.findByAltText("30 秒全通道滤波瀑布图");
    expect(api.getBlob).toHaveBeenLastCalledWith("/api/v1/event-windows/w1/waterfall-preview?analysis_id=a1&filtered=true", expect.any(AbortSignal));
    expect(api.getBlob).not.toHaveBeenCalledWith(filteredPath, expect.anything());
    expect(screen.queryByRole("button", { name: "查看区域 1" })).not.toBeInTheDocument();
});

it.each([undefined, 2])("does not redraw images without applicable regions (version=%s)", async visualizationVersion => {
    render(<WindowSignalViewer detail={{ ...detail, visualization_version: visualizationVersion, detection_regions: [] }}/>);
    const full = await screen.findByAltText("30 秒全通道原始瀑布图");
    const url = full.getAttribute("src");
    fireEvent.click(screen.getByRole("checkbox", { name: "显示疑似区域" }));
    expect(full).toHaveAttribute("src", url);
    expect(api.getBlob).toHaveBeenCalledTimes(1);
});

it("encodes preview ownership and preserves existing marked-region URLs", () => {
    expect(windowApi.waterfallPreviewUrl("w/1", "a &1", true)).toBe("/api/v1/event-windows/w%2F1/waterfall-preview?analysis_id=a+%261&filtered=true");
    expect(windowApi.regionPreviewUrl("w1", "a1", 1, false)).toBe(rawPreviewPath);
    expect(windowApi.regionPreviewUrl("w1", "a1", 1, false, true)).toBe(rawPreviewPath);
    expect(windowApi.regionPreviewUrl("w1", "a1", 1, false, false)).toBe(`${rawPreviewPath}&show_regions=false`);
});

it("keeps legacy raw images viewable without inventing region coordinates", async () => {
    render(<WindowSignalViewer detail={{ ...detail, visualization_version: undefined, detection_regions: undefined, plot_geometry: undefined, artifacts: [{ kind: "waterfall", url: rawPath }, { kind: "waveform_plot", url: "/old-waveform" }] }}/>);
    await screen.findByAltText("30 秒全通道原始瀑布图");
    expect(screen.getByText(/待补图/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /查看区域/ })).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "0.5–40 Hz 滤波瀑布图" })).toBeDisabled();
    expect(api.getBlob).toHaveBeenCalledTimes(1);
});
