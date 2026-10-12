import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { api } from "../api";
import ModelScans from "./ModelScans";
vi.mock("../api", () => ({ api: { get: vi.fn(), post: vi.fn() } }));
let scan: Record<string, unknown>;
beforeEach(() => {
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
    vi.clearAllMocks();
    scan = { id: "batch", status: "running", model: { id: "frozen", name: "300 m 模型", threshold: .04 },
        directories: ["jjk2023"], counts: { sources: 20, planned: 10, judged: 8, selected: 2,
            pending_artifacts: 2, ready: 0, negative: 6, unjudgeable: 0, source_errors: 0 },
        discovery_complete: false, total: null, estimate_status: "discovering", remaining_seconds: null,
        estimated_finish_at: null, errors: [] };
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [] :
        { items: [scan], total: 1, limit: 10, offset: 0 });
    vi.mocked(api.post).mockResolvedValue({});
});
afterEach(() => { vi.useRealTimers(); });
it("distinguishes discovery and pending artifacts and pauses the batch", async () => {
    render(<MemoryRouter><ModelScans /></MemoryRouter>);
    expect(await screen.findByText("正在统计数据范围")).toBeInTheDocument();
    expect(screen.getByText(/待出图 2/)).toBeInTheDocument();
    expect(screen.queryByText("全部完成")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "暂停" }));
    await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/v1/model-scans/batch/pause"));
    expect(screen.getByRole("link", { name: "查看入选窗口" })).toHaveAttribute("href", "/events?range=model_recommended&scan_id=batch");
});
it("never resumes cancelled scans and displays a real estimate when available", async () => {
    Object.assign(scan, { status: "cancelled", discovery_complete: true, estimate_status: "cancelled" });
    render(<MemoryRouter><ModelScans /></MemoryRouter>);
    const section = await screen.findByRole("region", { name: "模型批次 batch" });
    expect(within(section).queryByRole("button", { name: "继续" })).not.toBeInTheDocument();
    expect(within(section).queryByRole("button", { name: "重试" })).not.toBeInTheDocument();
});

it("discovers new scans at sixty seconds while idle and then follows active scans at ten seconds", async () => {
    vi.useFakeTimers();
    Object.assign(scan, { status: "succeeded", estimate_status: "complete" });
    await act(async () => { render(<MemoryRouter><ModelScans /></MemoryRouter>); });
    Object.assign(scan, { status: "running", estimate_status: "discovering" });
    await act(async () => { await vi.advanceTimersByTimeAsync(59000); });
    expect(screen.getByText("全部完成")).toBeInTheDocument();
    expect(vi.mocked(api.get).mock.calls.filter(([url]) => url === "/api/v1/model-scans")).toHaveLength(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByText("扫描中")).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
    expect(vi.mocked(api.get).mock.calls.filter(([url]) => url === "/api/v1/model-scans")).toHaveLength(3);
});

it("slows hidden scans and refreshes immediately on visibility and after an action", async () => {
    vi.useFakeTimers();
    await act(async () => { render(<MemoryRouter><ModelScans /></MemoryRouter>); });
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "hidden" });
    await act(async () => { document.dispatchEvent(new Event("visibilitychange")); await vi.advanceTimersByTimeAsync(59000); });
    expect(vi.mocked(api.get).mock.calls.filter(([url]) => url === "/api/v1/model-scans")).toHaveLength(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(vi.mocked(api.get).mock.calls.filter(([url]) => url === "/api/v1/model-scans")).toHaveLength(2);
    Object.assign(scan, { status: "paused", estimate_status: "paused" });
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
    await act(async () => { document.dispatchEvent(new Event("visibilitychange")); });
    expect(screen.getByText("已暂停")).toBeInTheDocument();
    Object.assign(scan, { status: "running", estimate_status: "discovering" });
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: "继续" })); });
    expect(screen.getByText("扫描中")).toBeInTheDocument();
    expect(vi.mocked(api.get).mock.calls.filter(([url]) => url === "/api/v1/model-scans")).toHaveLength(4);
});

it("retries a failed initial scan read without requiring the refresh button", async () => {
    vi.useFakeTimers();
    vi.mocked(api.get).mockImplementation(async url => {
        if (url === "/api/v1/sites") return [];
        throw new Error("scan-offline");
    });
    await act(async () => { render(<MemoryRouter><ModelScans /></MemoryRouter>); });
    expect(screen.getByRole("alert")).toHaveTextContent("scan-offline");
    vi.mocked(api.get).mockResolvedValue({ items: [scan], total: 1, limit: 10, offset: 0 });
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
    expect(screen.getByText("扫描中")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("aborts a stalled scan read and resumes polling after the idle retry interval", async () => {
    vi.useFakeTimers();
    vi.mocked(api.get).mockImplementation(async (url, _params, signal) => url === "/api/v1/sites" ? [] : new Promise((_resolve, reject) => {
        signal?.addEventListener("abort", () => reject(new Error("scan-timeout")));
    }));
    await act(async () => { render(<MemoryRouter><ModelScans /></MemoryRouter>); });
    await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
    expect(screen.getByRole("alert")).toHaveTextContent("scan-timeout");
    vi.mocked(api.get).mockResolvedValue({ items: [scan], total: 1, limit: 10, offset: 0 });
    await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
    expect(screen.getByText("扫描中")).toBeInTheDocument();
});
