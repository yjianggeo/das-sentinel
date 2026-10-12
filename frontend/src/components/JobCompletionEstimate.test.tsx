import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "../api";
import JobCompletionEstimate from "./JobCompletionEstimate";

vi.mock("../api", () => ({ api: { get: vi.fn() } }));
const running = {
    as_of: "2026-09-18T06:00:00Z", status: "running", queued: 9, running: 1,
    remaining_seconds: 900, estimated_finish_at: "2026-09-18T06:15:00Z", queues: [],
};
beforeEach(() => {
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
    vi.useFakeTimers(); vi.clearAllMocks(); vi.mocked(api.get).mockResolvedValue(running);
});
afterEach(() => vi.useRealTimers());

it("describes the estimate as covering every created whole-window artifact task", async () => {
    await act(async () => { render(<JobCompletionEstimate />); });
    expect(screen.getByRole("heading", { name: "整窗图件任务预计完成时间" })).toBeVisible();
    expect(screen.getByText(/仅统计已创建的整窗图件任务；模型扫描的完整进度与预计完成时间见上方独立批次/)).toBeVisible();
    expect(screen.getByText("待扫描任务")).toBeVisible();
    expect(screen.queryByText(/历史文件任务|历史任务/)).not.toBeInTheDocument();
});

it("discovers new artifact jobs at sixty seconds while idle and then follows them at five seconds", async () => {
    vi.mocked(api.get).mockResolvedValue({ ...running, status: "complete", queued: 0, running: 0, remaining_seconds: 0, estimated_finish_at: null });
    await act(async () => { render(<JobCompletionEstimate />); });
    vi.mocked(api.get).mockResolvedValue(running);
    await act(async () => { await vi.advanceTimersByTimeAsync(59000); });
    expect(screen.getByText("0 个")).toBeInTheDocument();
    expect(api.get).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByText("10 个")).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(api.get).toHaveBeenCalledTimes(3);
});

it("slows hidden estimates and immediately refreshes when returning to the page", async () => {
    await act(async () => { render(<JobCompletionEstimate />); });
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "hidden" });
    await act(async () => { document.dispatchEvent(new Event("visibilitychange")); await vi.advanceTimersByTimeAsync(59000); });
    expect(api.get).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(api.get).toHaveBeenCalledTimes(2);
    vi.mocked(api.get).mockResolvedValue({ ...running, queued: 3 });
    Object.defineProperty(document, "visibilityState", { configurable: true, value: "visible" });
    await act(async () => { document.dispatchEvent(new Event("visibilitychange")); });
    expect(screen.getByText("3 个排队 · 1 个运行中")).toBeInTheDocument();
    expect(api.get).toHaveBeenCalledTimes(3);
});

it("keeps retrying an initially unavailable estimate at the idle interval", async () => {
    vi.mocked(api.get).mockRejectedValueOnce(new Error("offline"));
    await act(async () => { render(<JobCompletionEstimate />); });
    expect(screen.getByText("估算更新失败")).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(59000); });
    expect(api.get).toHaveBeenCalledTimes(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByText("约 15 分钟")).toBeInTheDocument();
    expect(api.get).toHaveBeenCalledTimes(2);
});

it("refreshes the estimate immediately after the parent reports a job operation", async () => {
    let rerender: ReturnType<typeof render>["rerender"] = () => {};
    await act(async () => { rerender = render(<JobCompletionEstimate />).rerender; });
    vi.mocked(api.get).mockResolvedValue({ ...running, queued: 2 });
    await act(async () => { rerender(<JobCompletionEstimate revision={1} />); });
    expect(screen.getByText("2 个排队 · 1 个运行中")).toBeInTheDocument();
    expect(api.get).toHaveBeenCalledTimes(2);
});

it("updates the total every five seconds and aborts polling on unmount", async () => {
    let unmount = () => {};
    await act(async () => { unmount = render(<JobCompletionEstimate />).unmount; });
    expect(screen.getByText("约 15 分钟")).toBeInTheDocument();
    vi.mocked(api.get).mockResolvedValue({ ...running, queued: 4, remaining_seconds: 450 });
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(screen.getByText("约 8 分钟")).toBeInTheDocument();
    expect(screen.getByText("4 个排队 · 1 个运行中")).toBeInTheDocument();
    expect(api.get).toHaveBeenCalledTimes(2);
    const signal = vi.mocked(api.get).mock.calls.at(-1)?.[2];
    unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
    expect(api.get).toHaveBeenCalledTimes(2);
});

it.each(["estimating", "waiting_worker", "complete"])("does not display a finish time when %s", async status => {
    vi.mocked(api.get).mockResolvedValue({ ...running, status, remaining_seconds: status === "complete" ? 0 : null, estimated_finish_at: null });
    await act(async () => { render(<JobCompletionEstimate />); });
    expect(screen.queryByText("2026-09-18 14:15")).not.toBeInTheDocument();
    expect(screen.queryByText("约 15 分钟")).not.toBeInTheDocument();
    if (status !== "complete") expect(screen.getByText("暂无法估算")).toBeInTheDocument();
});

it("hides obsolete estimates after a request failure and recovers on the next poll", async () => {
    await act(async () => { render(<JobCompletionEstimate />); });
    expect(screen.getByText("2026-09-18 14:15")).toBeInTheDocument();
    vi.mocked(api.get).mockRejectedValueOnce(new Error("offline"));
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(screen.getByText("估算更新失败")).toBeInTheDocument();
    expect(screen.queryByText("2026-09-18 14:15")).not.toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(screen.getByText("2026-09-18 14:15")).toBeInTheDocument();
    expect(screen.queryByText("估算更新失败")).not.toBeInTheDocument();
});

it("times out a stalled request and keeps retrying without showing an old ETA", async () => {
    await act(async () => { render(<JobCompletionEstimate />); });
    vi.mocked(api.get).mockImplementationOnce((_url, _params, signal) => new Promise((_resolve, reject) => {
        signal?.addEventListener("abort", () => reject(new Error("aborted")));
    }));
    await act(async () => { await vi.advanceTimersByTimeAsync(20000); });
    expect(screen.getByText("估算更新失败")).toBeInTheDocument();
    expect(screen.queryByText("2026-09-18 14:15")).not.toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(api.get).toHaveBeenCalledTimes(3);
    expect(screen.getByText("2026-09-18 14:15")).toBeInTheDocument();
});
