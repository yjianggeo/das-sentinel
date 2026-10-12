import { act, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "../api";
import WindowTraining from "./WindowTraining";

vi.mock("../api", () => ({ api: { get: vi.fn(), post: vi.fn() } }));
const readiness = { window_count: 20, reviewed_windows: 2, eligible_windows: 2, ready: false, reasons: ["需要更多复核"], classes: {} };
beforeEach(() => { vi.clearAllMocks(); vi.useFakeTimers(); });
afterEach(() => { vi.useRealTimers(); });

it("does not repeatedly export waveform readiness while no training runs", async () => {
  vi.mocked(api.get).mockImplementation(async (url) => url.includes("readiness") ? readiness : []);
  await act(async () => { render(<WindowTraining siteId="site" />); });
  expect(screen.getByText(/可训练 2/)).toBeVisible();
  await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
  expect(api.get).toHaveBeenCalledTimes(2);
});

it("polls lightweight active jobs and refreshes readiness only on completion", async () => {
  let finished = false;
  vi.mocked(api.get).mockImplementation(async (url) => url.includes("readiness") ? readiness : [{ id: "train-1", status: finished ? "succeeded" : "running", error: null, metrics: null }]);
  await act(async () => { render(<WindowTraining siteId="site" />); });
  await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
  expect(vi.mocked(api.get).mock.calls.filter(([url]) => url.includes("readiness"))).toHaveLength(1);
  finished = true;
  await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
  expect(screen.getByText("succeeded")).toBeVisible();
  expect(vi.mocked(api.get).mock.calls.filter(([url]) => url.includes("readiness"))).toHaveLength(2);
  const requests = vi.mocked(api.get).mock.calls.length;
  await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
  expect(api.get).toHaveBeenCalledTimes(requests);
});

it("shows the binary training classes and can train without separate blast or traffic labels", async () => {
  vi.mocked(api.get).mockImplementation(async (url) => url.includes("readiness") ? {
    ...readiness, ready: true, reasons: [],
    classes: { earthquake: { present: 107, absent: 2153, unknown: 10 } },
  } : []);
  vi.mocked(api.post).mockResolvedValue({ id: "binary-1", status: "queued", error: null, metrics: null });
  await act(async () => { render(<WindowTraining siteId="site" />); });
  expect(screen.getByRole("region", { name: "整窗二分类训练" })).toBeVisible();
  expect(screen.getByText(/地震（含爆破）107/)).toBeVisible();
  expect(screen.getByText(/噪声（含车辆）2153/)).toBeVisible();
  const button = screen.getByRole("button", { name: "训练整窗二分类模型" });
  expect(button).toBeEnabled();
  await act(async () => { button.click(); });
  expect(screen.getByText("queued")).toBeVisible();
  expect(api.post).toHaveBeenCalledWith("/api/v1/window-training/jobs", { site_id: "site" });
});

it("shows the unified physical training grid and missing-channel treatment", async () => {
  vi.mocked(api.get).mockImplementation(async (url) => url.includes("readiness") ? {
    ...readiness, ready: true, reasons: [],
    training_grid: { channels: 834, spacing_m: 4.089979648590088, start_m: 0, stop_m: 3406.953,
      source_grid_count: 3 },
  } : []);
  await act(async () => { render(<WindowTraining siteId="site" />); });
  expect(screen.getByText(/834 道 × 3000 点/)).toBeVisible();
  expect(screen.getByText(/4.089980 m/)).toBeVisible();
  expect(screen.getByText(/缺失通道以无效掩码排除/)).toBeVisible();
});
