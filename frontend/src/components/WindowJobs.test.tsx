import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { api } from "../api";
import WindowJobs from "./WindowJobs";
vi.mock("../api", () => ({ api: { get: vi.fn(), post: vi.fn() } }));
let response: { items: Record<string, unknown>[]; total: number; limit: number; offset: number };
const jobsCalls = () => vi.mocked(api.get).mock.calls.filter(([url]) => url === "/api/v1/window-jobs");
beforeEach(() => {
    vi.clearAllMocks();
    response = { items: [{ id: "a1", window_id: "w1", site_name: "站点一", site_timezone: "Asia/Shanghai", start_utc: "2024-01-01T00:00:00Z", status: "failed", attempt_count: 1, error_message: "缺数据" }], total: 51, limit: 50, offset: 0 };
    vi.mocked(api.get).mockImplementation(async url => url === "/api/v1/sites" ? [{ id: "s1", name: "站点一" }] : response);
});
it("paginates and filters jobs and exposes retry errors", async () => {
    render(<WindowJobs />);
    await screen.findByText("缺数据");
    expect(screen.queryByLabelText("任务窗口长度")).not.toBeInTheDocument();
    expect(api.get).toHaveBeenCalledWith("/api/v1/window-jobs", expect.objectContaining({ window_seconds: "30" }), expect.any(AbortSignal));
    expect(screen.getByText("2024-01-01 08:00:00")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "下一页整窗任务" }));
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/window-jobs", expect.objectContaining({ offset: "50", limit: "50" }), expect.any(AbortSignal)));
    fireEvent.change(screen.getByLabelText("整窗任务站点"), { target: { value: "s1" } });
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/window-jobs", expect.objectContaining({ site_id: "s1", offset: "0" }), expect.any(AbortSignal)));
    fireEvent.change(screen.getByLabelText("整窗任务状态"), { target: { value: "failed" } });
    await waitFor(() => expect(api.get).toHaveBeenCalledWith("/api/v1/window-jobs", expect.objectContaining({ status: "failed", offset: "0" }), expect.any(AbortSignal)));
    vi.mocked(api.post).mockRejectedValueOnce(new Error("重试被拒绝"));
    fireEvent.click(await screen.findByRole("button", { name: "重试" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("重试被拒绝");
});
it("cancels active jobs and refreshes their status", async () => {
    response = { items: [{ id: "a2", window_id: "w2", site_name: "站点二", site_timezone: "Asia/Shanghai", start_utc: "2024-01-01T00:01:00Z", status: "running", attempt_count: 1 }], total: 1, limit: 50, offset: 0 };
    vi.mocked(api.post).mockResolvedValue({});
    render(<WindowJobs />);
    fireEvent.click(await screen.findByRole("button", { name: "取消" }));
    await waitFor(() => expect(api.post).toHaveBeenCalledWith("/api/v1/window-jobs/a2/cancel"));
    await waitFor(() => expect(jobsCalls()).toHaveLength(2));
});


it("never offers retry for a cancelled job",async()=>{
 response={items:[{id:"a2",window_id:"w2",site_name:"站点二",site_timezone:"Asia/Shanghai",start_utc:"2024-01-01T00:00:00Z",status:"cancelled",attempt_count:1}],total:1,limit:50,offset:0};
 render(<WindowJobs/>);await screen.findByText("已取消");expect(screen.queryByRole("button",{name:"重试"})).not.toBeInTheDocument();
 expect(jobsCalls()[0][1]).not.toHaveProperty("status");
});
it("polls active windows without clearing rows and stops once complete",async()=>{
 const job={id:"a2",window_id:"w2",site_name:"站点二",site_timezone:"Asia/Shanghai",start_utc:"2024-01-01T00:00:00Z",status:"running",attempt_count:1};
 vi.useFakeTimers();response={items:[job],total:1,limit:50,offset:0};await act(async()=>{render(<WindowJobs/>);});expect(screen.getByText("运行中")).toBeInTheDocument();
 let finish:(value:unknown)=>void=()=>{};vi.mocked(api.get).mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));
 try {
  await act(async()=>{await vi.advanceTimersByTimeAsync(5000);});expect(jobsCalls()).toHaveLength(2);expect(screen.getByText("运行中")).toBeInTheDocument();
  await act(async()=>{finish({items:[{...job,status:"succeeded"}],total:1,limit:30,offset:0});});expect(screen.getByText("成功")).toBeInTheDocument();
  await act(async()=>{await vi.advanceTimersByTimeAsync(10000);});expect(jobsCalls()).toHaveLength(2);
 } finally {vi.useRealTimers();}
});

it("uses truthful progress and explains coverage failures without an unusable retry", async () => {
    const base = response.items[0];
    response.items = [
        { ...base, id: "running", status: "running", error_message: null },
        { ...base, id: "done", status: "succeeded", error_message: null },
        { ...base, id: "gap", status: "failed", error_code: "coverage_incomplete" },
    ];
    render(<WindowJobs />);
    const running = await screen.findByRole("progressbar", { name: "任务 running 进度" });
    expect(running).not.toHaveAttribute("aria-valuenow");
    expect(running).toHaveAttribute("aria-valuetext", "分析及图件生成中");
    expect(screen.getByRole("progressbar", { name: "任务 done 进度" })).toHaveAttribute("aria-valuenow", "100");
    const failure = screen.getByText("gap").closest("tr")!;
    expect(within(failure).queryByRole("button", { name: "重试" })).not.toBeInTheDocument();
    expect(within(failure).getByText(/补齐数据后重新扫描/)).toBeInTheDocument();
});
