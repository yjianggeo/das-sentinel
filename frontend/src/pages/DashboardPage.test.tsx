import { screen, within } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, beforeEach, expect, it } from "vitest";
import { renderApp } from "../test/render";

const server = setupServer();

beforeAll(() => server.listen());
beforeEach(() => server.resetHandlers());
afterAll(() => server.close());

it("renders dashboard counts from one authoritative request", async () => {
  let dashboardRequests = 0;
  server.use(
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
    http.get("/api/v1/dashboard", () => {
      dashboardRequests += 1;
      return HttpResponse.json({
        window_status_counts: { planned: 14, queued: 11, running: 7, succeeded: 19, failed: 5 },
        model_scan_status_counts: { queued: 2, running: 3, succeeded: 4, failed: 6, paused: 12, completed_with_errors: 13 },
        review_counts: { earthquake: 8, noise: 9, pending: 10 },
        job_counts: { queued: 3, running: 2, failed: 1 },
        today_event_count: 145,
        machine_label_counts: { traffic: 120, earthquake: 25 },
      });
    }),
    http.get("/api/v1/jobs", () => {
      throw new Error("dashboard must not fetch jobs");
    }),
    http.get("/api/v1/events", () => {
      throw new Error("dashboard must not fetch events");
    }),
  );

  renderApp("/");

  const windows = await screen.findByRole("region", { name: "30 秒窗口任务" });
  expect(windows).toHaveTextContent("待规划：14");
  expect(windows).toHaveTextContent("排队：11");
  expect(windows).toHaveTextContent("运行中：7");
  expect(windows).toHaveTextContent("成功：19");
  expect(windows).toHaveTextContent("失败：5");
  const scans = screen.getByRole("region", { name: "模型扫描批次" });
  expect(scans).toHaveTextContent("运行中：3");
  expect(scans).toHaveTextContent("成功：4");
  expect(scans).toHaveTextContent("已暂停：12");
  expect(scans).toHaveTextContent("已结束，部分数据无法判断：13");
  const reviews = screen.getByRole("region", { name: "人工复核" });
  expect(within(reviews).getByText("地震类：8")).toBeVisible();
  expect(within(reviews).getByText("噪声类：9")).toBeVisible();
  expect(within(reviews).getByText("待复核：10")).toBeVisible();
  expect(screen.queryByText("145")).not.toBeInTheDocument();
  expect(screen.queryByText(/车辆：120/)).not.toBeInTheDocument();
  expect(dashboardRequests).toBe(1);
});
