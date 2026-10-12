import { screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { transferableAbortController } from "node:util";
import { afterAll, beforeAll, beforeEach, expect, it } from "vitest";
import { renderApp } from "../test/render";

const server = setupServer();
const domAbortController = globalThis.AbortController;
let legacyRequests = 0;
let windowTrainingRequests = 0;

const SITE = {
  id: "site-1",
  name: "JJK-01",
  data_path: "/data/jjk",
  timezone: "Asia/Shanghai",
  cable_type: "downhole",
  enabled: true,
  created_at: "2026-07-26T00:00:00Z",
  current_profile: {
    id: "profile-1",
    version: 1,
    payload: {},
    is_current: true,
    created_at: "2026-07-26T00:00:00Z",
  },
};

beforeAll(() => {
  globalThis.AbortController = transferableAbortController().constructor as typeof AbortController;
  server.listen();
});
beforeEach(() => {
  legacyRequests = 0;
  windowTrainingRequests = 0;
  server.resetHandlers();
  server.use(
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
    http.get("/api/v1/sites", () => HttpResponse.json([SITE])),
    http.get("/api/v1/window-training/readiness", () => {
      windowTrainingRequests += 1;
      return HttpResponse.json({
        window_count: 3522,
        reviewed_windows: 2270,
        eligible_windows: 2260,
        ready: false,
        reasons: ["需要更多独立验证分组"],
        classes: { earthquake: { present: 107, absent: 2153, unknown: 1262 } },
      });
    }),
    http.get("/api/v1/window-training/jobs", () => HttpResponse.json([])),
    http.get("/api/v1/window-models/active", () => HttpResponse.json(null)),
    http.get("/api/v1/sites/:siteId/models", () => {
      legacyRequests += 1;
      return HttpResponse.json([]);
    }),
    http.get("/api/v1/sites/:siteId/training-readiness", () => {
      legacyRequests += 1;
      return HttpResponse.json({});
    }),
  );
});
afterAll(() => {
  server.close();
  globalThis.AbortController = domAbortController;
});

it("shows only whole-window training and does not load legacy ROI models", async () => {
  renderApp("/models");

  await screen.findByRole("option", { name: "JJK-01" });
  expect(screen.getByRole("combobox", { name: "站点" })).toHaveValue("site-1");
  expect(screen.getByRole("region", { name: "整窗二分类训练" })).toBeVisible();
  expect(await screen.findByText(/窗口记录 3522/)).toBeVisible();
  expect(screen.queryByText("历史局部 ROI 模型（旧随机森林）")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /发布候选模型|回滚到上一模型|训练历史 ROI 模型/ })).not.toBeInTheDocument();
  await waitFor(() => expect(windowTrainingRequests).toBe(1));
  expect(legacyRequests).toBe(0);
});
