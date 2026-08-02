import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, beforeEach } from "vitest";
import { renderApp } from "../test/render";

const server = setupServer();

beforeAll(() => server.listen());
beforeEach(() => {
  server.resetHandlers();
  server.use(
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
  );
});
afterAll(() => server.close());

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

const CANDIDATE = {
  id: "model-1",
  site_id: "site-1",
  version: "1",
  status: "candidate",
  dataset_sha256: "a".repeat(64),
  bundle_sha256: "b".repeat(64),
  metrics: {
    macro_f1: 0.82,
    rule_baseline_macro_f1: 0.61,
    recall_by_label: { earthquake: 0.8, blast: 0.85, traffic: 0.83, noise: 0.8 },
    precision_by_label: { earthquake: 0.82, blast: 0.84, traffic: 0.8, noise: 0.83 },
    reviewed_count_by_label: { earthquake: 25, blast: 25, traffic: 25, noise: 25 },
    confusion_matrix: [
      [20, 2, 2, 1],
      [1, 21, 2, 1],
      [2, 1, 21, 1],
      [1, 2, 2, 20],
    ],
    feature_names: ["rms", "duration_s"],
  },
  created_at: "2026-07-26T00:00:00Z",
  published_at: null,
};

it("requires explicit confirmation before model publication", async () => {
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([SITE])),
    http.get("/api/v1/sites/site-1/models", () =>
      HttpResponse.json([CANDIDATE]),
    ),
  );
  renderApp("/models");

  await userEvent.click(
    await screen.findByRole("button", { name: "发布候选模型" }),
  );

  expect(
    screen.getByRole("dialog", { name: "确认发布模型" }),
  ).toBeVisible();
});

it("publishes after confirmation and shows the validation report", async () => {
  let published = false;
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([SITE])),
    http.get("/api/v1/sites/site-1/models", () =>
      HttpResponse.json([
        { ...CANDIDATE, status: published ? "active" : "candidate" },
      ]),
    ),
    http.post("/api/v1/models/model-1/publish", () => {
      published = true;
      return HttpResponse.json({ ...CANDIDATE, status: "active" });
    }),
  );
  renderApp("/models");

  await userEvent.click(
    await screen.findByRole("button", { name: "查看验证报告" }),
  );
  expect(await screen.findByText("宏平均 F1：0.820", { exact: false }))
    .toBeVisible();
  expect(screen.getByLabelText("混淆矩阵")).toBeVisible();

  await userEvent.click(
    screen.getByRole("button", { name: "发布候选模型" }),
  );
  await userEvent.click(
    screen.getByRole("button", { name: "确认发布" }),
  );

  expect(await screen.findByText("已发布")).toBeVisible();
  expect(published).toBe(true);
});
