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

const PROFILE = {
  id: "profile-1",
  version: 1,
  payload: {},
  is_current: true,
  created_at: "2026-07-26T00:00:00Z",
};

function site(name: string) {
  return {
    id: `id-${name}`,
    name,
    data_path: "/data/jjk",
    timezone: "Asia/Shanghai",
    cable_type: "urban",
    enabled: true,
    created_at: "2026-07-26T00:00:00Z",
    current_profile: PROFILE,
  };
}

it("creates a site with explicit timezone and cable type", async () => {
  let created = false;
  server.use(
    http.get("/api/v1/sites", () =>
      HttpResponse.json(created ? [site("JJK-01")] : []),
    ),
    http.post("/api/v1/sites", async ({ request }) => {
      const body = (await request.json()) as Record<string, unknown>;
      expect(body).toMatchObject({
        name: "JJK-01",
        data_path: "/data/jjk",
        timezone: "Asia/Shanghai",
        cable_type: "urban",
      });
      created = true;
      return HttpResponse.json(site("JJK-01"), { status: 201 });
    }),
  );
  renderApp("/sites");

  await userEvent.click(
    await screen.findByRole("button", { name: "新建站点" }),
  );
  await userEvent.type(screen.getByLabelText("名称"), "JJK-01");
  await userEvent.type(screen.getByLabelText("数据路径"), "/data/jjk");
  await userEvent.selectOptions(screen.getByLabelText("光缆类型"), "urban");
  await userEvent.clear(screen.getByLabelText("IANA 时区"));
  await userEvent.type(screen.getByLabelText("IANA 时区"), "Asia/Shanghai");
  await userEvent.click(screen.getByRole("button", { name: "保存" }));

  expect(await screen.findByText("JJK-01")).toBeVisible();
});

it("toggles a site enabled state", async () => {
  let enabled = true;
  server.use(
    http.get("/api/v1/sites", () =>
      HttpResponse.json([{ ...site("JJK-01"), enabled }]),
    ),
    http.patch("/api/v1/sites/id-JJK-01", () => {
      enabled = false;
      return HttpResponse.json({ ...site("JJK-01"), enabled });
    }),
  );
  renderApp("/sites");

  await userEvent.click(await screen.findByRole("button", { name: "停用" }));

  expect(await screen.findByText("停用")).toBeVisible();
});
