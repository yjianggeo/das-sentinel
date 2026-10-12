import { screen, within } from "@testing-library/react";
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

it("shows saved scan distance boundaries and treats old profiles as unrestricted", async () => {
  const cases = [
    { name: "旧配置", payload: {}, summary: "全距离（不限）" },
    { name: "起点", payload: { scan_distance_start_m: 300, scan_distance_stop_m: null }, summary: "从 300 m 起" },
    { name: "终点", payload: { scan_distance_start_m: null, scan_distance_stop_m: 900 }, summary: "至 900 m" },
    { name: "两端", payload: { scan_distance_start_m: 0, scan_distance_stop_m: 1200 }, summary: "0–1200 m" },
  ];
  server.use(http.get("/api/v1/sites", () => HttpResponse.json(cases.map((item) => ({
    ...site(item.name),
    current_profile: { ...PROFILE, payload: item.payload },
  })))));
  renderApp("/sites");

  for (const item of cases) {
    const row = (await screen.findByText(item.name)).closest("tr")!;
    expect(within(row).getByText(item.summary)).toBeVisible();
    expect(within(row).getByRole("button", { name: "设置扫描距离" })).toBeEnabled();
  }
});

it("saves scan distance as a new profile and clears blank boundaries with null", async () => {
  let current = {
    ...site("JJK-01"),
    current_profile: { ...PROFILE, payload: { scan_distance_start_m: 300 as number | null, scan_distance_stop_m: 1200 as number | null } },
  };
  const patches: unknown[] = [];
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([current])),
    http.patch("/api/v1/sites/id-JJK-01", async ({ request }) => {
      const body = await request.json() as { profile: typeof current.current_profile.payload };
      patches.push(body);
      current = { ...current, current_profile: { ...current.current_profile, version: current.current_profile.version + 1, payload: body.profile } };
      return HttpResponse.json(current);
    }),
  );
  renderApp("/sites");

  await userEvent.click(await screen.findByRole("button", { name: "设置扫描距离" }));
  expect(screen.getByLabelText("起点距离（m）")).toHaveValue("300");
  expect(screen.getByLabelText("终点距离（m）")).toHaveValue("1200");
  await userEvent.clear(screen.getByLabelText("起点距离（m）"));
  await userEvent.type(screen.getByLabelText("起点距离（m）"), "350");
  await userEvent.clear(screen.getByLabelText("终点距离（m）"));
  await userEvent.type(screen.getByLabelText("终点距离（m）"), "1500");
  await userEvent.click(screen.getByRole("button", { name: "保存扫描距离" }));

  expect(await screen.findByText("350–1500 m")).toBeVisible();
  expect(screen.getByText("v2")).toBeVisible();
  expect(screen.queryByLabelText("起点距离（m）")).not.toBeInTheDocument();
  expect(patches[0]).toEqual({ profile: { scan_distance_start_m: 350, scan_distance_stop_m: 1500 } });

  await userEvent.click(screen.getByRole("button", { name: "设置扫描距离" }));
  await userEvent.clear(screen.getByLabelText("起点距离（m）"));
  await userEvent.clear(screen.getByLabelText("终点距离（m）"));
  await userEvent.click(screen.getByRole("button", { name: "保存扫描距离" }));

  expect(await screen.findByText("全距离（不限）")).toBeVisible();
  expect(screen.getByText("v3")).toBeVisible();
  expect(patches[1]).toEqual({ profile: { scan_distance_start_m: null, scan_distance_stop_m: null } });
});

it.each([
  ["-1", "", "请输入有限且非负的距离"],
  ["abc", "", "请输入有限且非负的距离"],
  ["Infinity", "", "请输入有限且非负的距离"],
  ["1e309", "", "请输入有限且非负的距离"],
  ["", "-1", "请输入有限且非负的距离"],
  ["300", "300", "终点距离必须大于起点距离"],
  ["300", "200", "终点距离必须大于起点距离"],
])("rejects invalid scan distance %s to %s before saving", async (start, stop, message) => {
  let patches = 0;
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([site("JJK-01")])),
    http.patch("/api/v1/sites/id-JJK-01", () => {
      patches += 1;
      return HttpResponse.json(site("JJK-01"));
    }),
  );
  renderApp("/sites");
  await userEvent.click(await screen.findByRole("button", { name: "设置扫描距离" }));
  if (start) await userEvent.type(screen.getByLabelText("起点距离（m）"), start);
  if (stop) await userEvent.type(screen.getByLabelText("终点距离（m）"), stop);
  await userEvent.click(screen.getByRole("button", { name: "保存扫描距离" }));

  expect(await screen.findByRole("alert")).toHaveTextContent(message);
  expect(patches).toBe(0);
});

it("keeps scan distance input after a failed save and discards it on cancel", async () => {
  let patches = 0;
  server.use(
    http.get("/api/v1/sites", () => HttpResponse.json([site("JJK-01")])),
    http.patch("/api/v1/sites/id-JJK-01", () => {
      patches += 1;
      return HttpResponse.json({ detail: "暂时无法保存" }, { status: 503 });
    }),
  );
  renderApp("/sites");
  await userEvent.click(await screen.findByRole("button", { name: "设置扫描距离" }));
  await userEvent.type(screen.getByLabelText("起点距离（m）"), "300");
  await userEvent.click(screen.getByRole("button", { name: "保存扫描距离" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("暂时无法保存");
  expect(screen.getByLabelText("起点距离（m）")).toHaveValue("300");
  expect(screen.getByRole("button", { name: "保存扫描距离" })).toBeEnabled();
  await userEvent.click(screen.getByRole("button", { name: "取消" }));
  expect(screen.queryByLabelText("起点距离（m）")).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "设置扫描距离" }));
  expect(screen.getByLabelText("起点距离（m）")).toHaveValue("");
  expect(patches).toBe(1);
});
