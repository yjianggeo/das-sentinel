import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import * as apiModule from "../api";
import { renderApp } from "../test/render";

const server = setupServer();

beforeAll(() => server.listen());
beforeEach(() => server.resetHandlers());
afterAll(() => server.close());
afterEach(() => vi.restoreAllMocks());

it("does not arm global 401 handling while the session is unauthenticated", async () => {
  const unauthorizedHandlerSpy = vi.spyOn(apiModule, "setUnauthorizedHandler");
  server.use(
    http.get("/api/v1/session", () => new HttpResponse(null, { status: 401 })),
    http.post("/api/v1/session", () => new HttpResponse(null, { status: 401 })),
  );
  renderApp("/login");

  const submit = await screen.findByRole("button");
  const password = submit.closest("form")?.querySelector("input");
  if (!(password instanceof HTMLInputElement)) {
    throw new Error("login password input is missing");
  }
  await userEvent.type(password, "wrong");
  await userEvent.click(submit);

  await screen.findByRole("alert");
  expect(unauthorizedHandlerSpy).not.toHaveBeenCalledWith(expect.any(Function));
});

it("logs in and redirects to the dashboard", async () => {
  let authenticated = false;
  server.use(
    http.get("/api/v1/session", () => {
      if (!authenticated) {
        return new HttpResponse(null, { status: 401 });
      }
      return HttpResponse.json({ admin_id: 1, csrf_token: "token-1" });
    }),
    http.post("/api/v1/session", () => {
      authenticated = true;
      return new HttpResponse(null, {
        status: 204,
        headers: { "x-csrf-token": "token-1" },
      });
    }),
    http.get("/api/v1/dashboard", () => HttpResponse.json({
      job_counts: {},
      today_event_count: 0,
      machine_label_counts: {},
    })),
  );
  renderApp("/login");

  await userEvent.type(
    await screen.findByLabelText("管理员密码"),
    "correct horse",
  );
  await userEvent.click(screen.getByRole("button", { name: "登录" }));

  expect(
    await screen.findByRole("heading", { name: "运行总览" }),
  ).toBeVisible();
});

it("keeps the session and shows an error when logout fails", async () => {
  server.use(
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
    http.get("/api/v1/dashboard", () => HttpResponse.json({
      job_counts: {},
      today_event_count: 0,
      machine_label_counts: {},
    })),
    http.delete("/api/v1/session", () =>
      HttpResponse.json({ detail: "failed" }, { status: 500 }),
    ),
  );
  renderApp("/");

  await userEvent.click(await screen.findByRole("button", { name: "退出登录" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("退出失败");
  expect(screen.queryByRole("button", { name: "登录" })).not.toBeInTheDocument();
});

it("clears the session once when a dashboard request returns 401", async () => {
  let dashboardRequests = 0;
  server.use(
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
    http.get("/api/v1/dashboard", () => {
      dashboardRequests += 1;
      return HttpResponse.json({ detail: "Unauthorized" }, { status: 401 });
    }),
  );
  renderApp("/");

  expect(await screen.findByRole("button", { name: "登录" })).toBeVisible();
  expect(dashboardRequests).toBe(1);
});

it("clears the session after a successful logout", async () => {
  server.use(
    http.get("/api/v1/session", () =>
      HttpResponse.json({ admin_id: 1, csrf_token: "token-1" }),
    ),
    http.get("/api/v1/dashboard", () => HttpResponse.json({
      job_counts: {},
      today_event_count: 0,
      machine_label_counts: {},
    })),
    http.delete("/api/v1/session", () => new HttpResponse(null, { status: 204 })),
  );
  renderApp("/");

  await userEvent.click(await screen.findByRole("button", { name: "退出登录" }));

  expect(await screen.findByRole("button", { name: "登录" })).toBeVisible();
});

it("shows an error on a wrong password", async () => {
  server.use(
    http.get("/api/v1/session", () => new HttpResponse(null, { status: 401 })),
    http.post("/api/v1/session", () => new HttpResponse(null, { status: 401 })),
  );
  renderApp("/login");

  await userEvent.type(await screen.findByLabelText("管理员密码"), "wrong");
  await userEvent.click(screen.getByRole("button", { name: "登录" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("密码错误");
});
