import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, beforeEach } from "vitest";
import { renderApp } from "../test/render";

const server = setupServer();

beforeAll(() => server.listen());
beforeEach(() => server.resetHandlers());
afterAll(() => server.close());

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
    http.get("/api/v1/jobs", () => HttpResponse.json([])),
    http.get("/api/v1/events", () => HttpResponse.json([])),
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
