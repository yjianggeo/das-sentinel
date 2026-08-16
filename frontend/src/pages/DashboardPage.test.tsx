import { screen } from "@testing-library/react";
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

  expect(await screen.findByText("145")).toBeVisible();
  expect(screen.getByText(/车辆：120/)).toBeVisible();
  expect(dashboardRequests).toBe(1);
});
