import { afterEach, expect, it, vi } from "vitest";
import { api, ApiError, setUnauthorizedHandler } from "./api";

afterEach(() => {
  setUnauthorizedHandler(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

it("ignores a stale artifact 401 after the unauthorized handler changes", async () => {
  const oldHandler = vi.fn();
  const newHandler = vi.fn();
  let resolveResponse: (response: Response) => void = () => {
    throw new Error("artifact request did not start");
  };
  const response = new Promise<Response>((resolve) => {
    resolveResponse = resolve;
  });
  vi.stubGlobal("fetch", vi.fn(() => response));

  setUnauthorizedHandler(oldHandler);
  const request = api.getBlob("/api/v1/artifacts/art-1");
  setUnauthorizedHandler(newHandler);
  resolveResponse(
    Response.json({ detail: "Unauthorized" }, { status: 401 }),
  );

  const cause = await request.catch((error: unknown) => error);

  expect(cause).toBeInstanceOf(ApiError);
  expect(cause).toMatchObject({
    status: 401,
    message: "Unauthorized",
  });
  expect(oldHandler).not.toHaveBeenCalled();
  expect(newHandler).not.toHaveBeenCalled();
});

it("forwards AbortSignal for JSON POST requests", async () => {
  vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(Response.json({}))))
  const controller = new AbortController();
  const pending = api.post("/api/v1/events/e1/waterfall-preview", {
    distance_start_m: 12,
    distance_stop_m: 20,
  }, controller.signal);
  controller.abort();
  await expect(pending).rejects.toMatchObject({ name: "AbortError" });
});

it("rejects an aborted DELETE response instead of reporting successful deletion", async () => {
  vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(null, { status: 204 }))));
  const controller = new AbortController();
  const pending = api.delete("/api/v1/jobs/cancelled-job-1", controller.signal);
  controller.abort();
  await expect(pending).rejects.toMatchObject({ name: "AbortError" });
});
