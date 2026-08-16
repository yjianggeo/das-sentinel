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
