import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { transferableAbortController } from "node:util";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, beforeEach, expect, it, vi } from "vitest";
import type { Site } from "../api";
import ScanDirectoryDialog from "./ScanDirectoryDialog";

const SITE: Site = {
  id: "site-1",
  name: "JJK NAS Archive",
  data_path: "D:/private/nas/archive",
  timezone: "Asia/Shanghai",
  cable_type: "downhole",
  enabled: true,
  created_at: "2026-09-06T00:00:00Z",
  current_profile: {
    id: "profile-1",
    version: 1,
    payload: {},
    is_current: true,
    created_at: "2026-09-06T00:00:00Z",
  },
};

const ROOT_LISTING = {
  site_id: SITE.id,
  site_name: SITE.name,
  path: "",
  parent_path: null,
  directories: [{ name: "jjk20240608", path: "jjk20240608" }],
};

const DIRECTORY_LISTING = {
  site_id: SITE.id,
  site_name: SITE.name,
  path: "jjk20240608",
  parent_path: "",
  directories: [{ name: "day", path: "jjk20240608/day" }],
};

const server = setupServer();
const domAbortController = globalThis.AbortController;
const nativeAbortController = transferableAbortController()
  .constructor as typeof AbortController;

beforeAll(() => {
  globalThis.AbortController = nativeAbortController;
  server.listen();
});

beforeEach(() => {
  server.resetHandlers();
  server.use(
    http.get("/api/v1/sites/:siteId/scan-directories", ({ request }) => {
      const path = new URL(request.url).searchParams.get("path");
      return HttpResponse.json(path === "jjk20240608" ? DIRECTORY_LISTING : ROOT_LISTING);
    }),
  );
});

afterAll(() => {
  server.close();
  globalThis.AbortController = domAbortController;
});

it("opens at the server root without exposing the site's data path", async () => {
  render(<ScanDirectoryDialog site={SITE} onCancel={vi.fn()} onSelect={vi.fn()} />);

  expect(await screen.findByText("JJK NAS Archive")).toBeVisible();
  expect(screen.getByRole("button", { name: "选择当前目录" })).toBeDisabled();
  expect(screen.queryByRole("button", { name: "返回上级目录" })).not.toBeInTheDocument();
  expect(screen.queryByText(SITE.data_path)).not.toBeInTheDocument();
});

it("navigates server directories and selects the current non-root path", async () => {
  const onSelect = vi.fn();
  render(<ScanDirectoryDialog site={SITE} onCancel={vi.fn()} onSelect={onSelect} />);

  await userEvent.click(await screen.findByRole("button", { name: "jjk20240608" }));
  expect(await screen.findByText("JJK NAS Archive / jjk20240608")).toBeVisible();
  await userEvent.click(screen.getByRole("button", { name: "选择当前目录" }));
  expect(onSelect).toHaveBeenCalledWith("jjk20240608");
});

it("uses only prior server-returned paths while entering and returning from a directory", async () => {
  const requestedPaths: string[] = [];
  server.use(
    http.get("/api/v1/sites/:siteId/scan-directories", ({ request }) => {
      const path = new URL(request.url).searchParams.get("path") ?? "";
      requestedPaths.push(path);
      return HttpResponse.json(path === "jjk20240608" ? DIRECTORY_LISTING : ROOT_LISTING);
    }),
  );
  render(<ScanDirectoryDialog site={SITE} onCancel={vi.fn()} onSelect={vi.fn()} />);

  await userEvent.click(await screen.findByRole("button", { name: "jjk20240608" }));
  await screen.findByText("JJK NAS Archive / jjk20240608");
  await userEvent.click(screen.getByRole("button", { name: "返回上级目录" }));
  await screen.findByRole("button", { name: "jjk20240608" });

  expect(requestedPaths).toEqual(["", "jjk20240608", ""]);
  expect(requestedPaths).not.toContain(SITE.data_path);
});

it("shows a loading state while the directory response is pending", async () => {
  let respond: (() => void) | undefined;
  server.use(
    http.get("/api/v1/sites/:siteId/scan-directories", () =>
      new Promise((resolve) => {
        respond = () => resolve(HttpResponse.json(ROOT_LISTING));
      }),
    ),
  );
  render(<ScanDirectoryDialog site={SITE} onCancel={vi.fn()} onSelect={vi.fn()} />);

  expect(screen.getByText("正在加载目录…")).toBeVisible();
  expect(screen.queryByRole("button", { name: "返回上级目录" })).not.toBeInTheDocument();
  await waitFor(() => expect(respond).toBeDefined());
  await act(async () => {
    respond?.();
  });
  expect(await screen.findByRole("button", { name: "jjk20240608" })).toBeVisible();
});

it("shows a controlled directory loading error", async () => {
  server.use(
    http.get("/api/v1/sites/:siteId/scan-directories", () =>
      HttpResponse.json({ detail: "Invalid scan directory" }, { status: 422 }),
    ),
  );
  render(<ScanDirectoryDialog site={SITE} onCancel={vi.fn()} onSelect={vi.fn()} />);

  expect(await screen.findByRole("alert")).toHaveTextContent("无法加载目录");
});

it("cancels without selecting a directory", async () => {
  const onCancel = vi.fn();
  const onSelect = vi.fn();
  render(<ScanDirectoryDialog site={SITE} onCancel={onCancel} onSelect={onSelect} />);

  await userEvent.click(await screen.findByRole("button", { name: "取消" }));
  expect(onCancel).toHaveBeenCalledTimes(1);
  expect(onSelect).not.toHaveBeenCalled();
});
