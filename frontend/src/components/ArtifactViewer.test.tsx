import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { transferableAbortController } from "node:util";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, beforeEach, expect, it, vi } from "vitest";
import type { EventArtifact } from "../api";
import ArtifactViewer from "./ArtifactViewer";

const ARTIFACTS: EventArtifact[] = [
  { id: "art-1", kind: "waterfall_raw", media_type: "image/png" },
  { id: "art-2", kind: "psd", media_type: "image/png" },
];

const server = setupServer();
const domAbortController = globalThis.AbortController;
const nativeAbortController = transferableAbortController()
  .constructor as typeof AbortController;
const originalCreateObjectUrl = Object.getOwnPropertyDescriptor(URL, "createObjectURL");
const originalRevokeObjectUrl = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");
let objectUrlNumber = 0;

beforeAll(() => {
  globalThis.AbortController = nativeAbortController;
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    value: vi.fn(() => `blob:artifact-${++objectUrlNumber}`),
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    value: vi.fn(),
  });
  server.listen();
});

beforeEach(() => {
  objectUrlNumber = 0;
  vi.mocked(URL.createObjectURL).mockClear();
  vi.mocked(URL.revokeObjectURL).mockClear();
  server.resetHandlers();
  server.use(
    http.get("/api/v1/artifacts/:artifactId", () =>
      new HttpResponse(new Blob(["png"], { type: "image/png" })),
    ),
  );
});

afterAll(() => {
  server.close();
  globalThis.AbortController = domAbortController;
  if (originalCreateObjectUrl) {
    Object.defineProperty(URL, "createObjectURL", originalCreateObjectUrl);
  } else {
    Reflect.deleteProperty(URL, "createObjectURL");
  }
  if (originalRevokeObjectUrl) {
    Object.defineProperty(URL, "revokeObjectURL", originalRevokeObjectUrl);
  } else {
    Reflect.deleteProperty(URL, "revokeObjectURL");
  }
});

it("supports wrapped arrow, Home, and End navigation with a linked tabpanel", async () => {
  render(<ArtifactViewer artifacts={ARTIFACTS} />);
  const waterfall = screen.getByRole("tab", { name: "瀑布图" });
  const psd = screen.getByRole("tab", { name: "PSD" });

  waterfall.focus();
  await userEvent.keyboard("{ArrowRight}");
  expect(psd).toHaveFocus();
  expect(psd).toHaveAttribute("tabindex", "0");
  expect(waterfall).toHaveAttribute("tabindex", "-1");
  expect(screen.getByRole("tabpanel")).toHaveAccessibleName("PSD");

  await userEvent.keyboard("{Home}");
  expect(waterfall).toHaveFocus();
  await userEvent.keyboard("{ArrowLeft}");
  expect(psd).toHaveFocus();
  await userEvent.keyboard("{End}");
  expect(psd).toHaveFocus();
});

it("keeps a loaded image visible when selecting the already active tab", async () => {
  const requests = new Map<string, number>();
  server.use(
    http.get("/api/v1/artifacts/:artifactId", ({ params }) => {
      const artifactId = String(params.artifactId);
      requests.set(artifactId, (requests.get(artifactId) ?? 0) + 1);
      return new HttpResponse(new Blob(["png"], { type: "image/png" }));
    }),
  );
  render(<ArtifactViewer artifacts={ARTIFACTS} />);
  const waterfall = screen.getAllByRole("tab")[0];
  const psd = screen.getByRole("tab", { name: "PSD" });

  const waterfallImage = await screen.findByRole("img");
  await userEvent.click(waterfall);
  waterfall.focus();
  await userEvent.keyboard("{Home}");
  expect(waterfallImage).toBeVisible();
  expect(requests.get("art-1")).toBe(1);

  await userEvent.click(psd);
  const psdImage = await screen.findByRole("img");
  await userEvent.click(psd);
  psd.focus();
  await userEvent.keyboard("{End}");
  expect(psdImage).toBeVisible();
  expect(requests.get("art-2")).toBe(1);
});

it("renders a bidirectionally linked panel for every artifact tab", () => {
  render(<ArtifactViewer artifacts={ARTIFACTS} />);

  for (const tab of screen.getAllByRole("tab")) {
    const panelId = tab.getAttribute("aria-controls");
    const panel = panelId ? document.getElementById(panelId) : null;
    expect(panel).not.toBeNull();
    expect(panel).toHaveAttribute("role", "tabpanel");
    expect(panel).toHaveAttribute("aria-labelledby", tab.id);
  }
});

it("shows and retries an image decoding failure", async () => {
  let requests = 0;
  server.use(
    http.get("/api/v1/artifacts/art-1", () => {
      requests += 1;
      return new HttpResponse(new Blob(["png"], { type: "image/png" }));
    }),
  );
  render(<ArtifactViewer artifacts={ARTIFACTS} />);

  fireEvent.error(await screen.findByAltText("事件瀑布图"));
  expect(screen.getByRole("alert")).toHaveTextContent("图件加载失败");
  await userEvent.click(screen.getByRole("button", { name: "重试图件" }));

  await waitFor(() => expect(requests).toBe(2));
});

it("focuses the fullscreen close control and restores its trigger on Escape", async () => {
  render(<ArtifactViewer artifacts={ARTIFACTS} />);
  const trigger = await screen.findByRole("button", { name: "全屏查看" });

  await userEvent.click(trigger);
  const close = screen.getByRole("button", { name: "关闭全屏" });
  expect(close).toHaveFocus();
  await userEvent.tab();
  expect(close).toHaveFocus();
  await userEvent.tab({ shift: true });
  expect(close).toHaveFocus();
  await userEvent.keyboard("{Escape}");

  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

it("does not close the fullscreen dialog when its content is clicked", async () => {
  render(<ArtifactViewer artifacts={ARTIFACTS} />);
  await userEvent.click(await screen.findByRole("button", { name: "全屏查看" }));

  await userEvent.click(screen.getByRole("dialog"));

  expect(screen.getByRole("dialog")).toBeVisible();
});

it("revokes replaced and unmounted object URLs", async () => {
  const view = render(<ArtifactViewer artifacts={ARTIFACTS} />);
  expect(await screen.findByAltText("事件瀑布图")).toHaveAttribute(
    "src",
    "blob:artifact-1",
  );

  await userEvent.click(screen.getByRole("tab", { name: "PSD" }));
  expect(await screen.findByAltText("事件功率谱")).toHaveAttribute(
    "src",
    "blob:artifact-2",
  );
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:artifact-1");

  view.unmount();
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:artifact-2");
});
