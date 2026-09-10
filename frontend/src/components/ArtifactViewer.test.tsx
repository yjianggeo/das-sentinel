import { StrictMode } from "react";
import { act, fireEvent, render, renderHook, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { transferableAbortController } from "node:util";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, beforeEach, expect, it, vi } from "vitest";
import type { EventArtifact } from "../api";
import ArtifactViewer from "./ArtifactViewer";
import { useWaterfallPreview } from "./useWaterfallPreview";

const ARTIFACTS: EventArtifact[] = [
  { id: "art-1", kind: "waterfall_raw", media_type: "image/png" },
  { id: "art-2", kind: "psd", media_type: "image/png" },
];

const PREVIEW_ARTIFACTS: EventArtifact[] = [
  { id: "art-1", kind: "waterfall_raw", media_type: "image/png" },
  { id: "art-6", kind: "waterfall_filtered", media_type: "image/png" },
  { id: "art-3", kind: "waveform", media_type: "image/png" },
  { id: "art-2", kind: "psd", media_type: "image/png" },
  { id: "art-5", kind: "stft", media_type: "image/png" },
];

const OPTIONS = {
  automatic_distance_start_m: 12,
  automatic_distance_stop_m: 20,
  available_distance_start_m: 8,
  available_distance_stop_m: 24,
  primary_detector: "stalta",
  primary_detector_display: "STA/LTA",
  filter_low_hz: 3.5,
  filter_high_hz: 27.25,
  distance_unit: "m",
};

const PREVIEW = {
  distance_start_m: 12,
  distance_stop_m: 20,
  channel_count: 8,
  primary_detector: "stalta",
  primary_detector_display: "STA/LTA",
  filter_low_hz: 3.5,
  filter_high_hz: 27.25,
  raw_png_base64: "cG5nLXJhdy0x",
  filtered_png_base64: "cG5nLWZpbHRlcmVkLTE=",
  waveform_png_base64: "cG5nLXdhdmVmb3JtLTE=",
};

const server = setupServer();
const domAbortController = globalThis.AbortController;
const nativeAbortController = transferableAbortController()
  .constructor as typeof AbortController;
const originalCreateObjectUrl = Object.getOwnPropertyDescriptor(URL, "createObjectURL");
const originalRevokeObjectUrl = Object.getOwnPropertyDescriptor(URL, "revokeObjectURL");
let objectUrlNumber = 0;

async function waitForDefaultDistanceInputs(start = 12, stop = 20) {
  await waitFor(() => {
    expect(screen.getByLabelText("起始距离（m）")).toHaveValue(start);
    expect(screen.getByLabelText("结束距离（m）")).toHaveValue(stop);
  });
}

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
    http.get("/api/v1/events/:eventId/waterfall-options", () =>
      HttpResponse.json(OPTIONS),
    ),
    http.post("/api/v1/events/:eventId/waterfall-preview", () =>
      HttpResponse.json(PREVIEW),
    ),
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
  render(<ArtifactViewer eventId="event-1" artifacts={ARTIFACTS} />);
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
  render(<ArtifactViewer eventId="event-1" artifacts={ARTIFACTS} />);
  const waterfall = screen.getAllByRole("tab")[0];
  const psd = screen.getByRole("tab", { name: "PSD" });

  await screen.findByText("当前显示临时预览");
  await userEvent.click(screen.getByRole("button", { name: "查看原始图件" }));
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
  render(<ArtifactViewer eventId="event-1" artifacts={ARTIFACTS} />);

  for (const tab of screen.getAllByRole("tab")) {
    const panelId = tab.getAttribute("aria-controls");
    const panel = panelId ? document.getElementById(panelId) : null;
    expect(panel).not.toBeNull();
    expect(panel).toHaveAttribute("role", "tabpanel");
    expect(panel).toHaveAttribute("aria-labelledby", tab.id);
  }
});

it("shows and retries an original image decoding failure", async () => {
  let requests = 0;
  server.use(
    http.get("/api/v1/artifacts/art-1", () => {
      requests += 1;
      return new HttpResponse(new Blob(["png"], { type: "image/png" }));
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={ARTIFACTS} />);

  await screen.findByText("当前显示临时预览");
  await userEvent.click(screen.getByRole("button", { name: "查看原始图件" }));
  fireEvent.error(await screen.findByAltText("事件瀑布图"));
  expect(screen.getByRole("alert")).toHaveTextContent("图件加载失败");
  await userEvent.click(screen.getByRole("button", { name: "重试图件" }));

  await waitFor(() => expect(requests).toBe(2));
});

it("falls back explicitly when a preview image fails to decode and cancels a pending replacement", async () => {
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  let pendingSignal: AbortSignal | undefined;
  let requests = 0;
  server.use(http.post("/api/v1/events/event-1/waterfall-preview", async ({ request }) => {
    requests += 1;
    if (requests === 2) {
      pendingSignal = request.signal;
      await blocked;
    }
    return HttpResponse.json(PREVIEW);
  }));
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);
  await screen.findByText("当前显示临时预览");
  const previewUrl = screen.getByAltText("事件瀑布图").getAttribute("src");
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
  await waitFor(() => expect(pendingSignal).toBeDefined());
  await userEvent.click(screen.getByRole("button", { name: "全屏查看" }));
  fireEvent.error(within(screen.getByRole("dialog")).getByAltText("事件瀑布图"));

  expect(screen.getByRole("alert")).toHaveTextContent("临时预览图件加载失败");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(pendingSignal?.aborted).toBe(true);
  expect(URL.revokeObjectURL).toHaveBeenCalledTimes(3);
  expect(URL.revokeObjectURL).toHaveBeenCalledWith(previewUrl);
  await screen.findByText(/当前显示原始图件.*预览不可用/);
  const originalUrl = screen.getByAltText("事件瀑布图").getAttribute("src");
  expect(originalUrl).not.toBe(previewUrl);
  expect(screen.getByRole("link", { name: "下载" })).toHaveAttribute("href", "/api/v1/artifacts/art-1");
  await act(async () => {
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  expect(screen.getByAltText("事件瀑布图")).toHaveAttribute("src", originalUrl);
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
  await screen.findByText("当前显示临时预览");
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(screen.getByAltText("事件瀑布图")).not.toHaveAttribute("src", originalUrl);
});

it("ignores a late decode error for a preview trio that has already been replaced", async () => {
  const { result } = renderHook(() => useWaterfallPreview("event-1"));
  await waitFor(() => expect(result.current.preview).not.toBeNull());
  const firstUrl = result.current.preview!.waterfall_raw;
  await act(async () => { await result.current.generate(); });
  const currentPreview = result.current.preview;
  expect(currentPreview!.waterfall_raw).not.toBe(firstUrl);
  act(() => { result.current.rejectPreview(firstUrl); });
  expect(result.current.preview).toEqual(currentPreview);
  expect(result.current.error).toBe("");
  expect(URL.revokeObjectURL).toHaveBeenCalledTimes(3);
});

it("focuses the fullscreen close control and restores its trigger on Escape", async () => {
  render(<ArtifactViewer eventId="event-1" artifacts={ARTIFACTS} />);
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
  render(<ArtifactViewer eventId="event-1" artifacts={ARTIFACTS} />);
  await userEvent.click(await screen.findByRole("button", { name: "全屏查看" }));

  await userEvent.click(screen.getByRole("dialog"));

  expect(screen.getByRole("dialog")).toBeVisible();
});

it("revokes replaced and unmounted object URLs", async () => {
  const view = render(<ArtifactViewer eventId="event-1" artifacts={ARTIFACTS} />);
  await screen.findByText("当前显示临时预览");
  await userEvent.click(screen.getByRole("button", { name: "查看原始图件" }));
  const originalRawUrl = (await screen.findByAltText("事件瀑布图")).getAttribute("src");

  await userEvent.click(screen.getByRole("tab", { name: "PSD" }));
  const psdUrl = (await screen.findByAltText("事件功率谱")).getAttribute("src");
  expect(URL.revokeObjectURL).toHaveBeenCalledWith(originalRawUrl);

  view.unmount();
  expect(URL.revokeObjectURL).toHaveBeenCalledWith(psdUrl);
});

it("renders the available range, defaults, actual filter band, and detector", async () => {
  // Catches omission of the accessible distance controls and server-provided filter metadata.
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  expect(await screen.findByText("可用 8–24 m")).toBeVisible();
  await waitForDefaultDistanceInputs();
  expect(screen.getByText("滤波瀑布图（3.5–27.25 Hz）")).toBeVisible();
  expect(screen.getByText("主检测器：STA/LTA")).toBeVisible();
});

it("sends an empty automatic request before edits and again after reset", async () => {
  // Catches default/reset controls accidentally turning automatic selection into a manual range.
  const requestBodies: unknown[] = [];
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", async ({ request }) => {
      requestBodies.push(await request.json());
      return HttpResponse.json(PREVIEW);
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  await screen.findByText("可用 8–24 m");
  await waitForDefaultDistanceInputs();
  const generate = screen.getByRole("button", { name: "生成临时预览" });
  await screen.findByText("当前显示临时预览");
  await waitFor(() => expect(requestBodies).toHaveLength(1));
  expect(requestBodies[0]).toEqual({});

  const stopInput = screen.getByLabelText("结束距离（m）");
  await userEvent.clear(stopInput);
  expect(stopInput).toHaveValue(null);
  await userEvent.type(stopInput, "21");
  expect(stopInput).toHaveValue(21);
  await userEvent.click(screen.getByRole("button", { name: "查看原始图件" }));
  await userEvent.click(generate);

  await waitFor(() => expect(requestBodies).toHaveLength(2));
  expect(requestBodies[1]).toEqual({});
});

it("automatically loads one preview trio after options and hides old images and download while waiting", async () => {
  let releaseOptions!: () => void;
  let releasePreview!: () => void;
  const optionsBlocked = new Promise<void>((resolve) => { releaseOptions = resolve; });
  const previewBlocked = new Promise<void>((resolve) => { releasePreview = resolve; });
  const requestBodies: unknown[] = [];
  let originalRequests = 0;
  server.use(
    http.get("/api/v1/events/event-1/waterfall-options", async () => {
      await optionsBlocked;
      return HttpResponse.json(OPTIONS);
    }),
    http.post("/api/v1/events/event-1/waterfall-preview", async ({ request }) => {
      requestBodies.push(await request.json());
      await previewBlocked;
      return HttpResponse.json(PREVIEW);
    }),
    http.get("/api/v1/artifacts/:artifactId", () => {
      originalRequests += 1;
      return new HttpResponse(new Blob(["old UTC image"], { type: "image/png" }));
    }),
  );
  render(<StrictMode><ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} /></StrictMode>);

  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: "下载" })).not.toBeInTheDocument();
  expect(requestBodies).toEqual([]);
  await act(async () => { releaseOptions(); });
  await waitFor(() => expect(requestBodies).toEqual([{}]));
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: "下载" })).not.toBeInTheDocument();
  await act(async () => { releasePreview(); });
  const rawUrl = (await screen.findByAltText("事件瀑布图")).getAttribute("src");
  expect(rawUrl).toMatch(/^blob:/);
  expect(screen.getByRole("link", { name: "下载" })).toHaveAttribute("href", rawUrl);
  await userEvent.click(screen.getByRole("tab", { name: "滤波瀑布图" }));
  expect(screen.getByAltText("事件滤波瀑布图")).toBeVisible();
  await userEvent.click(screen.getByRole("tab", { name: "波形" }));
  expect(screen.getByAltText("事件波形图")).toBeVisible();
  await userEvent.click(screen.getByRole("tab", { name: "瀑布图" }));
  expect(screen.getByAltText("事件瀑布图")).toHaveAttribute("src", rawUrl);
  expect(requestBodies).toEqual([{}]);
  expect(originalRequests).toBe(0);
});

it.each([
  ["waterfall_raw", "瀑布图"],
  ["waterfall_filtered", "滤波瀑布图"],
  ["waveform", "波形"],
])("prefers %s over PSD or STFT when opening an event", async (kind, name) => {
  server.use(http.post("/api/v1/events/:eventId/waterfall-preview", () => HttpResponse.json(PREVIEW)));
  const artifacts = [
    { id: "art-2", kind: "psd", media_type: "image/png" },
    { id: "art-5", kind: "stft", media_type: "image/png" },
    { id: "art-selected", kind, media_type: "image/png" },
  ];
  render(<ArtifactViewer eventId="event-1" artifacts={artifacts} />);
  expect(screen.getByRole("tab", { name })).toHaveAttribute("aria-selected", "true");
  await screen.findByText("当前显示临时预览");
});

it.each(["options", "preview"])("labels original fallback when automatic %s loading fails", async (failure) => {
  server.use(
    http.get("/api/v1/events/event-1/waterfall-options", () => failure === "options"
      ? HttpResponse.json({ detail: "unavailable" }, { status: 500 })
      : HttpResponse.json(OPTIONS)),
    http.post("/api/v1/events/event-1/waterfall-preview", () =>
      HttpResponse.json({ detail: "unavailable" }, { status: 500 })),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={ARTIFACTS} />);
  await screen.findByRole("alert");
  expect(await screen.findByText(/当前显示原始图件/)).toBeVisible();
  expect(screen.getByText(/预览不可用/)).toBeVisible();
  expect(await screen.findByAltText("事件瀑布图")).toBeVisible();
});

it("keeps originals explicitly selected until regeneration or the event changes", async () => {
  const requestedEvents: string[] = [];
  server.use(http.post("/api/v1/events/:eventId/waterfall-preview", ({ params }) => {
    requestedEvents.push(String(params.eventId));
    return HttpResponse.json(PREVIEW);
  }));
  const view = render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);
  await screen.findByText("当前显示临时预览");
  await userEvent.click(screen.getByRole("button", { name: "查看原始图件" }));
  await screen.findByText(/当前显示原始图件/);
  await userEvent.click(screen.getByRole("tab", { name: "波形" }));
  await screen.findByAltText("事件波形图");
  expect(screen.getByText(/当前显示原始图件/)).toBeVisible();
  expect(requestedEvents).toEqual(["event-1"]);
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
  await screen.findByText("当前显示临时预览");
  expect(requestedEvents).toEqual(["event-1", "event-1"]);
  await userEvent.click(screen.getByRole("button", { name: "查看原始图件" }));
  view.rerender(<ArtifactViewer eventId="event-2" artifacts={PREVIEW_ARTIFACTS} />);
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: "下载" })).not.toBeInTheDocument();
  expect(screen.getByRole("tab", { name: "瀑布图" })).toHaveAttribute("aria-selected", "true");
  await screen.findByText("当前显示临时预览");
  expect(requestedEvents).toEqual(["event-1", "event-1", "event-2"]);
});

it("does not generate a preview from options that arrive after leaving the event", async () => {
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  let optionsSignal: AbortSignal | undefined;
  const requestedEvents: string[] = [];
  server.use(
    http.get("/api/v1/events/event-1/waterfall-options", async ({ request }) => {
      optionsSignal = request.signal;
      await blocked;
      return HttpResponse.json(OPTIONS);
    }),
    http.post("/api/v1/events/:eventId/waterfall-preview", ({ params }) => {
      requestedEvents.push(String(params.eventId));
      return HttpResponse.json(PREVIEW);
    }),
  );
  const view = render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);
  await waitFor(() => expect(optionsSignal).toBeDefined());
  view.rerender(<ArtifactViewer eventId="event-2" artifacts={PREVIEW_ARTIFACTS} />);
  expect(optionsSignal?.aborted).toBe(true);
  await screen.findByText("当前显示临时预览");
  const currentUrl = screen.getByAltText("事件瀑布图").getAttribute("src");
  await act(async () => {
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  expect(requestedEvents).toEqual(["event-2"]);
  expect(screen.getByAltText("事件瀑布图")).toHaveAttribute("src", currentUrl);
});

it("clears the previous event preview and fullscreen while the next event is preparing", async () => {
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => { release = resolve; });
  server.use(http.post("/api/v1/events/event-2/waterfall-preview", async () => {
    await blocked;
    return HttpResponse.json(PREVIEW);
  }));
  const view = render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);
  await screen.findByText("当前显示临时预览");
  const previousUrl = screen.getByAltText("事件瀑布图").getAttribute("src");
  await userEvent.click(screen.getByRole("button", { name: "全屏查看" }));
  expect(screen.getByRole("dialog")).toBeVisible();
  view.rerender(<ArtifactViewer eventId="event-2" artifacts={PREVIEW_ARTIFACTS} />);
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: "下载" })).not.toBeInTheDocument();
  expect(URL.revokeObjectURL).toHaveBeenCalledTimes(3);
  await act(async () => { release(); });
  await screen.findByText("当前显示临时预览");
  expect(screen.getByAltText("事件瀑布图")).not.toHaveAttribute("src", previousUrl);
});

it("sends both numeric bounds after either distance input is edited", async () => {
  // Catches edited controls continuing to request automatic event channels.
  let requestBody: unknown;
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", async ({ request }) => {
      requestBody = await request.json();
      return HttpResponse.json(PREVIEW);
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  await screen.findByText("可用 8–24 m");
  await waitForDefaultDistanceInputs();
  const stopInput = screen.getByLabelText("结束距离（m）");
  await userEvent.clear(stopInput);
  expect(stopInput).toHaveValue(null);
  await userEvent.type(stopInput, "21");
  expect(stopInput).toHaveValue(21);
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));

  await waitFor(() => expect(requestBody).toEqual({ distance_start_m: 12, distance_stop_m: 21 }));
});

it("selects the full available distance range before explicitly generating the preview", async () => {
  const requestBodies: unknown[] = [];
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", async ({ request }) => {
      requestBodies.push(await request.json());
      return HttpResponse.json(PREVIEW);
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);
  await screen.findByText("当前显示临时预览");
  await waitForDefaultDistanceInputs();

  await userEvent.clear(screen.getByLabelText("起始距离（m）"));
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
  expect(screen.getByRole("alert")).toHaveTextContent("请输入有效距离");
  await userEvent.click(screen.getByRole("button", { name: "全距离范围" }));

  expect(screen.getByLabelText("起始距离（m）")).toHaveValue(8);
  expect(screen.getByLabelText("结束距离（m）")).toHaveValue(24);
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(requestBodies).toEqual([{}]);
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
  await waitFor(() => expect(requestBodies).toEqual([
    {}, { distance_start_m: 8, distance_stop_m: 24 },
  ]));
  await waitFor(() => expect(screen.getByRole("button", { name: "查看原始图件" })).toBeEnabled());
  await userEvent.click(screen.getByRole("button", { name: "查看原始图件" }));
  await waitForDefaultDistanceInputs();
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
  await waitFor(() => expect(requestBodies[2]).toEqual({}));
});

it("returns to automatic mode when the selected event options change", async () => {
  // Catches manual mode leaking from the previously selected event.
  let requestBody: unknown;
  server.use(
    http.get("/api/v1/events/:eventId/waterfall-options", ({ params }) =>
      HttpResponse.json(
        params.eventId === "event-2"
          ? {
              ...OPTIONS,
              automatic_distance_start_m: 9,
              automatic_distance_stop_m: 19,
            }
          : OPTIONS,
      ),
    ),
    http.post("/api/v1/events/event-2/waterfall-preview", async ({ request }) => {
      requestBody = await request.json();
      return HttpResponse.json(PREVIEW);
    }),
  );
  const view = render(
    <ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />,
  );

  await screen.findByText("可用 8–24 m");
  await waitForDefaultDistanceInputs();
  const stopInput = screen.getByLabelText("结束距离（m）");
  await userEvent.clear(stopInput);
  expect(stopInput).toHaveValue(null);
  await userEvent.type(stopInput, "21");
  expect(stopInput).toHaveValue(21);
  view.rerender(<ArtifactViewer eventId="event-2" artifacts={PREVIEW_ARTIFACTS} />);
  await waitFor(() => expect(screen.getByLabelText("结束距离（m）")).toHaveValue(19));
  await screen.findByText("当前显示临时预览");

  await waitFor(() => expect(requestBody).toBeDefined());
  expect(requestBody).toEqual({});
});

it("allows an untouched single-channel automatic range", async () => {
  // Catches manual increasing-range validation being applied to automatic event-channel selection.
  let requestBody: unknown;
  server.use(
    http.get("/api/v1/events/event-1/waterfall-options", () =>
      HttpResponse.json({
        ...OPTIONS,
        automatic_distance_start_m: 12,
        automatic_distance_stop_m: 12,
      }),
    ),
    http.post("/api/v1/events/event-1/waterfall-preview", async ({ request }) => {
      requestBody = await request.json();
      return HttpResponse.json({
        ...PREVIEW,
        distance_stop_m: 12,
        channel_count: 1,
      });
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  await screen.findByText("可用 8–24 m");
  await screen.findByText("当前显示临时预览");

  await waitFor(() => expect(requestBody).toEqual({}));
  expect(screen.getByRole("img", { name: "事件瀑布图" })).toHaveAttribute(
    "src",
    expect.stringMatching(/^blob:/),
  );
});

it("rejects a blank manual distance without posting again after the automatic preview", async () => {
  // Catches Number("") silently converting a blank bound to zero.
  let postCount = 0;
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", () => {
      postCount += 1;
      return HttpResponse.json(PREVIEW);
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  await screen.findByText("可用 8–24 m");
  await waitForDefaultDistanceInputs();
  await screen.findByText("当前显示临时预览");
  expect(postCount).toBe(1);
  const startInput = screen.getByLabelText("起始距离（m）");
  await userEvent.clear(startInput);
  expect(startInput).toHaveValue(null);
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));

  expect(screen.getByRole("alert")).toHaveTextContent("请输入有效距离");
  expect(postCount).toBe(1);
});

it("rejects an invalid manual distance range without posting again after the automatic preview", async () => {
  // Catches client-side validation being bypassed before the preview POST.
  let postCount = 0;
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", () => {
      postCount += 1;
      return HttpResponse.json(PREVIEW);
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  await screen.findByText("可用 8–24 m");
  await waitForDefaultDistanceInputs();
  await screen.findByText("当前显示临时预览");
  expect(postCount).toBe(1);
  const stopInput = screen.getByLabelText("结束距离（m）");
  await userEvent.clear(stopInput);
  expect(stopInput).toHaveValue(null);
  await userEvent.type(stopInput, "10");
  expect(stopInput).toHaveValue(10);
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));

  expect(screen.getByRole("alert")).toHaveTextContent("起始距离必须小于结束距离");
  expect(postCount).toBe(1);
});

it("uses one preview response for both waterfalls and waveform while PSD and STFT stay original", async () => {
  // Catches non-atomic image selection or accidental replacement of the analysis-only artifacts.
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", () =>
      HttpResponse.json(PREVIEW),
    ),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  await screen.findByText("可用 8–24 m");
  await waitForDefaultDistanceInputs();
  await screen.findByText("当前显示临时预览");

  await waitFor(() => {
    expect(screen.getByRole("img", { name: "事件瀑布图" })).toHaveAttribute(
      "src",
      expect.stringMatching(/^blob:/),
    );
  });
  const rawDownload = screen.getByRole("link", { name: "下载" });
  const rawPreviewUrl = screen.getByRole("img", { name: "事件瀑布图" }).getAttribute("src");
  expect(rawDownload).toHaveAttribute("href", rawPreviewUrl);
  expect(screen.getAllByText("当前显示临时预览")).toHaveLength(1);

  await userEvent.click(screen.getByRole("tab", { name: "滤波瀑布图" }));
  expect(screen.getByRole("img", { name: "事件滤波瀑布图" })).toHaveAttribute(
    "src",
    expect.stringMatching(/^blob:/),
  );
  const filteredPreviewUrl = screen
    .getByRole("img", { name: "事件滤波瀑布图" })
    .getAttribute("src");
  expect(screen.getByRole("link", { name: "下载" })).toHaveAttribute("href", filteredPreviewUrl);
  expect(screen.getAllByText("当前显示临时预览")).toHaveLength(1);

  await userEvent.click(screen.getByRole("tab", { name: "波形" }));
  const waveformPreviewUrl = screen.getByRole("img", { name: "事件波形图" }).getAttribute("src");
  expect(waveformPreviewUrl).toMatch(/^blob:/);
  expect(waveformPreviewUrl).not.toBe(rawPreviewUrl);
  expect(waveformPreviewUrl).not.toBe(filteredPreviewUrl);
  expect(screen.getByRole("link", { name: "下载" })).toHaveAttribute("href", waveformPreviewUrl);
  expect(screen.getAllByText("当前显示临时预览")).toHaveLength(1);

  const stopInput = screen.getByLabelText("结束距离（m）");
  await userEvent.clear(stopInput);
  expect(stopInput).toHaveValue(null);
  await userEvent.type(stopInput, "21");
  expect(stopInput).toHaveValue(21);
  expect(screen.getAllByText("当前显示临时预览")).toHaveLength(1);
  expect(screen.getByRole("img", { name: "事件波形图" })).toHaveAttribute(
    "src",
    waveformPreviewUrl,
  );

  await userEvent.click(screen.getByRole("tab", { name: "PSD" }));
  await screen.findByAltText("事件功率谱");
  expect(screen.queryByText("当前显示临时预览")).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "下载" })).toHaveAttribute(
    "href",
    "/api/v1/artifacts/art-2",
  );
  await userEvent.click(screen.getByRole("tab", { name: "STFT" }));
  await screen.findByAltText("事件时频图");
  expect(screen.queryByText("当前显示临时预览")).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "下载" })).toHaveAttribute(
    "href",
    "/api/v1/artifacts/art-5",
  );
});

it("viewing originals restores automatic inputs, clears validation, and revokes the preview trio", async () => {
  // Catches a URL-only reset that leaves edited inputs or validation behind.
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", () =>
      HttpResponse.json(PREVIEW),
  ),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  await screen.findByText("可用 8–24 m");
  await waitForDefaultDistanceInputs();
  await screen.findByText("当前显示临时预览");
  const rawUrl = await waitFor(() => {
    const url = screen.getByRole("img", { name: "事件瀑布图" }).getAttribute("src");
    expect(url).toMatch(/^blob:/);
    return url!;
  });
  await userEvent.click(screen.getByRole("tab", { name: "滤波瀑布图" }));
  const filteredUrl = screen.getByRole("img", { name: "事件滤波瀑布图" }).getAttribute("src");
  expect(filteredUrl).toMatch(/^blob:/);
  await userEvent.click(screen.getByRole("tab", { name: "波形" }));
  const waveformUrl = screen.getByRole("img", { name: "事件波形图" }).getAttribute("src");
  expect(waveformUrl).toMatch(/^blob:/);

  const startInput = screen.getByLabelText("起始距离（m）");
  await userEvent.clear(startInput);
  expect(startInput).toHaveValue(null);
  await userEvent.type(startInput, "21");
  expect(startInput).toHaveValue(21);
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
  expect(screen.getByRole("alert")).toHaveTextContent("起始距离必须小于结束距离");

  await userEvent.click(screen.getByRole("button", { name: "查看原始图件" }));

  expect(screen.getByLabelText("起始距离（m）")).toHaveValue(12);
  expect(screen.getByLabelText("结束距离（m）")).toHaveValue(20);
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(await screen.findByRole("link", { name: "下载" })).toHaveAttribute(
    "href",
    "/api/v1/artifacts/art-3",
  );
  expect(URL.revokeObjectURL).toHaveBeenCalledWith(rawUrl);
  expect(URL.revokeObjectURL).toHaveBeenCalledWith(filteredUrl);
  expect(URL.revokeObjectURL).toHaveBeenCalledWith(waveformUrl);
});

it("revokes the first preview trio before displaying a second trio", async () => {
  // Catches object URL accumulation across repeated preview generation.
  let requestNumber = 0;
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", () => {
      requestNumber += 1;
      return HttpResponse.json({
        ...PREVIEW,
        raw_png_base64: `cG5nLXJhdy0${requestNumber}`,
        filtered_png_base64: `cG5nLWZpbHRlcmVkL${requestNumber}`,
        waveform_png_base64: `cG5nLXdhdmVmb3JtL${requestNumber}`,
      });
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  await screen.findByText("可用 8–24 m");
  await screen.findByText("当前显示临时预览");
  const firstRaw = await waitFor(() => {
    const url = screen.getByRole("img", { name: "事件瀑布图" }).getAttribute("src");
    expect(url).toMatch(/^blob:/);
    return url!;
  });
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));

  await waitFor(() => expect(URL.revokeObjectURL).toHaveBeenCalledWith(firstRaw));
  expect(URL.revokeObjectURL).toHaveBeenCalledTimes(3);
});

it.each([409, 422, 500])("preserves the current preview after a %s response", async (status) => {
  // Catches failed requests clearing a valid preview image.
  let requestNumber = 0;
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", () => {
      requestNumber += 1;
      return requestNumber === 1
        ? HttpResponse.json(PREVIEW)
        : HttpResponse.json({ detail: "failed" }, { status });
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  await screen.findByText("可用 8–24 m");
  await waitForDefaultDistanceInputs();
  const generate = screen.getByRole("button", { name: "生成临时预览" });
  await screen.findByText("当前显示临时预览");
  const firstUrl = await waitFor(() => {
    const url = screen.getByRole("img", { name: "事件瀑布图" }).getAttribute("src");
    expect(url).toMatch(/^blob:/);
    return url!;
  });
  const stopInput = screen.getByLabelText("结束距离（m）");
  await userEvent.clear(stopInput);
  expect(stopInput).toHaveValue(null);
  await userEvent.type(stopInput, "21");
  expect(stopInput).toHaveValue(21);
  await userEvent.click(generate);

  expect(await screen.findByRole("alert")).toHaveTextContent("无法生成临时预览");
  expect(screen.getByRole("img", { name: "事件瀑布图" })).toHaveAttribute("src", firstUrl);
});

it("reclaims only the new raw and filtered URLs when the waveform Base64 decode fails", async () => {
  // Catches a third-image decode failure leaking new URLs or clearing/revoking the active trio.
  let requestNumber = 0;
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", () => {
      requestNumber += 1;
      return HttpResponse.json(
        requestNumber === 1
          ? PREVIEW
          : { ...PREVIEW, waveform_png_base64: "%not-base64%" },
      );
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);
  await screen.findByText("可用 8–24 m");
  const generate = screen.getByRole("button", { name: "生成临时预览" });
  await screen.findByText("当前显示临时预览");
  const existingRawUrl = await waitFor(() => {
    const url = screen.getByRole("img", { name: "事件瀑布图" }).getAttribute("src");
    expect(url).toMatch(/^blob:/);
    return url!;
  });
  await userEvent.click(screen.getByRole("tab", { name: "滤波瀑布图" }));
  const existingFilteredUrl = screen
    .getByRole("img", { name: "事件滤波瀑布图" })
    .getAttribute("src")!;
  await userEvent.click(screen.getByRole("tab", { name: "波形" }));
  const existingWaveformUrl = screen
    .getByRole("img", { name: "事件波形图" })
    .getAttribute("src")!;
  const callsBeforeFailure = vi.mocked(URL.createObjectURL).mock.calls.length;

  await userEvent.click(generate);

  expect(await screen.findByRole("alert")).toHaveTextContent("无法生成临时预览");
  expect(screen.getByRole("img", { name: "事件波形图" })).toHaveAttribute(
    "src",
    existingWaveformUrl,
  );
  const newRawUrl = vi.mocked(URL.createObjectURL).mock.results[callsBeforeFailure].value;
  const newFilteredUrl = vi.mocked(URL.createObjectURL).mock.results[callsBeforeFailure + 1].value;
  expect(URL.revokeObjectURL).toHaveBeenCalledWith(newRawUrl);
  expect(URL.revokeObjectURL).toHaveBeenCalledWith(newFilteredUrl);
  expect(URL.revokeObjectURL).not.toHaveBeenCalledWith(existingRawUrl);
  expect(URL.revokeObjectURL).not.toHaveBeenCalledWith(existingFilteredUrl);
  expect(URL.revokeObjectURL).not.toHaveBeenCalledWith(existingWaveformUrl);
});

it("reclaims the first URL when the second createObjectURL call fails", async () => {
  // Catches partial URL creation replacing or leaking over an existing preview.
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", () =>
      HttpResponse.json(PREVIEW),
    ),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);
  await screen.findByText("可用 8–24 m");
  const generate = screen.getByRole("button", { name: "生成临时预览" });
  await screen.findByText("当前显示临时预览");
  const existingUrl = await waitFor(() => {
    const url = screen.getByRole("img", { name: "事件瀑布图" }).getAttribute("src");
    expect(url).toMatch(/^blob:/);
    return url!;
  });
  vi.mocked(URL.createObjectURL)
    .mockImplementationOnce(() => "blob:partial-raw")
    .mockImplementationOnce(() => {
      throw new Error("URL allocation failed");
    });

  await userEvent.click(generate);

  expect(await screen.findByRole("alert")).toHaveTextContent("无法生成临时预览");
  expect(screen.getByRole("img", { name: "事件瀑布图" })).toHaveAttribute(
    "src",
    existingUrl,
  );
  expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:partial-raw");
  expect(URL.revokeObjectURL).not.toHaveBeenCalledWith(existingUrl);
});

it("aborts superseded preview requests and ignores late results while revoking URLs on unmount", async () => {
  // Catches stale event responses and missing request/object URL cleanup.
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => {
    release = resolve;
  });
  let postSignal: AbortSignal | undefined;
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", async ({ request }) => {
      postSignal = request.signal;
      await blocked;
      return HttpResponse.json(PREVIEW);
    }),
  );
  const view = render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);
  await screen.findByText("可用 8–24 m");
  await waitFor(() => expect(postSignal).toBeDefined());

  view.rerender(<ArtifactViewer eventId="event-2" artifacts={PREVIEW_ARTIFACTS} />);
  expect(postSignal?.aborted).toBe(true);
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  await screen.findByText("当前显示临时预览");
  const currentUrl = screen.getByAltText("事件瀑布图").getAttribute("src");
  await act(async () => {
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  expect(screen.getByAltText("事件瀑布图")).toHaveAttribute("src", currentUrl);
  expect(URL.createObjectURL).toHaveBeenCalledTimes(3);
  view.unmount();
  expect(URL.revokeObjectURL).toHaveBeenCalledTimes(3);
});

it("aborts a pending preview on direct unmount without creating late object URLs", async () => {
  // Catches cleanup that aborts only the old options request after generate replaces its controller.
  let release!: () => void;
  const blocked = new Promise<void>((resolve) => {
    release = resolve;
  });
  let postSignal: AbortSignal | undefined;
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", async ({ request }) => {
      postSignal = request.signal;
      await blocked;
      return HttpResponse.json(PREVIEW);
    }),
  );
  const view = render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);
  await screen.findByText("可用 8–24 m");
  const createdBeforePreview = vi.mocked(URL.createObjectURL).mock.calls.length;
  await waitFor(() => expect(postSignal).toBeDefined());

  view.unmount();
  expect(postSignal?.aborted).toBe(true);
  await act(async () => {
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  expect(URL.createObjectURL).toHaveBeenCalledTimes(createdBeforePreview);
});
