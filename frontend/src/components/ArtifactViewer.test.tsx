import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

it("shows and retries an image decoding failure", async () => {
  let requests = 0;
  server.use(
    http.get("/api/v1/artifacts/art-1", () => {
      requests += 1;
      return new HttpResponse(new Blob(["png"], { type: "image/png" }));
    }),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={ARTIFACTS} />);

  fireEvent.error(await screen.findByAltText("事件瀑布图"));
  expect(screen.getByRole("alert")).toHaveTextContent("图件加载失败");
  await userEvent.click(screen.getByRole("button", { name: "重试图件" }));

  await waitFor(() => expect(requests).toBe(2));
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

it("renders the available range, defaults, actual filter band, and detector", async () => {
  // Catches omission of the accessible distance controls and server-provided filter metadata.
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  expect(await screen.findByText("可用 8–24 m")).toBeVisible();
  expect(screen.getByLabelText("起始距离（m）")).toHaveValue(12);
  expect(screen.getByLabelText("结束距离（m）")).toHaveValue(20);
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
  const generate = screen.getByRole("button", { name: "生成临时预览" });
  await userEvent.click(generate);
  await waitFor(() => expect(requestBodies).toHaveLength(1));
  expect(requestBodies[0]).toEqual({});

  await userEvent.clear(screen.getByLabelText("结束距离（m）"));
  await userEvent.type(screen.getByLabelText("结束距离（m）"), "21");
  await userEvent.click(screen.getByRole("button", { name: "重置临时预览" }));
  await userEvent.click(generate);

  await waitFor(() => expect(requestBodies).toHaveLength(2));
  expect(requestBodies[1]).toEqual({});
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
  await userEvent.clear(screen.getByLabelText("结束距离（m）"));
  await userEvent.type(screen.getByLabelText("结束距离（m）"), "21");
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));

  await waitFor(() => expect(requestBody).toBeDefined());
  expect(requestBody).toEqual({ distance_start_m: 12, distance_stop_m: 21 });
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
  await userEvent.clear(screen.getByLabelText("结束距离（m）"));
  await userEvent.type(screen.getByLabelText("结束距离（m）"), "21");
  view.rerender(<ArtifactViewer eventId="event-2" artifacts={PREVIEW_ARTIFACTS} />);
  await waitFor(() => expect(screen.getByLabelText("结束距离（m）")).toHaveValue(19));
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));

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
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));

  await waitFor(() => expect(requestBody).toEqual({}));
  expect(screen.getByRole("img", { name: "事件瀑布图" })).toHaveAttribute(
    "src",
    expect.stringMatching(/^blob:/),
  );
});

it("rejects a blank manual distance before numeric conversion without posting", async () => {
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
  await userEvent.clear(screen.getByLabelText("起始距离（m）"));
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));

  expect(screen.getByRole("alert")).toHaveTextContent("请输入有效距离");
  expect(postCount).toBe(0);
});

it("rejects an invalid manual distance range without posting", async () => {
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
  await userEvent.clear(screen.getByLabelText("结束距离（m）"));
  await userEvent.type(screen.getByLabelText("结束距离（m）"), "10");
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));

  expect(screen.getByRole("alert")).toHaveTextContent("起始距离必须小于结束距离");
  expect(postCount).toBe(0);
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
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));

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

  await userEvent.clear(screen.getByLabelText("结束距离（m）"));
  await userEvent.type(screen.getByLabelText("结束距离（m）"), "21");
  expect(screen.getAllByText("当前显示临时预览")).toHaveLength(1);
  expect(screen.getByRole("img", { name: "事件波形图" })).toHaveAttribute(
    "src",
    waveformPreviewUrl,
  );

  await userEvent.click(screen.getByRole("tab", { name: "PSD" }));
  expect(screen.queryByText("当前显示临时预览")).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "下载" })).toHaveAttribute(
    "href",
    "/api/v1/artifacts/art-2",
  );
  await userEvent.click(screen.getByRole("tab", { name: "STFT" }));
  expect(screen.queryByText("当前显示临时预览")).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "下载" })).toHaveAttribute(
    "href",
    "/api/v1/artifacts/art-5",
  );
});

it("reset restores automatic inputs, clears validation, permanent URLs, and preview objects", async () => {
  // Catches a URL-only reset that leaves edited inputs or validation behind.
  server.use(
    http.post("/api/v1/events/event-1/waterfall-preview", () =>
      HttpResponse.json(PREVIEW),
  ),
  );
  render(<ArtifactViewer eventId="event-1" artifacts={PREVIEW_ARTIFACTS} />);

  await screen.findByText("可用 8–24 m");
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
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

  await userEvent.clear(screen.getByLabelText("起始距离（m）"));
  await userEvent.type(screen.getByLabelText("起始距离（m）"), "21");
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
  expect(screen.getByRole("alert")).toHaveTextContent("起始距离必须小于结束距离");

  await userEvent.click(screen.getByRole("button", { name: "重置临时预览" }));

  expect(screen.getByLabelText("起始距离（m）")).toHaveValue(12);
  expect(screen.getByLabelText("结束距离（m）")).toHaveValue(20);
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "下载" })).toHaveAttribute(
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
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
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
  const generate = screen.getByRole("button", { name: "生成临时预览" });
  await userEvent.click(generate);
  const firstUrl = await waitFor(() => {
    const url = screen.getByRole("img", { name: "事件瀑布图" }).getAttribute("src");
    expect(url).toMatch(/^blob:/);
    return url!;
  });
  await userEvent.clear(screen.getByLabelText("结束距离（m）"));
  await userEvent.type(screen.getByLabelText("结束距离（m）"), "21");
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
  await userEvent.click(generate);
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
  await userEvent.click(generate);
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
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
  await waitFor(() => expect(postSignal).toBeDefined());

  view.rerender(<ArtifactViewer eventId="event-2" artifacts={PREVIEW_ARTIFACTS} />);
  expect(postSignal?.aborted).toBe(true);
  await act(async () => {
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  expect(screen.queryByRole("img", { name: "事件瀑布图" })).not.toHaveAttribute(
    "src",
    expect.stringMatching(/^blob:/),
  );
  view.unmount();
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
  await userEvent.click(screen.getByRole("button", { name: "生成临时预览" }));
  await waitFor(() => expect(postSignal).toBeDefined());

  view.unmount();
  expect(postSignal?.aborted).toBe(true);
  await act(async () => {
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
  expect(URL.createObjectURL).toHaveBeenCalledTimes(createdBeforePreview);
});
