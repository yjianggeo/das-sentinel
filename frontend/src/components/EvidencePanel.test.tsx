import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, vi } from "vitest";
import {
  api,
  type EventDetail,
  type EventMeasurementsResponse,
} from "../api";
import { EventFeatures } from "./EvidencePanel";


const EVENT: EventDetail = {
  id: "event-1",
  event_uid: "uid-1",
  site_id: "site-1",
  site_timezone: "Asia/Shanghai",
  occurred_at: "2026-09-10T00:00:00Z",
  start_sample: 100,
  end_sample: 200,
  channel_start: 10,
  channel_stop: 20,
  machine_label: "unknown",
  latest_review: null,
  features: { rms: 1.5, duration_s: 2.0 },
  evidence: [],
  machine_history: [],
  review_history: [],
  artifacts: [],
  run: {
    id: "run-1",
    source_file_id: "source-1",
    source_fingerprint: "a".repeat(64),
    processing_profile_id: "profile-1",
    requested_backend: "cpu",
    actual_backend: "cpu",
    backend_fallback_reason: null,
    profile_version: 1,
    pipeline_version: "pipeline-v1",
    model_version_id: null,
    timings_ms: {},
    started_at: null,
    finished_at: null,
  },
};

function available(
  values: Record<string, number | null>,
  status: Record<string, string>,
  context: Record<string, unknown> = {},
): EventMeasurementsResponse {
  return {
    available: true,
    measurements: {
      schema_version: "2",
      values,
      status,
      context,
    },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

afterEach(() => {
  vi.restoreAllMocks();
});


it("shows grouped v2 measurements with Chinese names, units, and missing reasons", async () => {
  vi.spyOn(api, "get").mockResolvedValue(
    available(
      {
        input_rms: 0.01345,
        snr_db: 12.345,
        trigger_duration_s: 2.5,
        dominant_frequency_hz: 10.0,
        distance_extent_m: 180.0,
        slowness_s_per_m: null,
        clipping_fraction: null,
      },
      {
        input_rms: "measured",
        snr_db: "measured",
        trigger_duration_s: "measured",
        dominant_frequency_hz: "measured",
        distance_extent_m: "measured",
        slowness_s_per_m: "insufficient_lag_overlap",
        clipping_fraction: "adc_limits_unavailable",
      },
      {
        algorithm_version: "measurements-v2.0",
        effective_frequency_band_hz: [0.5, 50.0],
        sampling_rate_hz: 100.0,
        input_unit: "strain_rate",
        background_seconds_available: 4.5,
        common_mode_treatment: "none",
        quality_flags: [
          "coarse_frequency_resolution",
          "incomplete_post_event_filter_guard",
        ],
        post_event_filter_guard_seconds_requested: 2.0,
        post_event_filter_guard_seconds_available: 0.5,
      },
    ),
  );
  render(<EventFeatures event={EVENT} />);

  const newMeasurements = await screen.findByRole("tab", { name: "新版测量" });
  expect(screen.getByRole("tab", { name: "历史特征" })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await userEvent.click(newMeasurements);

  const panel = screen.getByRole("tabpanel", { name: "新版测量" });
  expect(within(panel).getByRole("heading", { name: "信号与质量" })).toBeVisible();
  expect(
    within(panel).getByRole("row", {
      name: "带通均方根振幅 0.01345 strain_rate",
    }),
  ).toBeVisible();
  expect(within(panel).getByRole("row", { name: "信噪比 12.35 dB" })).toBeVisible();
  expect(within(panel).getByRole("heading", { name: "时间结构" })).toBeVisible();
  expect(within(panel).getByRole("row", { name: "触发持续时间 2.500 s" })).toBeVisible();
  expect(within(panel).getByRole("heading", { name: "频谱" })).toBeVisible();
  expect(within(panel).getByRole("row", { name: "主频 10.00 Hz" })).toBeVisible();
  expect(within(panel).getByRole("heading", { name: "空间与传播" })).toBeVisible();
  expect(within(panel).getByRole("row", { name: "距离跨度 180.0 m" })).toBeVisible();
  expect(
    within(panel).getByRole("row", {
      name: "慢度 不可用（可用于时移比较的窗口过短）",
    }),
  ).toBeVisible();
  expect(
    within(panel).getByRole("row", {
      name: "削顶占比 不可用（缺少 ADC 量程）",
    }),
  ).toBeVisible();
  expect(within(panel).getByRole("heading", { name: "测量质量提示" })).toBeVisible();
  expect(within(panel).getByText("事件窗口较短，频率分辨率较粗")).toBeVisible();
  expect(
    within(panel).getByText("后置滤波保护不完整：实际 0.500 s，目标 2.000 s"),
  ).toBeVisible();
  expect(within(panel).getByRole("heading", { name: "测量说明" })).toBeVisible();
  expect(within(panel).getByText("Schema 2 / measurements-v2.0")).toBeVisible();
  expect(within(panel).getByText("0.50–50.00 Hz")).toBeVisible();
  expect(within(panel).getByText("100.00 Hz")).toBeVisible();
  expect(within(panel).getByText("4.500 s")).toBeVisible();
  expect(within(panel).getByText("strain_rate")).toBeVisible();
  expect(within(panel).getByText("未去共模")).toBeVisible();
});


it("keeps historical features when measurements are unavailable or fail", async () => {
  const get = vi.spyOn(api, "get").mockResolvedValueOnce({ available: false });
  const view = render(<EventFeatures event={EVENT} />);

  expect(await screen.findByRole("row", { name: "均方根振幅 1.500" })).toBeVisible();
  expect(screen.queryByRole("tab", { name: "新版测量" })).not.toBeInTheDocument();

  get.mockRejectedValueOnce(new Error("offline"));
  view.rerender(<EventFeatures event={{ ...EVENT, id: "event-2" }} />);

  expect(await screen.findByRole("row", { name: "均方根振幅 1.500" })).toBeVisible();
  expect(screen.queryByRole("tab", { name: "新版测量" })).not.toBeInTheDocument();
});


it("renders a Chinese fallback when a null measurement has no status", async () => {
  vi.spyOn(api, "get").mockResolvedValue(available({ input_rms: null }, {}));
  render(<EventFeatures event={EVENT} />);
  await userEvent.click(await screen.findByRole("tab", { name: "新版测量" }));

  expect(
    screen.getByRole("row", { name: "带通均方根振幅 不可用（未提供原因）" }),
  ).toBeVisible();
});


it("ignores a stale measurement response after the selected event changes", async () => {
  const first = deferred<EventMeasurementsResponse>();
  let firstSignal: AbortSignal | undefined;
  vi.spyOn(api, "get").mockImplementation((path, _params, signal) => {
    if (path.endsWith("event-1/measurements")) {
      firstSignal = signal;
      return first.promise;
    }
    return Promise.resolve(available({ input_rms: 22.0 }, { input_rms: "measured" }));
  });
  const view = render(<EventFeatures event={EVENT} />);

  view.rerender(<EventFeatures event={{ ...EVENT, id: "event-2" }} />);
  expect(firstSignal?.aborted).toBe(true);
  await userEvent.click(await screen.findByRole("tab", { name: "新版测量" }));
  expect(screen.getByRole("row", { name: "带通均方根振幅 22.00" })).toBeVisible();

  await act(async () => {
    first.resolve(available({ input_rms: 11.0 }, { input_rms: "measured" }));
    await first.promise;
  });

  expect(screen.getByRole("row", { name: "带通均方根振幅 22.00" })).toBeVisible();
  expect(
    screen.queryByRole("row", { name: "带通均方根振幅 11.00" }),
  ).not.toBeInTheDocument();
});
