import { useEffect, useRef, useState } from "react";
import {
  api,
  type EventDetail,
  type EventMeasurements,
  type EventMeasurementsResponse,
} from "../api";

const DETECTOR_NAMES: Record<string, string> = {
  stalta: "STA/LTA 多道符合",
  energy: "多频带能量统计",
  coherence: "去偏相干性",
  trajectory: "车辆轨迹",
  template: "模板匹配",
};

const METHOD_NAMES: Record<string, string> = {
  rules: "规则判断",
  model: "模型判断",
  fusion: "融合判断",
};

const LABEL_NAMES: Record<string, string> = {
  earthquake: "地震",
  blast: "爆破",
  traffic: "车辆",
  noise: "噪声",
  unknown: "待定",
};

const FEATURE_NAMES: Record<string, string> = {
  duration_s: "持续时间（s）",
  rise_time_s: "上升时间（s）",
  pulse_count: "脉冲次数",
  pulse_interval_s: "平均脉冲间隔（s）",
  time_of_day_hours: "事件发生时刻（当地小时）",
  dominant_frequency_hz: "主频（Hz）",
  bandwidth_hz: "频谱带宽（标准差，Hz）",
  band_energy_ratio_low_mid: "低频／中频能量比",
  band_energy_ratio_mid_high: "中频／高频能量比",
  spectral_centroid_hz: "频谱质心（Hz）",
  spectral_entropy: "归一化频谱熵",
  distance_extent_m: "距离跨度（m）",
  energy_centroid_m: "能量重心位置（m）",
  shallow_deep_energy_ratio: "小距离段／大距离段能量比",
  continuity: "空间活跃通道占比",
  best_slowness_s_per_m: "最佳慢度（s/m）",
  apparent_velocity_mps: "视速度（m/s）",
  direction: "传播方向（符号）",
  coherence: "相干性",
  zero_slowness_flag: "零慢度标记",
  rms: "均方根振幅",
  kurtosis: "峰度",
  peak_median_ratio: "峰值／平均绝对幅值比",
  cross_channel_correlation: "通道间平均相关系数",
  repetition_rate_hz: "脉冲重复频率（Hz）",
  site_id: "站点标识",
  cable_type: "光缆类型",
  device_config: "设备配置",
};

interface MeasurementDefinition {
  name: string;
  unit?: string;
  digits?: number;
}

const MEASUREMENT_DEFINITIONS: Record<string, MeasurementDefinition> = {
  input_rms: { name: "带通均方根振幅" },
  snr_db: { name: "信噪比", unit: "dB", digits: 2 },
  channel_snr_median_db: { name: "通道信噪比中位数", unit: "dB", digits: 2 },
  channel_snr_p90_db: { name: "通道信噪比第 90 百分位", unit: "dB", digits: 2 },
  valid_channel_count: { name: "有效通道数", digits: 0 },
  valid_channel_fraction: { name: "有效通道占比", digits: 3 },
  bad_channel_fraction: { name: "异常通道占比", digits: 3 },
  clipping_fraction: { name: "削顶占比", digits: 3 },
  trigger_duration_s: { name: "触发持续时间", unit: "s", digits: 3 },
  energy_duration_5_95_s: { name: "能量 5%–95% 持续时间", unit: "s", digits: 3 },
  rise_time_s: { name: "上升时间", unit: "s", digits: 3 },
  pulse_count: { name: "脉冲数", digits: 0 },
  pulse_interval_s: { name: "脉冲间隔", unit: "s", digits: 3 },
  dominant_frequency_hz: { name: "主频", unit: "Hz", digits: 2 },
  spectral_centroid_hz: { name: "频谱质心", unit: "Hz", digits: 2 },
  spectral_entropy: { name: "频谱熵", digits: 3 },
  frequency_resolution_hz: { name: "频率分辨率", unit: "Hz", digits: 3 },
  band_fraction_low: { name: "低频能量占比", digits: 3 },
  band_fraction_mid: { name: "中频能量占比", digits: 3 },
  band_fraction_high: { name: "高频能量占比", digits: 3 },
  active_channel_fraction: { name: "活跃通道占比", digits: 3 },
  longest_active_span_m: { name: "最长活跃跨度", unit: "m", digits: 1 },
  distance_extent_m: { name: "距离跨度", unit: "m", digits: 1 },
  lagged_correlation: { name: "时移相关系数", digits: 3 },
  slowness_s_per_m: { name: "慢度", unit: "s/m", digits: 6 },
  apparent_velocity_mps: { name: "视速度", unit: "m/s", digits: 1 },
  direction: { name: "传播方向", digits: 0 },
  moveout_residual_s: { name: "时差拟合残差", unit: "s", digits: 4 },
};

const MEASUREMENT_GROUPS = [
  {
    name: "信号与质量",
    fields: [
      "input_rms",
      "snr_db",
      "channel_snr_median_db",
      "channel_snr_p90_db",
      "valid_channel_count",
      "valid_channel_fraction",
      "bad_channel_fraction",
      "clipping_fraction",
    ],
  },
  {
    name: "时间结构",
    fields: [
      "trigger_duration_s",
      "energy_duration_5_95_s",
      "rise_time_s",
      "pulse_count",
      "pulse_interval_s",
    ],
  },
  {
    name: "频谱",
    fields: [
      "dominant_frequency_hz",
      "spectral_centroid_hz",
      "spectral_entropy",
      "frequency_resolution_hz",
      "band_fraction_low",
      "band_fraction_mid",
      "band_fraction_high",
    ],
  },
  {
    name: "空间与传播",
    fields: [
      "active_channel_fraction",
      "longest_active_span_m",
      "distance_extent_m",
      "lagged_correlation",
      "slowness_s_per_m",
      "apparent_velocity_mps",
      "direction",
      "moveout_residual_s",
    ],
  },
] as const;

const STATUS_NAMES: Record<string, string> = {
  adc_limits_unavailable: "缺少 ADC 量程",
  insufficient_background: "事件前背景不足",
  zero_background_energy: "背景能量为零",
  constant_envelope: "包络无有效变化",
  single_pulse: "仅识别到一个脉冲",
  rise_not_resolved: "上升过程不可分辨",
  no_supported_pulse: "未识别到可信脉冲",
  no_event_energy: "事件能量不足",
  insufficient_frequency_resolution: "频率分辨率不足",
  insufficient_low_frequency_resolution: "低频分辨率不足",
  no_spectral_energy: "频谱能量不足",
  insufficient_spectral_bins: "有效频率点不足",
  band_outside_nyquist: "频带超出奈奎斯特范围",
  no_active_channels: "未识别到活跃通道",
  insufficient_active_channels: "活跃通道不足",
  insufficient_lag_overlap: "可用于时移比较的窗口过短",
  no_local_channel_pairs: "缺少局部通道对",
  insufficient_correlated_pairs: "相关通道对不足",
  lag_peak_at_search_boundary: "相关峰达到时移搜索边界",
  insufficient_nonzero_lags: "可分辨的传播时差数量不足",
  band_unresolved_at_frequency_resolution: "当前分辨率无法解析该频带",
  invalid_channel_geometry: "通道距离信息无效",
  rise_start_not_observed: "未完整观察到上升起点",
  no_spatial_extent: "有效空间跨度不足",
  unresolved_zero_lag: "传播时差不可分辨",
  inconsistent_moveout: "传播时差不一致",
  empty_channel_window: "通道窗口为空",
  no_valid_channels: "没有有效通道",
  empty_event_window: "事件窗口为空",
  frequency_band_unavailable: "有效频带不可用",
  nonfinite_result: "计算结果不是有限数值",
};

function isAbortError(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === "AbortError";
}

function HistoricalFeatures({ event }: { event: EventDetail }) {
  return (
    <table className="data">
      <tbody>
        {Object.entries(event.features).map(([name, value]) => (
          <tr key={name}>
            <th scope="row">{FEATURE_NAMES[name] ?? name}</th>
            <td>{typeof value === "number" ? value.toPrecision(4) : String(value)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MeasurementRows({
  measurements,
  fields,
}: {
  measurements: EventMeasurements;
  fields: readonly string[];
}) {
  return (
    <table className="data">
      <tbody>
        {fields.map((name) => (
          <tr key={name}>
            <th scope="row">{MEASUREMENT_DEFINITIONS[name]?.name ?? name}</th>
            <td>
              {formatMeasurement(
                name,
                measurements.values[name],
                measurements.status[name],
                measurements.context,
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MeasurementQuality({ context }: { context: Record<string, unknown> }) {
  const flags = Array.isArray(context.quality_flags)
    ? context.quality_flags.filter((flag): flag is string => typeof flag === "string")
    : [];
  const messages: string[] = [];
  if (flags.includes("coarse_frequency_resolution")) {
    messages.push("事件窗口较短，频率分辨率较粗");
  }
  if (flags.includes("incomplete_post_event_filter_guard")) {
    const available = finiteContextNumber(
      context.post_event_filter_guard_seconds_available,
    );
    const requested = finiteContextNumber(
      context.post_event_filter_guard_seconds_requested,
    );
    messages.push(
      available === null || requested === null
        ? "后置滤波保护不完整"
        : `后置滤波保护不完整：实际 ${available.toFixed(3)} s，目标 ${requested.toFixed(3)} s`,
    );
  }
  if (messages.length === 0) return null;
  return (
    <section>
      <h3>测量质量提示</h3>
      <ul>
        {messages.map((message) => <li key={message}>{message}</li>)}
      </ul>
    </section>
  );
}

function MeasurementDescription({ measurements }: { measurements: EventMeasurements }) {
  const context = measurements.context;
  const algorithm = typeof context.algorithm_version === "string"
    ? context.algorithm_version
    : null;
  const band = Array.isArray(context.effective_frequency_band_hz)
    ? context.effective_frequency_band_hz.map(finiteContextNumber)
    : [];
  const samplingRate = finiteContextNumber(context.sampling_rate_hz);
  const background = finiteContextNumber(context.background_seconds_available);
  const inputUnit = typeof context.input_unit === "string" && context.input_unit
    ? context.input_unit
    : null;
  const commonMode = context.common_mode_treatment === "none" ? "未去共模" : null;
  const spectralBands = context.effective_spectral_bands_hz;
  const bandItems = spectralBands && typeof spectralBands === "object" && !Array.isArray(spectralBands)
    ? Object.entries(spectralBands).flatMap(([key, bounds]) => {
      const names: Record<string, string> = { low: "低频范围", mid: "中频范围", high: "高频范围" };
      if (!names[key]) return [];
      const limits = Array.isArray(bounds) ? bounds.map(finiteContextNumber) : [];
      return [{ name: names[key], value: limits.length === 2 && limits[0] !== null && limits[1] !== null
        ? `${limits[0].toFixed(2)}–${limits[1].toFixed(2)} Hz` : "频带不可用" }];
    }) : [];
  const items = [
    {
      name: "版本",
      value: algorithm
        ? `Schema ${measurements.schema_version} / ${algorithm}`
        : `Schema ${measurements.schema_version}`,
    },
    {
      name: "实际频带",
      value: band.length === 2 && band[0] !== null && band[1] !== null
        ? `${band[0].toFixed(2)}–${band[1].toFixed(2)} Hz`
        : null,
    },
    {
      name: "采样率",
      value: samplingRate === null ? null : `${samplingRate.toFixed(2)} Hz`,
    },
    {
      name: "事件前背景",
      value: background === null ? null : `${background.toFixed(3)} s`,
    },
    { name: "输入单位", value: inputUnit },
    { name: "共模处理", value: commonMode },
    ...bandItems,
  ].filter((item): item is { name: string; value: string } => item.value !== null);
  return (
    <section>
      <h3>测量说明</h3>
      <dl>
        {items.map((item) => (
          <div key={item.name}>
            <dt>{item.name}</dt>
            <dd>{item.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function finiteContextNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatMeasurement(
  name: string,
  value: number | null,
  status: string | undefined,
  context: Record<string, unknown>,
): string {
  if (value === null || value === undefined) {
    const reason = status
      ? STATUS_NAMES[status] ?? `原因：${status}`
      : "未提供原因";
    return `不可用（${reason}）`;
  }
  if (name === "input_rms") {
    const unit = typeof context.input_unit === "string" && context.input_unit
      ? ` ${context.input_unit}`
      : "";
    return `${value.toPrecision(4)}${unit}`;
  }
  const definition = MEASUREMENT_DEFINITIONS[name];
  const rendered = value.toFixed(definition?.digits ?? 4);
  return definition?.unit ? `${rendered} ${definition.unit}` : rendered;
}

function MeasurementTables({ measurements }: { measurements: EventMeasurements }) {
  const known = new Set<string>(MEASUREMENT_GROUPS.flatMap((group) => [...group.fields]));
  const extra = Object.keys(measurements.values).filter((name) => !known.has(name));
  return (
    <>
      {MEASUREMENT_GROUPS.map((group) => {
        const fields = group.fields.filter((name) => name in measurements.values);
        if (fields.length === 0) return null;
        return (
          <section key={group.name}>
            <h3>{group.name}</h3>
            <MeasurementRows measurements={measurements} fields={fields} />
          </section>
        );
      })}
      {extra.length > 0 && (
        <section>
          <h3>其他</h3>
          <MeasurementRows measurements={measurements} fields={extra} />
        </section>
      )}
      <MeasurementDescription measurements={measurements} />
      <MeasurementQuality context={measurements.context} />
    </>
  );
}

export function EventFeatures({ event }: { event: EventDetail }) {
  const [measurements, setMeasurements] = useState<{
    eventId: string;
    value: EventMeasurements;
  } | null>(null);
  const [activeTab, setActiveTab] = useState<"history" | "measurements">("history");
  const requestSequence = useRef(0);
  const currentEventId = useRef(event.id);
  currentEventId.current = event.id;

  useEffect(() => {
    const controller = new AbortController();
    const sequence = ++requestSequence.current;
    const requestedEventId = event.id;
    setMeasurements(null);
    setActiveTab("history");
    api.get<EventMeasurementsResponse>(
      `/api/v1/events/${event.id}/measurements`,
      undefined,
      controller.signal,
    ).then((response) => {
      if (
        sequence !== requestSequence.current
        || requestedEventId !== currentEventId.current
        || !response.available
        || response.measurements.schema_version !== "2"
      ) return;
      setMeasurements({ eventId: requestedEventId, value: response.measurements });
    }).catch((cause: unknown) => {
      if (isAbortError(cause) || sequence !== requestSequence.current) return;
      setMeasurements(null);
    });
    return () => controller.abort();
  }, [event.id]);

  const available = measurements?.eventId === event.id ? measurements.value : null;
  const showingMeasurements = available !== null && activeTab === "measurements";
  return (
    <section className="event-features" aria-label="事件特征" tabIndex={0}>
      <h2 className="event-panel-heading">事件特征</h2>
      <div className="artifact-tabs">
        <div className="artifact-tablist" role="tablist" aria-label="事件特征版本">
          <button
            id="event-features-history-tab"
            type="button"
            role="tab"
            aria-controls="event-features-history-panel"
            aria-selected={!showingMeasurements}
            className={!showingMeasurements ? "tab active" : "tab"}
            onClick={() => setActiveTab("history")}
          >
            历史特征
          </button>
          {available && (
            <button
              id="event-features-measurements-tab"
              type="button"
              role="tab"
              aria-controls="event-features-measurements-panel"
              aria-selected={showingMeasurements}
              className={showingMeasurements ? "tab active" : "tab"}
              onClick={() => setActiveTab("measurements")}
            >
              新版测量
            </button>
          )}
        </div>
      </div>
      <div
        id="event-features-history-panel"
        role="tabpanel"
        aria-labelledby="event-features-history-tab"
        hidden={showingMeasurements}
      >
        <HistoricalFeatures event={event} />
      </div>
      {available && (
        <div
          id="event-features-measurements-panel"
          role="tabpanel"
          aria-labelledby="event-features-measurements-tab"
          hidden={!showingMeasurements}
        >
          <MeasurementTables measurements={available} />
        </div>
      )}
    </section>
  );
}

export default function EvidencePanel({ event }: { event: EventDetail }) {
  const latestMachine = event.machine_history[event.machine_history.length - 1];
  return (
    <div className="evidence-panel">
      <section>
        <h3>检测证据</h3>
        <ul>
          {event.evidence.map((item, index) => (
            <li key={`${item.detector}-${index}`}>
              {DETECTOR_NAMES[item.detector] ?? item.detector}：
              得分 {item.score.toFixed(2)}
            </li>
          ))}
          {event.evidence.length === 0 && <li>无检测证据</li>}
        </ul>
      </section>
      <section>
        <h3>{latestMachine ? METHOD_NAMES[latestMachine.method] ?? latestMachine.method : "规则判断"}</h3>
        {latestMachine ? (
          <>
            <p>
              机器标签：
              {LABEL_NAMES[latestMachine.label] ?? latestMachine.label}（置信{" "}
              {latestMachine.decision_score.toFixed(2)}）
            </p>
            <ul>
              {latestMachine.reasons.map((reason, index) => (
                <li key={index}>{reason}</li>
              ))}
            </ul>
          </>
        ) : (
          <p>暂无机器分类</p>
        )}
      </section>
    </div>
  );
}
