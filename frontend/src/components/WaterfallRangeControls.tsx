import { useEffect, useState } from "react";
import type { WaterfallOptions } from "../api";

export default function WaterfallRangeControls({
  options,
  optionsError,
  busy,
  error,
  onGenerate,
  onReset,
}: {
  options: WaterfallOptions | null;
  optionsError: string;
  busy: boolean;
  error: string;
  onGenerate: (startM: number, stopM: number) => Promise<void>;
  onReset: () => void;
}) {
  const [start, setStart] = useState("");
  const [stop, setStop] = useState("");
  const [validationError, setValidationError] = useState("");

  useEffect(() => {
    if (!options) return;
    setStart(String(options.automatic_distance_start_m));
    setStop(String(options.automatic_distance_stop_m));
    setValidationError("");
  }, [options]);

  if (!options) {
    return optionsError ? <p className="error" role="alert">{optionsError}</p> : null;
  }

  const submit = async () => {
    const startM = Number(start);
    const stopM = Number(stop);
    if (!Number.isFinite(startM) || !Number.isFinite(stopM)) {
      setValidationError("请输入有效距离");
      return;
    }
    if (startM >= stopM) {
      setValidationError("起始距离必须小于结束距离");
      return;
    }
    if (
      startM < options.available_distance_start_m
      || stopM > options.available_distance_stop_m
    ) {
      setValidationError("距离必须位于可用范围内");
      return;
    }
    setValidationError("");
    await onGenerate(startM, stopM);
  };

  return (
    <section className="waterfall-range-controls" aria-label="瀑布图距离范围">
      <p>可用 {options.available_distance_start_m}–{options.available_distance_stop_m} {options.distance_unit}</p>
      <p>滤波瀑布图（{options.filter_low_hz}–{options.filter_high_hz} Hz）</p>
      <p>主检测器：{options.primary_detector_display}</p>
      <div className="waterfall-range-row">
        <label>
          起始距离（m）
          <input
            type="number"
            step="any"
            min={options.available_distance_start_m}
            max={options.available_distance_stop_m}
            value={start}
            onChange={(event) => setStart(event.target.value)}
          />
        </label>
        <label>
          结束距离（m）
          <input
            type="number"
            step="any"
            min={options.available_distance_start_m}
            max={options.available_distance_stop_m}
            value={stop}
            onChange={(event) => setStop(event.target.value)}
          />
        </label>
        <button type="button" onClick={submit} disabled={busy} aria-busy={busy}>
          生成临时预览
        </button>
        <button type="button" className="subtle" onClick={onReset} disabled={busy}>
          重置临时预览
        </button>
      </div>
      {busy && <p role="status">正在生成临时预览…</p>}
      {(validationError || error) && <p className="error" role="alert">{validationError || error}</p>}
    </section>
  );
}
