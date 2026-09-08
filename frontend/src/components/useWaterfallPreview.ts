import { useCallback, useEffect, useRef, useState } from "react";
import { api, type WaterfallOptions, type WaterfallPreview } from "../api";

export interface WaterfallPreviewUrls {
  waterfall_raw: string;
  waterfall_filtered: string;
  waveform: string;
}

function revokePreview(preview: WaterfallPreviewUrls | null): void {
  if (!preview) return;
  URL.revokeObjectURL(preview.waterfall_raw);
  URL.revokeObjectURL(preview.waterfall_filtered);
  URL.revokeObjectURL(preview.waveform);
}

function decodePng(base64: string): Blob {
  const decoded = atob(base64);
  const bytes = new Uint8Array(decoded.length);
  for (let index = 0; index < decoded.length; index += 1) {
    bytes[index] = decoded.charCodeAt(index);
  }
  return new Blob([bytes], { type: "image/png" });
}

function previewUrls(value: WaterfallPreview): WaterfallPreviewUrls {
  let rawUrl: string | null = null;
  let filteredUrl: string | null = null;
  let waveformUrl: string | null = null;
  try {
    rawUrl = URL.createObjectURL(decodePng(value.raw_png_base64));
    filteredUrl = URL.createObjectURL(decodePng(value.filtered_png_base64));
    waveformUrl = URL.createObjectURL(decodePng(value.waveform_png_base64));
    return {
      waterfall_raw: rawUrl,
      waterfall_filtered: filteredUrl,
      waveform: waveformUrl,
    };
  } catch (cause) {
    if (rawUrl) URL.revokeObjectURL(rawUrl);
    if (filteredUrl) URL.revokeObjectURL(filteredUrl);
    if (waveformUrl) URL.revokeObjectURL(waveformUrl);
    throw cause;
  }
}

export function useWaterfallPreview(eventId: string): {
  options: WaterfallOptions | null;
  optionsError: string;
  preview: WaterfallPreviewUrls | null;
  busy: boolean;
  error: string;
  generate(startM?: number, stopM?: number): Promise<void>;
  reset(): void;
} {
  const [options, setOptions] = useState<WaterfallOptions | null>(null);
  const [optionsError, setOptionsError] = useState("");
  const [preview, setPreview] = useState<WaterfallPreviewUrls | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const controllerRef = useRef<AbortController | null>(null);
  const sequenceRef = useRef(0);
  const previewRef = useRef<WaterfallPreviewUrls | null>(null);

  const replacePreview = useCallback((next: WaterfallPreviewUrls | null) => {
    revokePreview(previewRef.current);
    previewRef.current = next;
    setPreview(next);
  }, []);

  useEffect(() => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const sequence = ++sequenceRef.current;
    replacePreview(null);
    setOptions(null);
    setOptionsError("");
    setError("");
    setBusy(false);

    api
      .get<WaterfallOptions>(
        `/api/v1/events/${eventId}/waterfall-options`,
        undefined,
        controller.signal,
      )
      .then((value) => {
        if (sequence === sequenceRef.current && !controller.signal.aborted) {
          setOptions(value);
        }
      })
      .catch(() => {
        if (sequence === sequenceRef.current && !controller.signal.aborted) {
          setOptionsError("无法加载瀑布图范围");
        }
      });

    return () => {
      controller.abort();
      controllerRef.current?.abort();
      controllerRef.current = null;
      sequenceRef.current += 1;
      replacePreview(null);
    };
  }, [eventId, replacePreview]);

  const generate = useCallback(async (startM?: number, stopM?: number) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const sequence = ++sequenceRef.current;
    setBusy(true);
    setError("");
    try {
      const value = await api.post<WaterfallPreview>(
        `/api/v1/events/${eventId}/waterfall-preview`,
        startM === undefined && stopM === undefined
          ? {}
          : { distance_start_m: startM, distance_stop_m: stopM },
        controller.signal,
      );
      if (sequence !== sequenceRef.current || controller.signal.aborted) return;
      const next = previewUrls(value);
      if (sequence !== sequenceRef.current || controller.signal.aborted) {
        revokePreview(next);
        return;
      }
      replacePreview(next);
    } catch (cause) {
      if (sequence === sequenceRef.current && !controller.signal.aborted) {
        setError("无法生成临时预览");
      }
    } finally {
      if (sequence === sequenceRef.current && !controller.signal.aborted) {
        setBusy(false);
      }
    }
  }, [eventId, replacePreview]);

  const reset = useCallback(() => {
    replacePreview(null);
    setError("");
  }, [replacePreview]);

  return { options, optionsError, preview, busy, error, generate, reset };
}
