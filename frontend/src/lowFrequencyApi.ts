import { api, type EventSummary } from "./api";

export type MachineLabel = EventSummary["machine_label"];
export interface LowFrequencyReview {
  label: MachineLabel;
  notes: string;
  created_at: string;
}
export interface LowFrequencyCandidate {
  id: string;
  start_utc: string;
  end_utc: string;
  peak_utc: string;
  score: number;
  snr_db: number | null;
  active_channel_fraction: number | null;
  coherence: number | null;
  dominant_frequency_hz: number | null;
  frequency_resolution_hz: number | null;
  energy_duration_s: number | null;
  latest_review: LowFrequencyReview | null;
}
export interface LowFrequencyScanRequest {
  site_id: string;
  relative_path: string;
  start_utc: string;
  end_utc: string;
  distance_start_m: number;
  distance_stop_m: number | null;
}
export interface LowFrequencyScanSummary extends LowFrequencyScanRequest {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  created_at: string;
  error_message: string | null;
}
export type LowFrequencyPlot = "waterfall" | "waveform" | "psd" | "stft";
export interface LowFrequencyResult {
  sample_rate_hz: number;
  distance_start_m: number;
  distance_stop_m: number;
  channel_count: number;
  source_count: number;
  warnings: string[];
  measurements: Record<string, unknown>;
  candidates: LowFrequencyCandidate[];
  plots: Record<LowFrequencyPlot, string>;
}
export interface LowFrequencyScanDetail extends LowFrequencyScanSummary {
  result: LowFrequencyResult | null;
}

const path = "/api/v1/low-frequency/scans";
export const lowFrequencyApi = {
  list: (signal?: AbortSignal) => api.get<LowFrequencyScanSummary[]>(path, undefined, signal),
  create: (body: LowFrequencyScanRequest, signal?: AbortSignal) => api.post<LowFrequencyScanSummary>(path, body, signal),
  detail: (id: string, signal?: AbortSignal) => api.get<LowFrequencyScanDetail>(`${path}/${encodeURIComponent(id)}`, undefined, signal),
  review: (scanId: string, candidateId: string, body: { label: MachineLabel; notes: string; request_id: string }) =>
    api.post<LowFrequencyReview>(`${path}/${encodeURIComponent(scanId)}/candidates/${encodeURIComponent(candidateId)}/reviews`, body),
};
