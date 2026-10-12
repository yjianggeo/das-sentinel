import { api } from "./api";
export type WindowPresence = "present" | "absent" | "unknown";
export type WindowLabels = { earthquake: WindowPresence };
export type WindowStatus = "pending" | "queued" | "running" | "succeeded" | "failed" | "cancelled";
export interface WindowReview {
    id?: string;
    labels: WindowLabels;
    note: string | null;
    created_at: string;
    stale: boolean;
}
export interface WindowSpatialCoverage {
    algorithm_version: "borehole-coverage-v1" | "borehole-coverage-v2" | "borehole-coverage-v3";
    score: number | null;
    coverage_score?: number | null;
    coverage_fraction: number | null;
    deep_coverage_fraction: number | null;
    continuous_span_m: number | null;
    median_correlation: number | null;
    valid_coverage_fraction: number;
    propagation_slowness_s_per_m?: number | null;
    apparent_velocity_mps?: number | null;
    propagation_residual_s?: number | null;
    propagation_direction?: "+" | "-" | null;
    propagation_confidence?: number | null;
    propagation_status?: "resolved" | "unresolved_zero_lag" | "low_correlation" | "ambiguous_periodic" | "insufficient_support" | "edge_or_boundary";
    common_mode_energy_fraction?: number | null;
    common_mode_channel_fraction?: number | null;
    residual_coverage_score?: number | null;
    common_mode_status?: "component_supported" | "persistent_supported" | "not_supported" | "unknown";
    common_mode_penalty?: number;
    review_protected?: boolean;
    analysis_id: string;
    input_sha256: string;
}
export interface WindowItem {
    spatial_coverage?: WindowSpatialCoverage | null;
    model_recommendation?: { scan_id: string; model_id: string; model_name: string; score: number; threshold: number; analysis_id: string; input_sha256: string } | null;
    window_seconds?: 30 | 60;
    id: string;
    site_id: string;
    site_name: string;
    site_timezone: string;
    start_utc: string;
    end_utc: string;
    status: WindowStatus;
    favorite: boolean;
    score: number | null;
    priority: number;
    legacy_count: number;
    catalog_matches: Record<string, unknown>[];
    analysis_id: string | null;
    input_sha256: string | null;
    latest_review: WindowReview | null;
    model_review_attention?: { kind: "FN" | "FP"; source_review_id: string; note: string } | null;
}
export interface WindowArtifact {
    kind: string;
    url: string;
}
export interface WindowDetectionRegion {
    id: number;
    start_sample: number;
    end_sample: number;
    peak_sample: number;
    channel_start: number;
    channel_stop: number;
    distance_start_m: number;
    distance_stop_m: number;
    score: number;
    detectors: string[];
    evidence_count: number;
}
export interface WindowPlotGeometry {
    axes: [number, number, number, number];
    distance_min_m: number;
    distance_max_m: number;
}
export interface WindowDetail extends WindowItem {
    waterfall_regions_baked?: boolean;
    analysis_error_code?: string | null;
    analysis_error_message?: string | null;
    candidate?: boolean | null;
    visualization_version?: number | null;
    detection_regions?: WindowDetectionRegion[];
    plot_geometry?: WindowPlotGeometry | null;
    legacy_label_import?: {
        state: string;
        labels: WindowLabels;
        history_count: number;
        event_count: number;
    } | null;
    features: Record<string, number | null>;
    recognition_range_m: {
        start: number | null;
        stop: number | null;
    } | null;
    shape: [
        number,
        number
    ] | null;
    unit: string | null;
    machine_labels: unknown[];
    artifacts: WindowArtifact[];
    legacy_total: number;
    legacy_events: {
        id: string;
        start_utc: string;
        channel_start: number;
        channel_stop: number;
        label: string | null;
        note: string | null;
    }[];
}
export interface WindowJob {
    model_scan_id?: string | null;
    site_id: string; site_name: string; site_timezone: string; start_utc: string; end_utc: string; profile_id: string;
    id: string;
    window_id: string;
    status: WindowStatus;
    worker_id: string | null;
    attempt_count: number;
    heartbeat_at: string | null;
    error_code: string | null;
    error_message: string | null;
    created_at: string;
}
export interface WindowPage<T> {
    items: T[];
    total: number;
    limit: number;
    offset: number;
}
export interface WindowReviewCreate {
    analysis_id: string;
    input_sha256: string;
    labels: WindowLabels;
    note: string;
    request_id: string;
}
export interface WindowFavoriteResult {
    window_id: string;
    favorite: boolean;
    created_at: string | null;
}
export const windowStatusNames: Record<WindowStatus, string> = { pending: "待分析", queued: "排队", running: "运行中", succeeded: "成功", failed: "失败", cancelled: "已取消" };
export const windowApi = {
    list: (params: Record<string, string>, signal: AbortSignal) => api.get<WindowPage<WindowItem>>("/api/v1/event-windows", params, signal),
    detail: (id: string, signal: AbortSignal, scanId?: string) => api.get<WindowDetail>(`/api/v1/event-windows/${id}`, scanId ? { model_scan_id: scanId } : undefined, signal),
    review: (id: string, body: WindowReviewCreate) => api.post<WindowReview>(`/api/v1/event-windows/${id}/reviews`, body),
    favorite: (id: string, favorite: boolean) => api.put<WindowFavoriteResult>(`/api/v1/event-windows/${id}/favorite`, { favorite }),
    evidence: (id: string, analysisId: string, offset: number, signal: AbortSignal) => api.get<WindowPage<Record<string, unknown>>>(`/api/v1/event-windows/${id}/evidence`, { analysis_id: analysisId, limit: "50", offset: String(offset) }, signal),
    waterfallPreviewUrl: (id: string, analysisId: string, filtered: boolean) => `/api/v1/event-windows/${encodeURIComponent(id)}/waterfall-preview?${new URLSearchParams({ analysis_id: analysisId, filtered: String(filtered) })}`,
    regionPreviewUrl: (id: string, analysisId: string, regionId: number, filtered: boolean, showRegions = true) => `/api/v1/event-windows/${encodeURIComponent(id)}/region-preview?${new URLSearchParams({ analysis_id: analysisId, region_id: String(regionId), filtered: String(filtered), ...(!showRegions ? { show_regions: "false" } : {}) })}`,
    jobs: (params: Record<string, string>, signal: AbortSignal) => api.get<WindowPage<WindowJob>>("/api/v1/window-jobs", params, signal),
    jobAction: (id: string, action: "cancel" | "retry") => api.post<WindowJob>(`/api/v1/window-jobs/${id}/${action}`),
};
