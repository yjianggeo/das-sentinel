export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

let csrfToken = "";
let unauthorizedHandler: (() => void) | null = null;
let unauthorizedGeneration = 0;

export function setCsrfToken(token: string): void {
  csrfToken = token;
}

export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler;
  unauthorizedGeneration += 1;
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  params?: Record<string, string>,
  signal?: AbortSignal,
): Promise<T> {
  const requestUnauthorizedGeneration = unauthorizedGeneration;
  const url = new URL(path, window.location.origin);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      url.searchParams.set(key, value);
    }
  }
  const headers: Record<string, string> = {};
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  if (method !== "GET") {
    headers["X-CSRF-Token"] = csrfToken;
  }
  const response = await fetch(url, {
    method,
    headers,
    credentials: "same-origin",
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  signal?.throwIfAborted();
  if (response.status === 204) {
    setCsrfToken(response.headers.get("x-csrf-token") ?? csrfToken);
    return undefined as T;
  }
  if (!response.ok) {
    throw await apiErrorFromResponse(response, requestUnauthorizedGeneration);
  }
  return (await response.json()) as T;
}

async function requestBlob(path: string, signal?: AbortSignal): Promise<Blob> {
  const requestUnauthorizedGeneration = unauthorizedGeneration;
  const response = await fetch(new URL(path, window.location.origin), {
    method: "GET",
    credentials: "same-origin",
    signal,
  });
  if (!response.ok) {
    throw await apiErrorFromResponse(response, requestUnauthorizedGeneration);
  }
  return response.blob();
}

async function apiErrorFromResponse(
  response: Response,
  requestUnauthorizedGeneration: number,
): Promise<ApiError> {
  let detail = response.statusText;
  try {
    const payload = (await response.json()) as { detail?: string };
    detail = payload.detail ?? detail;
  } catch {
    // keep statusText
  }
  if (
    response.status === 401
    && requestUnauthorizedGeneration === unauthorizedGeneration
  ) {
    unauthorizedHandler?.();
  }
  return new ApiError(response.status, detail);
}

export const api = {
  get: <T>(path: string, params?: Record<string, string>, signal?: AbortSignal) =>
    request<T>("GET", path, undefined, params, signal),
  getBlob: (path: string, signal?: AbortSignal) => requestBlob(path, signal),
  post: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
    request<T>("POST", path, body, undefined, signal),
  patch: <T>(path: string, body: unknown) => request<T>("PATCH", path, body),
  delete: <T>(path: string) => request<T>("DELETE", path),
};

export interface SessionInfo {
  admin_id: number;
  csrf_token: string;
}

export interface Dashboard {
  job_counts: Record<string, number>;
  today_event_count: number;
  machine_label_counts: Record<string, number>;
}

export interface ProfileInfo {
  id: string;
  version: number;
  payload: Record<string, unknown>;
  is_current: boolean;
  created_at: string;
}

export interface Site {
  id: string;
  name: string;
  data_path: string;
  timezone: string;
  cable_type: "downhole" | "urban" | "subsea";
  enabled: boolean;
  created_at: string;
  current_profile: ProfileInfo;
}

export interface Job {
  id: string;
  source_file_id: string;
  processing_profile_id: string;
  pipeline_version: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  attempt_count: number;
  progress_numerator: number;
  progress_denominator: number;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface Review {
  id: number;
  label: string;
  note: string | null;
  created_at: string;
}

export interface EventSummary {
  id: string;
  event_uid: string;
  site_id: string;
  site_timezone: string;
  occurred_at: string;
  start_sample: number;
  end_sample: number;
  channel_start: number;
  channel_stop: number;
  machine_label: "earthquake" | "blast" | "traffic" | "noise" | "unknown";
  latest_review: Review | null;
}

export interface EventEvidence {
  detector: string;
  score: number;
  payload: Record<string, unknown>;
}

export interface MachineClassification {
  id: number;
  label: string;
  decision_score: number;
  method: string;
  reasons: string[];
  model_version_id: string | null;
  created_at: string;
}

export interface EventArtifact {
  id: string;
  kind: string;
  media_type: string;
}

export interface WaterfallOptions {
  automatic_distance_start_m: number;
  automatic_distance_stop_m: number;
  available_distance_start_m: number;
  available_distance_stop_m: number;
  primary_detector: string;
  primary_detector_display: string;
  filter_low_hz: number;
  filter_high_hz: number;
  distance_unit: "m";
}

export interface WaterfallPreview {
  distance_start_m: number;
  distance_stop_m: number;
  channel_count: number;
  primary_detector: string;
  primary_detector_display: string;
  filter_low_hz: number;
  filter_high_hz: number;
  raw_png_base64: string;
  filtered_png_base64: string;
}

export interface EventRun {
  id: string;
  source_file_id: string;
  source_fingerprint: string;
  processing_profile_id: string;
  requested_backend: string;
  actual_backend: string;
  backend_fallback_reason: string | null;
  profile_version: number;
  pipeline_version: string;
  model_version_id: string | null;
  timings_ms: Record<string, number>;
  started_at: string | null;
  finished_at: string | null;
}

export interface EventDetail extends EventSummary {
  features: Record<string, number | string>;
  evidence: EventEvidence[];
  machine_history: MachineClassification[];
  review_history: Review[];
  artifacts: EventArtifact[];
  run: EventRun;
}

export interface ModelInfo {
  id: string;
  site_id: string;
  version: string;
  status: "candidate" | "active" | "retired" | "invalid";
  dataset_sha256: string | null;
  bundle_sha256: string;
  metrics: {
    macro_f1?: number;
    recall_by_label?: Record<string, number>;
    precision_by_label?: Record<string, number>;
    confusion_matrix?: number[][];
    rule_baseline_macro_f1?: number;
    reviewed_count_by_label?: Record<string, number>;
    feature_names?: string[];
  } & Record<string, unknown>;
  created_at: string;
  published_at: string | null;
}
