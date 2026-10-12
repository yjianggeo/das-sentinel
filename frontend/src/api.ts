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
  put: <T>(path: string, body: unknown) => request<T>("PUT", path, body),
  patch: <T>(path: string, body: unknown) => request<T>("PATCH", path, body),
  delete: <T>(path: string, signal?: AbortSignal) =>
    request<T>("DELETE", path, undefined, undefined, signal),
};

export interface SessionInfo {
  admin_id: number;
  csrf_token: string;
}

export interface Dashboard {
  window_status_counts: Record<string, number>;
  model_scan_status_counts: Record<string, number>;
  review_counts: { earthquake: number; noise: number; pending: number };
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

export interface ScanDirectoryEntry {
  name: string;
  path: string;
}

export interface ScanDirectoryListing {
  site_id: string;
  site_name: string;
  path: string;
  parent_path: string | null;
  directories: ScanDirectoryEntry[];
}

export type ScanPhase =
  | "initial_scan"
  | "stability_wait"
  | "verification_scan"
  | "enqueueing"
  | "completed"
  | "failed";

export interface ScanStatus {
  id: string;
  site_id: string;
  path: string;
  phase: ScanPhase;
  initial_checked_files: number;
  initial_found_files: number;
  verification_checked_files: number;
  verification_found_files: number;
  stable_files: number;
  submitted_files: number;
  skipped_files: number;
  stability_seconds_remaining: number;
  error_message: string | null;
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
  detection_score: number | null;
  latest_review: Review | null;
}
