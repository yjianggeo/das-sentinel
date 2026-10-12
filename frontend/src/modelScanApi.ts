import { api } from "./api";
import type { WindowPage } from "./windowApi";
export const MODEL_SCAN_DIRECTORIES = ["jjk2023", "jjk20230919", "jjk20230926", "jjk20230930", "jjk20240127", "jjk20240416", "jjk20240424", "jjk20240513", "jjk20240521", "jjk20240527", "jjk20240608", "jjk2025_2026"];
export interface ModelScan {
    id: string; site_id: string; status: string;
    model: { id: string; name: string; threshold: number };
    directories: string[]; discovery_complete: boolean; total: number | null;
    counts: Record<string, number>;
    estimate_status: string; remaining_seconds: number | null; estimated_finish_at: string | null;
    errors: { phase: string; code: string; message: string }[]; retryable?: boolean;
}
export const modelScanApi = {
    list: (offset: number, signal: AbortSignal) => api.get<WindowPage<ModelScan>>("/api/v1/model-scans", { limit: "10", offset: String(offset) }, signal),
    action: (id: string, action: "pause" | "resume" | "cancel" | "retry") => api.post<ModelScan>(`/api/v1/model-scans/${id}/${action}`),
    create: (site: string, model: string, directories: string[], requestId: string) => api.post<ModelScan>(`/api/v1/sites/${site}/model-scans`, {
        request_id: requestId, expected_model_id: model, directories, include_rule_skips: true,
    }),
};
