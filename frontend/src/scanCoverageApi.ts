import { api } from "./api";

export interface ScanCoverage {
    year: number;
    timezone: "Asia/Shanghai";
    available_years: number[];
    days: { date: string; scanned_seconds: number }[];
    total_scanned_seconds: number;
    scanned_days: number;
}

export interface ScanCoverageDay {
    date: string;
    timezone: "Asia/Shanghai";
    scanned_seconds: number;
    intervals: { start_at: string; end_at: string }[];
}

export const scanCoverageApi = {
    year: (year?: number, siteId = "", signal?: AbortSignal) => {
        const params: Record<string, string> = {};
        if (year !== undefined) params.year = String(year);
        if (siteId) params.site_id = siteId;
        return api.get<ScanCoverage>("/api/v1/scan-coverage", params, signal);
    },
    day: (date: string, siteId = "", signal?: AbortSignal) => {
        const params: Record<string, string> = { date };
        if (siteId) params.site_id = siteId;
        return api.get<ScanCoverageDay>("/api/v1/scan-coverage/day", params, signal);
    },
};
