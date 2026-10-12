export function formatScanDistance(payload: Record<string, unknown>): string {
  const start = typeof payload.scan_distance_start_m === "number"
    ? payload.scan_distance_start_m : null;
  const stop = typeof payload.scan_distance_stop_m === "number"
    ? payload.scan_distance_stop_m : null;
  if (start === null && stop === null) return "全距离（不限）";
  if (start === null) return `至 ${stop} m`;
  if (stop === null) return `从 ${start} m 起`;
  return `${start}–${stop} m`;
}
