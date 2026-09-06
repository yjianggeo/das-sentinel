import type { ScanStatus } from "../api";

type Props = {
  scan: ScanStatus;
};

export function isActiveScan(scan: ScanStatus): boolean {
  return (
    scan.phase === "initial_scan"
    || scan.phase === "stability_wait"
    || scan.phase === "verification_scan"
    || scan.phase === "enqueueing"
  );
}

function statusMessage(scan: ScanStatus): string {
  switch (scan.phase) {
    case "initial_scan":
      return `正在扫描：已检查 ${scan.initial_checked_files} 个文件，发现 ${scan.initial_found_files} 个 H5 数据文件`;
    case "stability_wait":
      return `请等待稳定性复检，剩余 ${scan.stability_seconds_remaining} 秒，无需再次点击`;
    case "verification_scan":
      return `正在进行稳定性复检：已复检 ${scan.verification_checked_files} 个文件，确认 ${scan.stable_files} 个稳定文件`;
    case "enqueueing":
      return "正在为稳定文件创建分析任务";
    case "completed":
      if (scan.initial_found_files === 0 && scan.verification_found_files === 0) {
        return "未发现可处理的 H5 数据文件";
      }
      return `扫描完成：首轮发现 ${scan.initial_found_files} 个，复检发现 ${scan.verification_found_files} 个，稳定 ${scan.stable_files} 个，已提交 ${scan.submitted_files} 个，未入队 ${scan.skipped_files} 个`;
    case "failed":
      return scan.error_message ?? "扫描失败";
  }
}

export default function ScanStatusCard({ scan }: Props) {
  return (
    <div className="scan-status-card">
      <p role={scan.phase === "failed" ? "alert" : "status"}>
        {statusMessage(scan)}
      </p>
    </div>
  );
}
