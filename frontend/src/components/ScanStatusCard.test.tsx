import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { ScanPhase, ScanStatus } from "../api";
import ScanStatusCard, { isActiveScan } from "./ScanStatusCard";

const SCAN: ScanStatus = {
  id: "scan-1",
  site_id: "site-1",
  path: "jjk20240608",
  phase: "initial_scan",
  initial_checked_files: 1250,
  initial_found_files: 86,
  verification_checked_files: 64,
  verification_found_files: 60,
  stable_files: 58,
  submitted_files: 55,
  skipped_files: 3,
  stability_seconds_remaining: 8,
  error_message: null,
};

it.each<[ScanPhase, string]>([
  ["initial_scan", "正在扫描：已检查 1250 个文件，发现 86 个 H5 数据文件"],
  ["stability_wait", "请等待稳定性复检，剩余 8 秒，无需再次点击"],
  ["verification_scan", "正在进行稳定性复检：已复检 64 个文件，确认 58 个稳定文件"],
  ["enqueueing", "正在为稳定文件创建分析任务"],
])("renders %s", (phase, expected) => {
  render(<ScanStatusCard scan={{ ...SCAN, phase }} />);

  expect(screen.getByRole("status")).toHaveTextContent(expected);
});

it("renders every completed counter", () => {
  render(<ScanStatusCard scan={{ ...SCAN, phase: "completed" }} />);

  expect(screen.getByRole("status")).toHaveTextContent(
    "扫描完成：首轮发现 86 个，复检发现 60 个，稳定 58 个，已提交 55 个，未入队 3 个",
  );
});

it("renders the empty completed result", () => {
  render(
    <ScanStatusCard
      scan={{
        ...SCAN,
        phase: "completed",
        initial_found_files: 0,
        verification_found_files: 0,
        stable_files: 0,
        submitted_files: 0,
        skipped_files: 0,
      }}
    />,
  );

  expect(screen.getByRole("status")).toHaveTextContent(
    "未发现可处理的 H5 数据文件",
  );
});

it("renders the controlled server failure as an alert", () => {
  render(
    <ScanStatusCard
      scan={{ ...SCAN, phase: "failed", error_message: "扫描目录不可读" }}
    />,
  );

  expect(screen.getByRole("alert")).toHaveTextContent("扫描目录不可读");
  expect(screen.queryByText(SCAN.path)).not.toBeInTheDocument();
});

it.each<[ScanPhase, boolean]>([
  ["initial_scan", true],
  ["stability_wait", true],
  ["verification_scan", true],
  ["enqueueing", true],
  ["completed", false],
  ["failed", false],
])("classifies %s active state", (phase, expected) => {
  expect(isActiveScan({ ...SCAN, phase })).toBe(expected);
});
