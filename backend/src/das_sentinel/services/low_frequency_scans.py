from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import UUID, uuid4, uuid5

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class LowFrequencyScanService:
    """Independent, persistent continuous scans; never pretend a manifest is one source file."""

    def __init__(self, root: Path, data_root: Path):
        self.root = Path(root)
        self.data_root = Path(data_root)
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="low-frequency"
        )
        self._pending = 0
        self._closed = False
        for scan in self.list():
            if scan["status"] in {"queued", "running"}:
                scan.update(
                    status="failed",
                    error_message="服务重启导致扫描中断，请重新发起扫描。",
                )
                self._write(self._folder(scan["id"]) / "scan.json", scan)

    @staticmethod
    def _read(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write(path: Path, value) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, path)

    def _folder(self, scan_id: str) -> Path:
        try:
            if str(UUID(scan_id)) != scan_id:
                raise ValueError
        except (ValueError, AttributeError) as error:
            raise LookupError("扫描不存在") from error
        folder = self.root / scan_id
        if folder.is_symlink() or not folder.resolve().is_relative_to(
            self.root.resolve()
        ):
            raise LookupError("扫描不存在")
        return folder

    def list(self) -> list[dict]:
        with self._lock:
            scans = []
            for path in self.root.glob("*/scan.json"):
                if path.parent.is_symlink():
                    continue
                scans.append(self._read(path))
            return sorted(
                scans, key=lambda scan: (scan["created_at"], scan["id"]), reverse=True
            )

    def get(self, scan_id: str) -> dict:
        with self._lock:
            folder = self._folder(scan_id)
            if not (folder / "scan.json").is_file():
                raise LookupError("扫描不存在")
            scan = self._read(folder / "scan.json")
            result = None
            if scan["status"] == "succeeded":
                result = self._read(folder / "result.json")
                reviews = (
                    self._read(folder / "reviews.json")
                    if (folder / "reviews.json").exists()
                    else []
                )
                latest = {review["candidate_id"]: review for review in reviews}
                for candidate in result["candidates"]:
                    review = latest.get(candidate["id"])
                    candidate["latest_review"] = (
                        None
                        if review is None
                        else {
                            key: review[key] for key in ("label", "notes", "created_at")
                        }
                    )
            return {**scan, "result": result}

    def _directory(self, site_root: Path, relative: str) -> Path:
        relative = relative.replace("\\", "/")
        parts = relative.split("/")
        if (
            not relative
            or any(part in {"", ".", ".."} for part in parts)
            or PureWindowsPath(relative).drive
            or PurePosixPath(relative).is_absolute()
        ):
            raise ValueError("必须选择站点内的相对数据目录")
        root = site_root.resolve(strict=True)
        if not root.is_relative_to(self.data_root.resolve(strict=True)):
            raise ValueError("站点目录超出数据根目录")
        directory = root.joinpath(*parts)
        for parent in (directory, *directory.parents):
            if parent == root:
                break
            if parent.is_symlink() or parent.is_junction():
                raise ValueError("数据目录不能包含链接")
        directory = directory.resolve(strict=True)
        if not directory.is_relative_to(root) or not directory.is_dir():
            raise ValueError("数据目录不在该站点内")
        return directory

    def create(self, payload: dict, site_root: Path) -> dict:
        directory = self._directory(site_root, payload["relative_path"])
        with self._lock:
            if self._closed or self._pending >= 10:
                raise ValueError("低频扫描队列已满，请等待当前扫描完成")
            scan = {
                **payload,
                "id": str(uuid4()),
                "status": "queued",
                "created_at": _now(),
                "error_message": None,
                "algorithm_version": "low-frequency-v1",
            }
            self._write(self._folder(scan["id"]) / "scan.json", scan)
            self._pending += 1
            self._executor.submit(self._run, scan, directory)
            return dict(scan)

    def _run(self, scan: dict, directory: Path) -> None:
        folder = self._folder(scan["id"])
        try:
            with self._lock:
                if self._closed:
                    raise ValueError("服务关闭导致扫描中断，请重新发起扫描。")
                scan.update(status="running")
                self._write(folder / "scan.json", scan)
            result = self._analyze(scan, directory)
            with self._lock:
                self._write(folder / "result.json", result)
                scan.update(status="succeeded")
                self._write(folder / "scan.json", scan)
        except Exception as error:
            logger.exception("Low-frequency scan %s failed", scan["id"])
            with self._lock:
                scan.update(
                    status="failed",
                    error_message=str(error)
                    if isinstance(error, ValueError)
                    else "低频扫描失败，请检查数据可用性和服务日志。",
                )
                self._write(folder / "scan.json", scan)
        finally:
            with self._lock:
                self._pending -= 1

    def _analyze(self, scan: dict, directory: Path) -> dict:
        from das_sentinel.low_frequency import analyze_low_frequency
        from das_sentinel.low_frequency_plots import render_low_frequency_plots
        from das_sentinel.readers.continuous_low_frequency import (
            read_low_frequency_window,
        )

        paths = []
        for path in directory.iterdir():
            if path.suffix.lower() not in {".h5", ".hdf5"}:
                continue
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(
                directory
            ):
                raise ValueError("数据文件链接超出选定目录")
            if path.is_file():
                paths.append(path)
            if len(paths) > 2000:
                raise ValueError("单次目录文件超过2000个，请选择更小的数据目录")
        start = datetime.fromisoformat(scan["start_utc"])
        end = datetime.fromisoformat(scan["end_utc"])
        window = read_low_frequency_window(
            paths, start, end, scan["distance_start_m"], scan["distance_stop_m"]
        )
        start = window.block.metadata.start_time_utc + timedelta(
            seconds=window.analysis_start / window.block.metadata.sample_rate_hz
        )
        analysis = analyze_low_frequency(
            window.block, window.analysis_start, window.analysis_stop
        )
        folder = self._folder(scan["id"])
        plots = render_low_frequency_plots(
            window,
            analysis,
            folder,
            timezone_name=scan.get("site_timezone", "Asia/Shanghai"),
        )
        candidates = []
        for index, candidate in enumerate(analysis.candidates):
            item = dict(candidate)
            for name in ("start", "end", "peak"):
                item[name + "_utc"] = (
                    start + timedelta(seconds=item.pop(name + "_seconds"))
                ).isoformat()
            item["id"] = str(uuid5(UUID(scan["id"]), str(index)))
            candidates.append(item)
        distances = window.block.metadata.channel_distance_m
        return {
            "algorithm_version": scan["algorithm_version"],
            "sample_rate_hz": window.block.metadata.sample_rate_hz,
            "distance_start_m": float(distances.min()),
            "distance_stop_m": float(distances.max()),
            "channel_count": len(distances),
            "source_count": len(window.sources),
            "sources": list(window.sources),
            "warnings": list(dict.fromkeys((*window.warnings, *analysis.warnings))),
            "measurements": analysis.measurements,
            "candidates": candidates,
            "plots": {
                name: f"/api/v1/low-frequency/scans/{scan['id']}/plots/{name}"
                for name in plots
            },
            "plot_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in plots.items()
            },
        }

    def review(
        self, scan_id: str, candidate_id: str, label: str, notes: str, request_id: str
    ) -> dict:
        if label not in {"earthquake", "blast", "traffic", "noise", "unknown"}:
            raise ValueError("人工标签无效")
        with self._lock:
            scan = self.get(scan_id)
            if scan["status"] != "succeeded":
                raise ValueError("扫描尚未完成")
            if not any(
                candidate["id"] == candidate_id
                for candidate in scan["result"]["candidates"]
            ):
                raise LookupError("候选不存在")
            path = self._folder(scan_id) / "reviews.json"
            reviews = self._read(path) if path.exists() else []
            for existing in reviews:
                if existing["request_id"] == request_id:
                    if (
                        existing["candidate_id"],
                        existing["label"],
                        existing["notes"],
                    ) != (candidate_id, label, notes):
                        raise ValueError("重复请求编号的复核内容不一致")
                    return {
                        key: existing[key] for key in ("label", "notes", "created_at")
                    }
            review = {
                "candidate_id": candidate_id,
                "label": label,
                "notes": notes,
                "request_id": request_id,
                "created_at": _now(),
            }
            self._write(path, [*reviews, review])
            return {key: review[key] for key in ("label", "notes", "created_at")}

    def plot(self, scan_id: str, kind: str) -> Path:
        if kind not in {"waterfall", "waveform", "psd", "stft"}:
            raise LookupError("图件不存在")
        scan = self.get(scan_id)
        if scan["status"] != "succeeded":
            raise LookupError("图件尚未生成")
        path = self._folder(scan_id) / f"{kind}.png"
        if (
            not path.is_file()
            or path.is_symlink()
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != scan["result"]["plot_sha256"].get(kind)
        ):
            raise LookupError("图件不可用")
        return path

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False)
