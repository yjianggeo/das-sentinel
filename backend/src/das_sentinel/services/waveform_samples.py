"""Build and persist full-channel numeric waveform samples."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from das_sentinel.low_frequency import analyze_low_frequency
from das_sentinel.readers.continuous_low_frequency import read_low_frequency_window

_VIEWS = {
    "standard": (60, 100, 0.0),
    "low_frequency": (600, 5, 300.0),
}
_REQUIRED_ARRAYS = {
    "waveform", "distance_m", "relative_seconds", "valid_channel_mask",
    "region_id", "recognition_channel_mask",
}


def _utc(value: datetime | str, field: str) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field}必须包含时区")
    return parsed.astimezone(UTC)


def _epoch_microseconds(value: datetime) -> int:
    elapsed = value - datetime(1970, 1, 1, tzinfo=UTC)
    return (elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds


def _mapped_annotations(
    annotations: Sequence[Mapping[str, Any]], start: datetime, end: datetime,
    rate: int, channels: int,
) -> list[dict[str, Any]]:
    mapped = []
    for annotation in annotations:
        item = json.loads(json.dumps(dict(annotation), allow_nan=False))
        item_start = _utc(item["start_utc"], "annotation.start_utc")
        item_end = _utc(item["end_utc"], "annotation.end_utc")
        if item_end <= item_start:
            raise ValueError("annotation结束时间必须晚于开始时间")
        overlap_start, overlap_end = max(start, item_start), min(end, item_end)
        channel_start = max(0, int(item["channel_start"]))
        channel_stop = min(channels, int(item["channel_stop"]))
        if overlap_end <= overlap_start or channel_stop <= channel_start:
            continue
        # All sample starts are on the rate grid, so reviewed UTC bounds map
        # inward to the overlapping half-open sample ROI.
        first = max(0, int(np.ceil((overlap_start - start).total_seconds() * rate - 1e-9)))
        stop = min(round((end - start).total_seconds() * rate),
                   int(np.ceil((overlap_end - start).total_seconds() * rate - 1e-9)))
        if stop <= first:
            continue
        item.update(
            time_start_index=first, time_stop_index=stop,
            sample_channel_start=channel_start,
            sample_channel_stop=channel_stop,
        )
        mapped.append(item)
    return mapped


def build_sample(
    paths: Sequence[Path],
    start_utc: datetime,
    view: str,
    reference_distances: Sequence[float],
    reference_unit: str,
    *,
    suspended_end_m: float,
    borehole_start_m: float,
    annotations: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build one fixed-duration sample beginning on an absolute UTC grid."""
    if view not in _VIEWS:
        raise ValueError("view必须为standard或low_frequency")
    start = _utc(start_utc, "start_utc")
    seconds, rate, margin = _VIEWS[view]
    if _epoch_microseconds(start) % (1_000_000 // rate):
        raise ValueError("start_utc必须与视图UTC采样网格aligned")
    if not np.isfinite(suspended_end_m) or not np.isfinite(borehole_start_m):
        raise ValueError("站点区段边界必须有限")
    if borehole_start_m < suspended_end_m:
        raise ValueError("井下起点不能早于悬空区终点")
    end = start + timedelta(seconds=seconds)
    window = read_low_frequency_window(
        paths, start, end, 0, None,
        margin_seconds=margin, sample_rate_hz=rate,
    )
    block = window.block
    distances = np.asarray(block.metadata.channel_distance_m)
    reference = np.asarray(reference_distances)
    if not np.array_equal(distances, reference):
        raise ValueError("源文件距离坐标与参考距离不完全一致")
    if block.metadata.unit != reference_unit:
        raise ValueError("源文件单位与参考单位不一致")
    expected = seconds * rate
    raw = np.asarray(block.data[window.analysis_start:window.analysis_stop])
    if raw.shape != (expected, len(distances)):
        raise ValueError("读取器返回的样本形状不符合固定视图契约")
    if not np.isfinite(raw).all():
        raise ValueError("源波形包含非有限值")
    waveform = np.ascontiguousarray(raw.T, dtype=np.float32)
    valid = np.ptp(waveform, axis=1) > 0
    filtered = None
    if view == "low_frequency":
        analysis = analyze_low_frequency(
            block, window.analysis_start, window.analysis_stop
        )
        filtered = np.ascontiguousarray(analysis.filtered_data.T, dtype=np.float32)
        if filtered.shape != waveform.shape or not np.isfinite(filtered).all():
            raise ValueError("低频滤波结果不符合样本契约")
        valid &= np.asarray(analysis.valid_channel_mask, dtype=bool)
        filtered[~valid] = 0
    region = np.where(
        distances < suspended_end_m, 0,
        np.where(distances < borehole_start_m, 1, 2),
    ).astype(np.uint8)
    arrays = {
        "waveform": waveform,
        "distance_m": np.asarray(distances, dtype=np.float64),
        "relative_seconds": np.arange(expected, dtype=np.float64) / rate,
        "valid_channel_mask": np.asarray(valid, dtype=bool),
        "region_id": region,
        "recognition_channel_mask": np.asarray(valid & (region != 0), dtype=bool),
    }
    if filtered is not None:
        arrays["filtered_waveform"] = filtered
    metadata = {
        "schema_version": 1,
        "view": view,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "sample_rate_hz": rate,
        "unit": reference_unit,
        "shape": [len(distances), expected],
        "suspended_end_m": float(suspended_end_m),
        "borehole_start_m": float(borehole_start_m),
        "sources": [dict(source) for source in window.sources],
        "warnings": list(window.warnings),
        "annotations": _mapped_annotations(annotations, start, end, rate, len(distances)),
    }
    return arrays, metadata


def _validate(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    missing = _REQUIRED_ARRAYS - arrays.keys()
    if missing:
        raise ValueError(f"样本数组contract缺少：{sorted(missing)}")
    view = metadata.get("view")
    if view not in _VIEWS:
        raise ValueError("metadata view无效")
    seconds, rate, _ = _VIEWS[view]
    allowed = _REQUIRED_ARRAYS | ({"filtered_waveform"} if view == "low_frequency" else set())
    if arrays.keys() - allowed or allowed - arrays.keys():
        raise ValueError("样本数组contract包含缺失或未知数组")
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError(f"{name}必须是不含对象的有限数值数组")
    waveform = np.asarray(arrays["waveform"])
    if waveform.ndim != 2 or waveform.dtype != np.float32 or not np.isfinite(waveform).all():
        raise ValueError("waveform必须为有限float32二维数组")
    channels, samples = waveform.shape
    if channels <= 0 or samples != seconds * rate:
        raise ValueError("waveform不符合视图固定形状")
    for name in ("distance_m", "valid_channel_mask", "region_id", "recognition_channel_mask"):
        if np.asarray(arrays[name]).shape != (channels,):
            raise ValueError(f"{name}维度不符合样本数组contract")
    relative = np.asarray(arrays["relative_seconds"])
    if relative.shape != (samples,) or not np.array_equal(
        relative, np.arange(samples, dtype=np.float64) / rate
    ):
        raise ValueError("relative_seconds维度或数值无效")
    distances = np.asarray(arrays["distance_m"])
    if not np.isfinite(distances).all() or np.any(np.diff(distances) <= 0):
        raise ValueError("distance_m必须有限且严格递增")
    if not np.isin(arrays["region_id"], [0, 1, 2]).all():
        raise ValueError("region_id包含未知区段")
    valid = np.asarray(arrays["valid_channel_mask"])
    recognition = np.asarray(arrays["recognition_channel_mask"])
    if valid.dtype != np.bool_ or recognition.dtype != np.bool_:
        raise ValueError("通道掩码必须为bool")
    if "filtered_waveform" in arrays:
        filtered = np.asarray(arrays["filtered_waveform"])
        if filtered.shape != waveform.shape or filtered.dtype != np.float32 or not np.isfinite(filtered).all():
            raise ValueError("filtered_waveform不符合样本数组contract")
    required_metadata = {
        "schema_version", "view", "start_utc", "end_utc", "sample_rate_hz",
        "unit", "shape", "sources", "warnings", "annotations",
    }
    if required_metadata - metadata.keys():
        raise ValueError("metadata schema缺少必要字段")
    if (metadata["schema_version"] != 1 or metadata["shape"] != [channels, samples]
            or metadata["sample_rate_hz"] != rate):
        raise ValueError("metadata schema版本或shape无效")
    start = _utc(metadata["start_utc"], "metadata.start_utc")
    end = _utc(metadata["end_utc"], "metadata.end_utc")
    if end - start != timedelta(seconds=seconds):
        raise ValueError("metadata UTC范围不符合视图时长")
    tick_us = 1_000_000 // rate
    if _epoch_microseconds(start) % tick_us:
        raise ValueError("metadata start_utc未与UTC采样网格对齐")
    try:
        suspended = float(metadata["suspended_end_m"])
        borehole = float(metadata["borehole_start_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("metadata缺少有效站点区段边界") from error
    if (not np.isfinite(suspended) or not np.isfinite(borehole)
            or suspended < 0 or borehole < suspended):
        raise ValueError("metadata站点区段边界无效")
    expected_region = np.where(distances < suspended, 0,
                               np.where(distances < borehole, 1, 2)).astype(np.uint8)
    if not np.array_equal(arrays["region_id"], expected_region):
        raise ValueError("region_id与站点区段边界不一致")
    if not np.array_equal(recognition, valid & (expected_region != 0)):
        raise ValueError("recognition_channel_mask与有效区段不一致")
    for annotation in metadata["annotations"]:
        bounds = []
        for name in ("time_start_index", "time_stop_index",
                     "sample_channel_start", "sample_channel_stop"):
            value = annotation.get(name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("annotation ROI索引必须为整数")
            bounds.append(value)
        first, stop, channel_first, channel_stop = bounds
        if not (0 <= first < stop <= samples and
                0 <= channel_first < channel_stop <= channels):
            raise ValueError("annotation ROI超出样本范围")


def save_sample(directory: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    """Atomically create a sample directory, refusing any existing target."""
    directory = Path(directory)
    _validate(arrays, metadata)
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        np.savez(temporary / "arrays.npz", **arrays)
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, directory)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_sample(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and validate a saved sample without permitting pickle payloads."""
    directory = Path(directory)
    with np.load(directory / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    _validate(arrays, metadata)
    return arrays, metadata
