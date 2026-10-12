"""Offline station waveform datasets: python -m das_sentinel.waveform_dataset."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from das_sentinel.readers.zd_hdf5 import ZdHdf5Reader
from das_sentinel.services.discovery import fingerprint

app = typer.Typer(no_args_is_help=True, help="只读原数据，导出站点统一二维波形样本。")
_VIEWS = {"standard": (60, 100, 0), "low_frequency": (600, 5, 300)}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def grid_fingerprint(distances: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(distances, dtype="<f8").tobytes()).hexdigest()


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("时间必须带 UTC 或时区偏移，例如 2026-06-16T09:05:00+00:00")
    return parsed.astimezone(UTC)


def _view(view: str) -> tuple[int, int, int]:
    if view not in _VIEWS:
        raise ValueError("view 必须为 standard 或 low_frequency")
    return _VIEWS[view]


def _aligned_start(start: datetime, rate: int) -> datetime:
    elapsed = start - _EPOCH
    microseconds = (elapsed.days * 86400 + elapsed.seconds) * 1_000_000 + elapsed.microseconds
    step = 1_000_000 // rate
    return _EPOCH + timedelta(microseconds=(microseconds // step) * step)


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _source_index(directories: list[Path]) -> tuple[list[dict], list[dict]]:
    reader = ZdHdf5Reader()
    paths: set[Path] = set()
    for directory in directories:
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError(f"不是源目录：{directory}")
        candidates = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".h5", ".hdf5"} and p.is_file())
        if len(candidates) > 2000:
            raise ValueError(f"单目录超过 2000 个 HDF5，请指定较小目录：{directory}")
        paths.update(p.resolve(strict=True) for p in candidates)
    if not paths:
        raise ValueError("指定源目录没有 HDF5 文件")
    result, failed = [], []
    for index, path in enumerate(sorted(paths), 1):
        try:
            before = fingerprint(path)
            metadata = reader.read_metadata(path)
            count = reader.sample_count(path)
            if not np.isfinite(metadata.sample_rate_hz) or metadata.sample_rate_hz <= 0 or count <= 0:
                raise ValueError("源采样率或样本数无效")
            if fingerprint(path) != before:
                raise ValueError("建立索引时源文件发生变化")
            result.append({
                "path": path, "start": metadata.start_time_utc,
                "end": metadata.start_time_utc + timedelta(seconds=count / metadata.sample_rate_hz),
                "fingerprint": before,
            })
        except (OSError, ValueError, KeyError) as error:
            failed.append({"path": str(path), "reason": str(error)})
        if index % 100 == 0 or index == len(paths):
            typer.echo(f"源文件索引：{index}/{len(paths)}", err=True)
    return sorted(result, key=lambda item: (item["start"], str(item["path"]))), failed


def _paths_for(index: list[dict], start: datetime, view: str) -> list[Path]:
    duration, _, margin = _view(view)
    begin = start - timedelta(seconds=margin + 10)
    end = start + timedelta(seconds=duration + margin + 10)
    matches = [item for item in index if item["end"] > begin and item["start"] < end]
    for item in matches:
        if fingerprint(item["path"]) != item["fingerprint"]:
            raise ValueError(f"建立索引后源文件发生变化：{item['path']}")
    return [item["path"] for item in matches]


def _reference(path: Path):
    metadata = ZdHdf5Reader().read_metadata(path)
    return metadata.channel_distance_m, metadata.unit


def _verify_supervision(metadata: dict, arrays: dict, event_id: str) -> None:
    """Bind each exported label ROI to the actual recording that supplied it."""
    start, end = _utc(metadata["start_utc"]), _utc(metadata["end_utc"])
    mapped_target = next((item for item in metadata["annotations"] if item["event_id"] == event_id), None)
    if mapped_target is None:
        raise ValueError("目标降采样后没有有效标注点，不能作为监督训练样本")
    first = mapped_target["sample_channel_start"]
    stop = mapped_target["sample_channel_stop"]
    if not arrays["recognition_channel_mask"][first:stop].any():
        raise ValueError("目标的有效标注没有可用于识别的通道")
    for item in metadata["annotations"]:
        if item["label"] not in {"earthquake", "blast", "traffic", "noise"}:
            raise ValueError("人工标注类别不能用于四类监督训练")
        begin, finish = max(start, _utc(item["start_utc"])), min(end, _utc(item["end_utc"]))
        expected_path = Path(item["source_path"]).resolve(strict=True)
        matching = [
            source for source in metadata["sources"]
            if Path(source["path"]).resolve(strict=True) == expected_path
            and source["fingerprint"] == item["source_fingerprint"]
            and _utc(source["read_start_utc"]) <= begin
            and _utc(source["read_stop_utc"]) >= finish
        ]
        if not matching:
            raise ValueError(f"人工标注 {item['event_id']} 的源文件及区间未被实际读取，不能转移标签")


def _report(view: str, output: Path, succeeded: list[dict], failed: list[dict], index_failed: list[dict], started: float) -> None:
    report = {
        "schema_version": 1, "view": view, "created_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "succeeded": succeeded, "failed": failed, "source_index_failed": index_failed,
        "production_training_changed": False,
    }
    _write_json(output / "report.json", report)
    typer.echo(f"导出完成：成功 {len(succeeded)}，失败 {len(failed)}；报告 {output / 'report.json'}")


@app.command()
def prepare(
    database: Annotated[Path, typer.Option(help="现有 SQLite 数据库，只读打开")],
    site_id: Annotated[str, typer.Option()],
    output: Annotated[Path, typer.Option(help="新目录，保存完整清单")],
) -> None:
    """冻结最新人工标签与完整站点分组；不读取整段波形。"""
    from das_sentinel.services.waveform_manifest import create_manifest

    try:
        if output.exists():
            raise ValueError("输出目录已存在，请使用新目录")
        manifest = create_manifest(database, site_id)
        output.mkdir(parents=True)
        _write_json(output / "manifest.json", manifest)
        typer.echo(json.dumps({
            "manifest": str(output / "manifest.json"),
            "targets": len(manifest["targets"]), "excluded": len(manifest["excluded"]),
            "labels": dict(Counter(item["label"] for item in manifest["targets"])),
            "groups": len({item["group"] for item in manifest["targets"]}),
        }, ensure_ascii=False))
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


@app.command()
def build(
    manifest: Annotated[Path, typer.Option(help="prepare 生成的完整 manifest.json")],
    source_dir: Annotated[list[Path], typer.Option(help="非递归源目录，可重复指定；跨日需相邻目录")],
    reference_source: Annotated[Path, typer.Option(help="固定本数据集真实通道网格与单位的源文件")],
    output: Annotated[Path, typer.Option(help="新数据集目录")],
    view: Annotated[str, typer.Option()] = "standard",
    limit: Annotated[int, typer.Option(min=1, help="默认小批 20 条；不改变完整分组")] = 20,
    event_id: Annotated[list[str] | None, typer.Option(help="只导出指定事件，可重复；优先于 limit")] = None,
    suspended_end_m: Annotated[float, typer.Option(min=0)] = 0,
    borehole_start_m: Annotated[float, typer.Option(min=0)] = 0,
) -> None:
    """导出固定形状波形与局部人工标注；失败逐项记录，绝不补造缺失数据。"""
    from das_sentinel.services.waveform_samples import build_sample, save_sample

    started = time.monotonic()
    try:
        duration, rate, _ = _view(view)
        if output.exists():
            raise ValueError("输出目录已存在，请使用新目录")
        if borehole_start_m < suspended_end_m:
            raise ValueError("井下起点不能小于悬空段终点")
        contents = manifest.read_bytes()
        frozen = json.loads(contents)
        if frozen["schema_version"] != 1:
            raise ValueError("不支持的清单版本")
        targets = frozen["targets"]
        if not targets:
            raise ValueError("清单没有可导出的人工复核目标")
        if len({item["event_id"] for item in targets}) != len(targets):
            raise ValueError("清单包含重复 event_id")
        requested = set(event_id or [])
        missing = requested - {item["event_id"] for item in targets}
        if missing:
            raise ValueError(f"事件不在完整清单中：{sorted(missing)}")
        selected = [item for item in targets if item["event_id"] in requested] if requested else targets[:limit]
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", item["event_id"]) for item in selected):
            raise ValueError("event_id 含无效路径字符")
        distances, unit = _reference(reference_source)
        grid_hash = grid_fingerprint(distances)
        index, index_failed = _source_index(source_dir)
        output.mkdir(parents=True)
        with (output / "manifest.json").open("xb") as stream:
            stream.write(contents)
        succeeded, failed = [], []
        for number, target in enumerate(selected, 1):
            event = target["event_id"]
            typer.echo(f"构建 {number}/{len(selected)}：{event}", err=True)
            try:
                first, last = _utc(target["start_utc"]), _utc(target["end_utc"])
                if last <= first or (last - first).total_seconds() > duration:
                    raise ValueError("目标时长无效或超过固定视图，不能截断后继承整段标签")
                start = _aligned_start(first + (last - first) / 2 - timedelta(seconds=duration / 2), rate)
                if last > start + timedelta(seconds=duration):
                    raise ValueError("目标不能完整纳入对齐后的固定视图")
                if target["source_grid_sha256"] != grid_hash or target["source_unit"] != unit:
                    raise ValueError("目标源文件的真实通道网格或单位与参考文件不同")
                if fingerprint(Path(target["source_path"])) != target["source_fingerprint"]:
                    raise ValueError("目标源文件已改变，请重新复核和准备清单")
                annotations = [item for item in targets if _utc(item["end_utc"]) > start and _utc(item["start_utc"]) < start + timedelta(seconds=duration)]
                if any(item["source_grid_sha256"] != grid_hash or item["source_unit"] != unit for item in annotations):
                    raise ValueError("同窗人工标注存在不同通道网格或单位，需单独核对")
                for source_path, source_fingerprint in {(item["source_path"], item["source_fingerprint"]) for item in annotations}:
                    if fingerprint(Path(source_path)) != source_fingerprint:
                        raise ValueError("同窗标注的源文件已改变，请重新准备清单")
                arrays, metadata = build_sample(
                    _paths_for(index, start, view), start, view, distances, unit,
                    suspended_end_m=suspended_end_m, borehole_start_m=borehole_start_m,
                    annotations=annotations,
                )
                _verify_supervision(metadata, arrays, event)
                metadata.update({
                    "site_id": frozen["site_id"], "target_event_id": event,
                    "group": target["group"], "manifest_sha256": hashlib.sha256(contents).hexdigest(),
                    "source_kind": "human_reviewed_event", "supervised_eligible": True,
                    "supervision_scope": "annotation_rois_only",
                    "reference_source": str(reference_source.resolve()),
                    "reference_grid_sha256": grid_hash,
                })
                save_sample(output / event, arrays, metadata)
                succeeded.append({"event_id": event, "label": target["label"], "group": target["group"], "path": event, "shape": list(arrays["waveform"].shape)})
            except (OSError, ValueError, KeyError) as error:
                failed.append({"event_id": event, "reason": str(error)})
        _report(view, output, succeeded, failed, index_failed, started)
        if failed:
            raise typer.Exit(1)
    except (OSError, ValueError, KeyError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


@app.command()
def window(
    source_dir: Annotated[list[Path], typer.Option()],
    reference_source: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    start_utc: Annotated[str, typer.Option(help="明确带时区的窗口零点；向下对齐采样网格")],
    view: Annotated[str, typer.Option()] = "low_frequency",
    suspended_end_m: Annotated[float, typer.Option(min=0)] = 0,
    borehole_start_m: Annotated[float, typer.Option(min=0)] = 0,
) -> None:
    """导出无标签真实窗口，用于低频信号校验；不能直接监督训练。"""
    from das_sentinel.services.waveform_samples import build_sample, save_sample

    started = time.monotonic()
    try:
        _, rate, _ = _view(view)
        start = _aligned_start(_utc(start_utc), rate)
        if output.exists():
            raise ValueError("输出目录已存在，请使用新目录")
        if borehole_start_m < suspended_end_m:
            raise ValueError("井下起点不能小于悬空段终点")
        distances, unit = _reference(reference_source)
        index, index_failed = _source_index(source_dir)
        output.mkdir(parents=True)
        succeeded, failed = [], []
        try:
            arrays, metadata = build_sample(
                _paths_for(index, start, view), start, view, distances, unit,
                suspended_end_m=suspended_end_m, borehole_start_m=borehole_start_m,
            )
            metadata.update({
                "source_kind": "unlabelled_window", "supervised_eligible": False,
                "reference_source": str(reference_source.resolve()),
                "reference_grid_sha256": grid_fingerprint(distances),
            })
            save_sample(output / "sample", arrays, metadata)
            succeeded.append({"path": "sample", "shape": list(arrays["waveform"].shape)})
        except (OSError, ValueError, KeyError) as error:
            failed.append({"reason": str(error)})
        _report(view, output, succeeded, failed, index_failed, started)
        if failed:
            raise typer.Exit(1)
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
