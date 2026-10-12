"""Prepare the deployed waveform model input without detection, features or plots."""

import hashlib
from pathlib import Path

import numpy as np

from das_sentinel.model_scan_contract import PROTOCOL, canonical, utc
from das_sentinel.readers.continuous_low_frequency import (
    CoreCache,
    read_low_frequency_window,
)
from das_sentinel.whole_window import inspect_sources
from das_sentinel.window_grid import (
    native_geometry,
    normalize_arrays,
    validate_native_geometry,
)

ARRAY_KEYS = ("waveform", "distance_m", "relative_seconds", "valid_channel_mask",
              "recognition_channel_mask", "region_id")


def arrays_sha256(arrays, start_utc, unit):
    value = hashlib.sha256(canonical({"protocol": PROTOCOL, "start_utc": utc(start_utc).isoformat(),
                                      "unit": unit}).encode())
    for key in ARRAY_KEYS:
        array = np.ascontiguousarray(arrays[key])
        value.update(canonical([key, array.dtype.str, array.shape]).encode())
        value.update(memoryview(array))
    return value.hexdigest()


def read_scan_input(plan, model_metadata, cache_root, *, core_cache=None):
    if plan["coverage"] != "complete":
        raise ValueError("incomplete source/context coverage")
    paths = [Path(s["path"]) for s in plan["sources"]]
    if inspect_sources(paths) != plan["sources"]:
        raise ValueError("source changed before model read")
    if any(s["unit"] != model_metadata["unit"] for s in plan["sources"]):
        raise ValueError("model unit mismatch")
    start, end = utc(plan["start_utc"]), utc(plan["end_utc"])
    if (end - start).total_seconds() != 30:
        raise ValueError("model requires 30 seconds")
    geometries = [native_geometry(s, Path(cache_root) / "native-grid") for s in plan["sources"]]
    geometry = geometries[0]
    for other in geometries:
        if any(other[k] != geometry[k] for k in ("spacing_m", "start_locus", "channels", "unit")):
            raise ValueError("native geometry changes within a window")
    validate_native_geometry(geometry, model_metadata["training_grid"])
    read = read_low_frequency_window(paths, start, end, None, None, margin_seconds=30, sample_rate_hz=100,
                                     core_cache=core_cache)
    if inspect_sources(paths) != plan["sources"]:
        raise ValueError("source changed during model read")
    raw = read.block.data[read.analysis_start:read.analysis_stop].copy()
    distance = read.block.metadata.channel_distance_m
    if (raw.shape != (3000, len(distance)) or not np.isfinite(read.block.data).all()
            or not np.array_equal(distance, np.asarray(plan["sources"][0]["distance_m"]))):
        raise ValueError("invalid whole-window waveform")
    arrays = normalize_arrays({"waveform": raw.T, "distance_m": distance,
        "relative_seconds": np.arange(3000) / 100, "valid_channel_mask": np.ptp(raw, axis=0) > 0},
        geometry, model_metadata["training_grid"], {"start": model_metadata["distance_start_m"], "stop": None})
    if hashlib.sha256(arrays["distance_m"].tobytes()).hexdigest() != model_metadata["training_grid"]["grid_sha256"]:
        raise ValueError("model normalization grid mismatch")
    return arrays, {"model_input_sha256": arrays_sha256(arrays, start, model_metadata["unit"]),
                    "sources": plan["sources"], "read_sources": list(read.sources),
                    "unit": model_metadata["unit"], "warnings": list(read.warnings)}


def iter_scan_inputs(plans, metadata, cache_root, *, read_mode="single"):
    if read_mode not in ("single", "block"):
        raise ValueError("unsupported read mode")
    if len(plans) > 10 or (plans and (utc(plans[-1]["end_utc"]) - utc(plans[0]["start_utc"])).total_seconds() > 300):
        raise ValueError("model scan block exceeds five minutes")
    cores = CoreCache() if read_mode == "block" else None
    for plan in plans:
        arrays, provenance = read_scan_input(plan, metadata, cache_root, core_cache=cores)
        yield plan, arrays, provenance
