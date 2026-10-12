"""Align training arrays on verified native channel coordinates without interpolation."""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np

from das_sentinel.services.discovery import fingerprint

VERSION = "native-channel-union-v1"
logger = logging.getLogger(__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save_json(path, value):
    temporary = path.with_name(f"{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf8")
    try:
        os.link(temporary, path)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf8")) != value:
            raise ValueError("immutable grid cache conflict") from None
    finally:
        temporary.unlink()


def native_geometry(source, cache_root):
    """Read authoritative HDF5 geometry; cache only for an unchanged source identity."""
    path = Path(source["path"])
    identity = {k: source[k] for k in ("path", "fingerprint", "size_bytes", "mtime_ns")}

    def unchanged():
        stat = path.stat()
        if (stat.st_size != source["size_bytes"] or stat.st_mtime_ns != source["mtime_ns"]
                or fingerprint(path) != source["fingerprint"]):
            raise ValueError(f"native grid source changed: {path}")

    unchanged()
    root = Path(cache_root) / "sources"
    root.mkdir(parents=True, exist_ok=True)
    cached = root / f"{_digest([VERSION, identity])}.json"
    if cached.exists():
        saved = json.loads(cached.read_text(encoding="utf8"))
        if saved["source"] != identity or saved["version"] != VERSION:
            raise ValueError("native grid source cache mismatch")
        return saved["geometry"]
    with h5py.File(path, "r") as handle:
        acquisition, raw = handle["Acquisition"], handle["Acquisition/Raw[0]"]
        owner = raw if "SpatialSamplingInterval" in raw.attrs else acquisition
        if "SpatialSamplingInterval" not in owner.attrs:
            raise ValueError(f"native grid spacing metadata missing: {path}")
        spacing = float(owner.attrs["SpatialSamplingInterval"])
        units = owner.attrs.get("SpatialSamplingIntervalUnit", "m")
        units = units.decode() if isinstance(units, bytes) else str(units)
        if not np.isfinite(spacing) or spacing <= 0 or units != "m":
            raise ValueError("native grid spacing must be finite positive metres")
        start = raw.attrs.get("StartLocusIndex", acquisition.attrs.get("StartLocusIndex"))
        if start is None or not np.isfinite(start) or int(start) != start:
            raise ValueError("native grid start locus must be an integer")
        shape, samples = raw["RawData"].shape, raw["RawDataTime"].shape
        if len(shape) != 2 or len(samples) != 1 or samples[0] not in shape:
            raise ValueError("native grid data dimensions mismatch")
        count = shape[1] if shape[0] == samples[0] else shape[0]
        unit = raw.attrs["RawDataUnit"]
        unit = unit.decode() if isinstance(unit, bytes) else str(unit)
        geometry = {"spacing_m": spacing, "start_locus": int(start), "channels": count,
                    "unit": unit, "metadata_group": owner.name}
    unchanged()
    _save_json(cached, {"version": VERSION, "source": identity, "geometry": geometry})
    return geometry


def validate_native_geometry(geometry, fixed_grid):
    """Reject incompatible source coordinates before reading large waveforms."""
    first, count = fixed_grid["start_locus"], fixed_grid["channels"]
    if (fixed_grid["version"] != VERSION or geometry["spacing_m"] != fixed_grid["spacing_m"]
            or geometry["start_locus"] < first
            or geometry["start_locus"] + geometry["channels"] > first + count):
        raise ValueError("native geometry exceeds the fixed station training grid")


def normalize_arrays(arrays, geometry, fixed_grid, recognition_range):
    """Map verified native channels to a fixed grid without distance interpolation."""
    validate_native_geometry(geometry, fixed_grid)
    first, count = fixed_grid["start_locus"], fixed_grid["channels"]
    wave = np.asarray(arrays["waveform"])
    valid = np.asarray(arrays["valid_channel_mask"], bool)
    time = np.asarray(arrays["relative_seconds"])
    if (wave.shape != (geometry["channels"], 3000) or not np.isfinite(wave).all()
            or valid.shape != (len(wave),) or not valid.any()
            or time.shape != (3000,) or not np.allclose(time, np.arange(3000) / 100, rtol=0, atol=1e-8)):
        raise ValueError("source waveform geometry contract mismatch")
    distance = (first + np.arange(count, dtype="<f8")) * fixed_grid["spacing_m"]
    offset = geometry["start_locus"] - first
    padded = np.zeros((count, 3000), dtype=wave.dtype)
    padded[offset:offset + len(wave)] = wave
    valid_mask = np.zeros(count, bool)
    valid_mask[offset:offset + len(wave)] = valid
    recognition = valid_mask.copy()
    begin, end = (recognition_range.get(k) for k in ("start", "stop"))
    if begin is not None:
        recognition &= distance >= begin
    if end is not None:
        recognition &= distance <= end
    if not recognition.any():
        raise ValueError("no recognized channels on native grid")
    return {"waveform": padded, "distance_m": distance, "relative_seconds": time,
                "valid_channel_mask": valid_mask, "recognition_channel_mask": recognition,
                "region_id": np.where(distance < 299, 0, np.where(distance < 499, 1, 2)).astype(np.int8)}


def normalize_rows(rows, cache_root):
    """Create immutable derived arrays; preserve raw input identity and labels."""
    if not rows:
        return [], None
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    sources = {}
    for row in rows:
        if row["window_seconds"] != 30 or not row.get("sources"):
            raise ValueError("native grid needs 30-second source-bound windows")
        for source in row["sources"]:
            identity = {k: source[k] for k in ("path", "fingerprint", "size_bytes", "mtime_ns")}
            if source["path"] in sources and sources[source["path"]] != identity:
                raise ValueError("conflicting source identities")
            sources[source["path"]] = identity
    geometries = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (path, geometry) in enumerate(zip(sources, pool.map(
            lambda source: native_geometry(source, root), sources.values()
        )), 1):
            geometries[path] = geometry
            if index % 100 == 0 or index == len(sources):
                logger.info("Verified native geometry %s/%s", index, len(sources))
    native = []
    for row in rows:
        geometry = geometries[row["sources"][0]["path"]]
        for source in row["sources"]:
            other = geometries[source["path"]]
            if any(other[k] != geometry[k] for k in ("spacing_m", "start_locus", "channels", "unit")):
                raise ValueError("native geometry changes within a window")
            if source.get("channel_start", 0) != 0 or source.get("channel_stop", geometry["channels"]) != geometry["channels"]:
                raise ValueError("native grid requires full source channels")
        if geometry["unit"] != row["unit"]:
            raise ValueError("native grid physical unit mismatch")
        native.append(geometry)
    spacing = native[0]["spacing_m"]
    if any(g["spacing_m"] != spacing for g in native):
        raise ValueError("different native spacing requires a separately validated resampling protocol")
    first = min(g["start_locus"] for g in native)
    stop = max(g["start_locus"] + g["channels"] for g in native)
    contract_path = root / "grid.json"
    contract = {"version": VERSION, "spacing_m": spacing, "start_locus": first, "channels": stop - first}
    fixed = json.loads(contract_path.read_text(encoding="utf8")) if contract_path.exists() else contract
    if (fixed["version"] != VERSION or fixed["spacing_m"] != spacing
            or first < fixed["start_locus"] or stop > fixed["start_locus"] + fixed["channels"]):
        raise ValueError("native geometry exceeds the fixed station training grid")
    first, stop = fixed["start_locus"], fixed["start_locus"] + fixed["channels"]
    distance = (first + np.arange(stop - first, dtype="<f8")) * spacing
    grid = {"version": VERSION, "spacing_m": spacing, "start_locus": first, "channels": len(distance),
            "start_m": float(distance[0]), "stop_m": float(distance[-1]),
            "grid_sha256": hashlib.sha256(distance.tobytes()).hexdigest(),
            "source_grid_count": len({r["grid_sha256"] for r in rows})}
    normalized = []
    for index, (row, geometry) in enumerate(zip(rows, native), 1):
        recipe = {"version": VERSION, "source_input_sha256": row["input_sha256"],
                  "source_waveform_sha256": row["waveform_sha256"], "source_grid_sha256": row["grid_sha256"],
                  "native_geometry": geometry, "target_grid": grid["grid_sha256"],
                  "recognition_range_m": row["recognition_range_m"]}
        key = _digest(recipe)
        folder = root / "arrays"
        folder.mkdir(parents=True, exist_ok=True)
        output, metadata = folder / f"{key}.npz", folder / f"{key}.json"
        if output.exists() and metadata.exists():
            saved = json.loads(metadata.read_text(encoding="utf8"))
            if saved["recipe"] != recipe or _sha(output) != saved["sha256"]:
                raise ValueError("normalized waveform cache hash mismatch")
        else:
            if _sha(row["waveform"]) != row["waveform_sha256"]:
                raise ValueError("source waveform hash mismatch")
            with np.load(row["waveform"], allow_pickle=False) as arrays:
                wave = arrays["waveform"]
                valid = np.asarray(arrays["valid_channel_mask"], bool)
                raw_distance = np.asarray(arrays["distance_m"], dtype="<f8")
                time = arrays["relative_seconds"]
                if (wave.shape != (geometry["channels"], 3000) or not np.isfinite(wave).all()
                        or valid.shape != (len(wave),) or not valid.any()
                        or raw_distance.shape != (len(wave),)
                        or hashlib.sha256(raw_distance.tobytes()).hexdigest() != row["grid_sha256"]
                        or time.shape != (3000,) or not np.allclose(time, np.arange(3000) / 100)):
                    raise ValueError("source waveform geometry contract mismatch")
                aligned = normalize_arrays({"waveform": wave, "valid_channel_mask": valid,
                    "relative_seconds": time}, geometry, fixed, row["recognition_range_m"])
                temporary = output.with_name(f"{uuid4().hex}.tmp")
                with temporary.open("wb") as stream:
                    np.savez(stream, **aligned)
                try:
                    os.link(temporary, output)
                except FileExistsError:
                    if _sha(temporary) != _sha(output):
                        raise ValueError("immutable waveform cache conflict") from None
                finally:
                    temporary.unlink()
            saved = {"recipe": recipe, "sha256": _sha(output)}
            _save_json(metadata, saved)
        normalized.append(row | {"source_waveform": row["waveform"], "source_waveform_sha256": row["waveform_sha256"],
                                  "source_grid_sha256": row["grid_sha256"], "native_geometry": geometry,
                                  "grid_normalization": VERSION, "grid_sha256": grid["grid_sha256"],
                                  "waveform": str(output), "waveform_sha256": saved["sha256"]})
        if index % 100 == 0 or index == len(rows):
            logger.info("Normalized training window %s/%s", index, len(rows))
    _save_json(contract_path, fixed)
    return normalized, grid
