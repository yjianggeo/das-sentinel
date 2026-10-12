"""Physical-distance cropping for offline whole-window model comparisons."""

import numpy as np

from das_sentinel.classifiers.window_resnet import prepare_waveform

_CHANNEL_FIELDS = (
    "distance_m",
    "valid_channel_mask",
    "recognition_channel_mask",
    "region_id",
)


def crop_distance_arrays(arrays, distance_start_m):
    """Return an un-interpolated channel suffix beginning at ``distance_start_m``."""
    start_array = np.asarray(distance_start_m)
    if start_array.ndim != 0:
        raise ValueError("distance_start_m must be a finite non-negative scalar")
    try:
        start = float(start_array)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "distance_start_m must be a finite non-negative scalar"
        ) from exc
    if not np.isfinite(start) or start < 0:
        raise ValueError("distance_start_m must be a finite non-negative scalar")

    try:
        waveform = np.asarray(arrays["waveform"])
        distance = np.asarray(arrays["distance_m"])
        valid = np.asarray(arrays["valid_channel_mask"])
        recognized = np.asarray(arrays["recognition_channel_mask"])
        regions = np.asarray(arrays["region_id"])
        relative_seconds = np.asarray(arrays["relative_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("array dimensions do not match") from exc

    if (
        waveform.ndim != 2
        or distance.shape != (waveform.shape[0],)
        or valid.shape != (waveform.shape[0],)
        or recognized.shape != (waveform.shape[0],)
        or regions.shape != (waveform.shape[0],)
        or relative_seconds.shape != (waveform.shape[1],)
    ):
        raise ValueError("array dimensions do not match")
    try:
        numeric_distance = distance.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance_m must be finite and strictly increasing") from exc
    if not np.isfinite(numeric_distance).all() or not np.all(
        np.diff(numeric_distance) > 0
    ):
        raise ValueError("distance_m must be finite and strictly increasing")

    selected = numeric_distance >= start
    active = valid.astype(bool, copy=False) & recognized.astype(bool, copy=False)
    if not np.any(selected & active):
        raise ValueError("crop must contain a valid recognized channel")

    loaded = {
        "waveform": waveform,
        "distance_m": distance,
        "valid_channel_mask": valid,
        "recognition_channel_mask": recognized,
        "region_id": regions,
        "relative_seconds": relative_seconds,
    }
    result = {key: loaded[key] if key in loaded else arrays[key] for key in arrays}
    result["waveform"] = waveform[selected].copy()
    for field, values in zip(_CHANNEL_FIELDS, (distance, valid, recognized, regions)):
        result[field] = values[selected].copy()
    result["relative_seconds"] = relative_seconds.copy()
    return result


def prepare_distance_waveform(arrays, distance_start_m):
    """Crop by physical distance, then apply the standard waveform preparation."""
    return prepare_waveform(crop_distance_arrays(arrays, distance_start_m))
