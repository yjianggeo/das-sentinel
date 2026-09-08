from __future__ import annotations

import math


def event_display_window(
    start_sample: int,
    end_sample: int,
    sample_count: int,
    pad_seconds: float,
    sample_rate_hz: float,
) -> tuple[int, int]:
    """Return a source-bounded event window with at least three seconds when available."""
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if not math.isfinite(pad_seconds) or pad_seconds < 0:
        raise ValueError("pad_seconds must be non-negative and finite")
    if sample_count <= 0 or not 0 <= start_sample < end_sample <= sample_count:
        raise ValueError("event bounds must define a non-empty range within the source")

    padding = max(0, round(pad_seconds * sample_rate_hz))
    start = max(0, start_sample - padding)
    stop = min(sample_count, end_sample + padding)
    target = min(sample_count, max(stop - start, math.ceil(3 * sample_rate_hz)))
    missing = target - (stop - start)
    start = max(0, min(start - missing // 2, sample_count - target))
    return start, start + target
