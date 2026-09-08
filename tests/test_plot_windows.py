from __future__ import annotations

import math

import pytest
from das_sentinel.plot_windows import event_display_window


@pytest.mark.parametrize(
    ("start", "end", "total", "pad", "fs", "want"),
    [
        (450, 460, 1000, 0, 100, (305, 605)),
        (0, 10, 1000, 0, 100, (0, 300)),
        (990, 1000, 1000, 0, 100, (700, 1000)),
        (450, 850, 1000, 0, 100, (450, 850)),
        (40, 50, 100, 0, 100, (0, 100)),
        (100, 101, 300, 0, 10.5, (85, 117)),
        (450, 460, 2000, 5, 100, (0, 960)),
    ],
)
def test_display_window(start, end, total, pad, fs, want):
    """Changing centering, boundary fill, or minimum-duration rounding must fail."""
    assert event_display_window(start, end, total, pad, fs) == want


@pytest.mark.parametrize("sample_rate_hz", [0, -1, math.inf, -math.inf, math.nan])
def test_display_window_rejects_invalid_sample_rate(sample_rate_hz):
    """Accepting a rate that cannot define a sample duration must fail."""
    with pytest.raises(ValueError):
        event_display_window(1, 2, 10, 0, sample_rate_hz)


@pytest.mark.parametrize("pad_seconds", [-0.01, math.inf, -math.inf, math.nan])
def test_display_window_rejects_invalid_padding(pad_seconds):
    """Accepting negative or non-finite configured context must fail."""
    with pytest.raises(ValueError):
        event_display_window(1, 2, 10, pad_seconds, 100)


@pytest.mark.parametrize(
    ("start_sample", "end_sample", "sample_count"),
    [
        (-1, 1, 10),
        (1, 1, 10),
        (2, 1, 10),
        (1, 11, 10),
        (0, 1, 0),
    ],
)
def test_display_window_rejects_invalid_event_bounds(start_sample, end_sample, sample_count):
    """Accepting an event outside its non-empty source sample range must fail."""
    with pytest.raises(ValueError):
        event_display_window(start_sample, end_sample, sample_count, 0, 100)
