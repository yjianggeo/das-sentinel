import io
from datetime import UTC, datetime

import numpy as np
import pytest
from das_sentinel.whole_window_plots import create_visualization, render_region_preview
from das_sentinel.window_regions import merge_regions, preview_bounds


def evidence(start, stop, first=1, last=4, *, qualified=True, detector="stalta:short"):
    return {"start_sample": start, "end_sample": stop, "peak_sample": max(0, start),
                "channel_start": first, "channel_stop": last, "score": 3.0,
                "qualified": qualified, "detector": detector}


def test_merges_only_qualified_overlapping_evidence_and_keeps_separate_signals():
    rows = [evidence(400, 500), evidence(450, 650, detector="energy:band-0"),
            evidence(2000, 2200), evidence(100, 2300, qualified=False)]
    regions = merge_regions(rows, np.arange(8) * 4.0, np.ones(8, bool), 3000, 0)
    assert len(regions) == 2
    assert (regions[0]["start_sample"], regions[0]["end_sample"]) == (400, 650)
    assert regions[0]["evidence_count"] == 2
    assert regions[0]["detectors"] == ["energy:band-0", "stalta:short"]
    assert regions == merge_regions(rows[::-1], np.arange(8) * 4.0, np.ones(8, bool), 3000, 0)


def test_clips_time_and_excludes_nonrecognition_channels():
    rows = [evidence(-50, 150, 0, 3), evidence(2950, 3100, 4, 8),
            evidence(400, 600, 0, 2), evidence(3100, 3300)]
    mask = np.array([False, False, True, True, True, True, True, True])
    regions = merge_regions(rows, np.arange(8) * 4.0, mask, 3000, 0)
    assert len(regions) == 2
    assert (regions[0]["start_sample"], regions[0]["channel_start"]) == (0, 2)
    assert regions[1]["end_sample"] == 3000
    assert (regions[0]["distance_start_m"], regions[0]["distance_stop_m"]) == (6, 10)
    assert all(0 <= r["peak_sample"] < 3000 for r in regions)


@pytest.mark.parametrize("start,stop", [(0, 5), (1000, 1001), (2998, 3000)])
def test_preview_is_at_least_three_seconds_and_stays_within_training_window(start, stop):
    bounds = preview_bounds({"start_sample": start, "end_sample": stop,
                             "channel_start": 1, "channel_stop": 3}, 3000, 8)
    a, b, c, d = bounds
    assert 0 <= a < b <= 3000 and b - a >= 300
    assert 0 <= c < d <= 8


def test_visualization_generates_two_distinct_waterfalls_and_cached_filtered_samples():
    raw = np.random.default_rng(5).normal(size=(3000, 8)).astype("f4")
    filtered = raw * 0.2
    metadata, files = create_visualization(
        raw, filtered, np.arange(8) * 4.0, datetime(2023, 1, 1, tzinfo=UTC),
        np.ones(8, bool), "strain", [evidence(400, 650)], merge_gap_seconds=0,
    )
    assert set(files) == {"waterfall_raw_v2.png", "waterfall_filtered_v2.png", "filtered_waveform_v2.npz"}
    assert files["waterfall_raw_v2.png"] != files["waterfall_filtered_v2.png"]
    assert metadata["visualization_version"] == 2 and len(metadata["detection_regions"]) == 1
    left, top, width, height = metadata["plot_geometry"]["axes"]
    assert 0 < left < left + width < 1 and 0 < top < top + height < 1
    with np.load(io.BytesIO(files["filtered_waveform_v2.npz"]), allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache["waveform"], filtered.T)
    preview = render_region_preview(raw, np.arange(8) * 4.0, datetime(2023, 1, 1, tzinfo=UTC),
                                    np.ones(8, bool), "strain", metadata["detection_regions"][0], filtered=False)
    assert preview.startswith(b"\x89PNG")
