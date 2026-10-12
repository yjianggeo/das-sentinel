import io
from datetime import UTC, datetime

import numpy as np
from das_sentinel.whole_window_plots import (
    create_visualization,
    render_waterfall_preview,
)
from PIL import Image


def test_new_waterfalls_have_plain_pixels_and_preserve_detection_geometry():
    raw = np.tile(np.sin(np.arange(1000) / 11)[:, None], (1, 4))
    filtered = raw.astype("f4") * 0.5
    distance = np.array([300., 301., 302., 303.])
    recognition = np.array([False, True, True, True])
    start = datetime(2026, 1, 1, tzinfo=UTC)
    evidence = [{"start_sample": 400, "end_sample": 500, "peak_sample": 450,
                 "channel_start": 1, "channel_stop": 3, "detector": "stalta:short",
                 "score": 3.0, "qualified": True}]
    metadata, files = create_visualization(raw, filtered, distance, start, recognition, "rad",
                                           evidence, merge_gap_seconds=0)
    assert len(metadata["detection_regions"]) == 1
    assert metadata["plot_geometry"]["distance_min_m"] == 299.5
    assert metadata["plot_geometry"]["distance_max_m"] == 303.5
    for name, samples, is_filtered in (("waterfall_raw_v2.png", raw, False),
                                       ("waterfall_filtered_v2.png", filtered, True)):
        plain = render_waterfall_preview(samples, distance, start, recognition, "rad", filtered=is_filtered)
        marked = render_waterfall_preview(samples, distance, start, recognition, "rad", filtered=is_filtered,
                                          regions=metadata["detection_regions"])
        pixels = np.asarray(Image.open(io.BytesIO(files[name])))
        plain_pixels = np.asarray(Image.open(io.BytesIO(plain)))
        marked_pixels = np.asarray(Image.open(io.BytesIO(marked)))
        assert np.array_equal(pixels, plain_pixels)
        assert np.count_nonzero(np.any(pixels != marked_pixels, axis=2)) > 100
        assert np.array_equal(pixels[:70], marked_pixels[:70])
    assert metadata["waterfall_regions_baked"] is False
    with np.load(io.BytesIO(files["filtered_waveform_v2.npz"]), allow_pickle=False) as arrays:
        assert np.array_equal(arrays["waveform"], filtered.T)
