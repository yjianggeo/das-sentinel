import json
from datetime import timedelta

import numpy as np
import pytest
from das_sentinel.config import ProcessingProfile
from das_sentinel.whole_window import (
    analyze_window,
    inspect_sources,
    plan_windows,
    window_id,
    window_start,
)

from tests import test_event_windows as windows
from tests.test_whole_window import BASE, write

context = windows.context


def test_default_half_minute_alignment_and_separate_legacy_identity():
    assert window_start(BASE + timedelta(seconds=45)) == BASE + timedelta(seconds=30)
    assert window_id("site", BASE) != window_id("site", BASE, window_seconds=60)
    assert window_start(BASE + timedelta(seconds=45), window_seconds=60) == BASE


def test_actual_thirty_second_array_keeps_all_channels_and_clips_peak_assignment(tmp_path):
    source = write(tmp_path / "source.h5")
    result = analyze_window([source], BASE, ProcessingProfile(device="cpu", min_coincidence=2), tmp_path / "out")
    assert result["shape"] == [8,3000]
    assert result["end_utc"] == (BASE+timedelta(seconds=30)).isoformat()
    with np.load(tmp_path / "out/waveform.npz") as arrays:
        assert arrays["waveform"].shape == (8,3000)
        assert arrays["relative_seconds"][-1] == 29.99
    evidence = json.loads((tmp_path / "out/evidence.json").read_text())
    assert evidence and all(0 <= e["peak_sample"] < 3000 for e in evidence)
    plans=plan_windows(inspect_sources([source]))
    assert any(p["start_utc"] == (BASE+timedelta(seconds=30)).isoformat() for p in plans)


def test_same_start_allows_new_thirty_second_and_preserved_sixty_second(context):
    from das_sentinel.services.event_windows import enqueue_windows

    session,site,profile,service=context
    original=windows.plan()
    old=enqueue_windows(session,site.id,profile.id,[original])[0]
    short=original|{"end_utc":(BASE.replace(hour=0,minute=0)+timedelta(seconds=30)).isoformat()}
    new=enqueue_windows(session,site.id,profile.id,[short])[0]
    assert old.window_id != new.window_id
    assert service.list()["total"] == 1
    assert service.list()["items"][0]["id"] == new.window_id
    assert service.list(window_seconds=60)["items"][0]["id"] == old.window_id
    assert service.detail(new.window_id)["window_seconds"] == 30
    with pytest.raises(ValueError):
        window_start(BASE, window_seconds=45)
