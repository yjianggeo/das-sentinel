import json
from datetime import UTC, datetime, timedelta, timezone

import h5py
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

BASE = datetime(2023, 9, 27, 3, 2, tzinfo=UTC)


def write(path, start=-60, duration=180, channels=8):
    t = start + np.arange(duration * 100) / 100
    rng = np.random.default_rng(7)
    data = rng.normal(0, 0.01, (len(t), channels)).astype("f4")
    for first, last, arrival in [(0, 4, 15), (4, 8, 40)]:
        data[:, first:last] += (
            np.sin(2 * np.pi * 8 * t) * np.exp(-(((t - arrival) / 0.7) ** 2))
        )[:, None]
    with h5py.File(path, "w") as h:
        raw = h.create_group("Acquisition/Raw[0]")
        raw.create_dataset("RawData", data=data)
        raw.create_dataset(
            "RawDataTime", data=np.rint(BASE.timestamp() * 1e6 + t * 1e6).astype("i8")
        )
        raw.attrs.update(
            OutputDataRate=100,
            SpatialSamplingInterval=100.0,
            StartLocusIndex=0,
            RawDataUnit="strain_rate",
        )
    return path


def test_identity_timezone_and_seconds():
    a = BASE + timedelta(seconds=1)
    assert window_id("site", a) == window_id("site", a + timedelta(seconds=1))
    assert window_start(a.astimezone(timezone(timedelta(hours=8)))) == BASE
    with pytest.raises(ValueError):
        window_start(datetime(2023, 1, 1))  # noqa: DTZ001 -- reject naive input


def test_planning_context_and_remote_gap(tmp_path):
    paths = [write(tmp_path / "a.h5", -60, 120), write(tmp_path / "b.h5", 60, 120)]
    sources = inspect_sources(paths)
    plans = plan_windows(sources)
    main = next(p for p in plans if p["start_utc"] == BASE.isoformat())
    assert main["coverage"] == "complete" and len(main["sources"]) == 2
    assert plans[0]["coverage"] == "incomplete"
    far = dict(
        sources[0],
        start_utc=(BASE + timedelta(days=300)).isoformat(),
        end_utc=(BASE + timedelta(days=300, seconds=60)).isoformat(),
    )
    assert len(plan_windows([sources[0], far])) == 6


@pytest.mark.parametrize("skip_empty", [False, True])
def test_analysis_preserves_all_channels_and_artifacts(tmp_path, skip_empty):
    path = write(tmp_path / "a.h5")
    result = analyze_window(
        [path],
        BASE,
        ProcessingProfile(device="cpu", min_coincidence=2),
        tmp_path / "out",
        skip_empty=skip_empty,
    )
    assert result["shape"] == [8, 3000]
    assert result["candidate"] is True
    evidence = json.loads((tmp_path / "out/evidence.json").read_text())
    assert len(evidence) == result["evidence_count"] and len(evidence) > 1
    assert all(
        BASE <= datetime.fromisoformat(e["peak_utc"]) < BASE + timedelta(seconds=30)
        for e in evidence
    )
    arrays = np.load(tmp_path / "out/waveform.npz")
    assert arrays["waveform"].shape == (8, 3000)
    assert arrays["relative_seconds"][-1] == 29.99
    assert len(result["artifacts"]) == 6
    assert result["visualization_version"] == 2
    assert result["detection_regions"]
    assert {a["kind"] for a in result["artifacts"]} == {
        "waveform", "metadata", "evidence", "waterfall", "waterfall_filtered", "filtered_waveform",
    }
    assert not (tmp_path / "out/waveform.png").exists()
    assert json.loads((tmp_path / "out/result.json").read_text()) == result


def test_excluded_raw_values_retained(tmp_path):
    path = write(tmp_path / "a.h5")
    profile = ProcessingProfile(device="cpu", scan_distance_start_m=400.0)
    result = analyze_window([path], BASE, profile, tmp_path / "out")
    arrays = np.load(tmp_path / "out/waveform.npz")
    assert not arrays["recognition_channel_mask"][:4].any()
    assert np.max(np.abs(arrays["waveform"][:4])) > 0.5
    assert result["recognition_channel_mask"] == [False] * 4 + [True] * 4


@pytest.mark.parametrize("skip_empty", [False, True])
def test_insufficient_context_fails(tmp_path, skip_empty):
    with pytest.raises(ValueError):
        analyze_window(
            [write(tmp_path / "short.h5", 0, 60)],
            BASE,
            ProcessingProfile(),
            tmp_path / "out",
            skip_empty=skip_empty,
        )


@pytest.mark.parametrize("skip_empty", [False, True])
def test_changed_source_rejected_before_artifacts(tmp_path, monkeypatch, skip_empty):
    import das_sentinel.whole_window as module

    path = write(tmp_path / "a.h5")
    original = module.read_low_frequency_window

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        with h5py.File(path, "r+") as h:
            h.attrs["mutation"] = "changed"
        return result

    monkeypatch.setattr(module, "read_low_frequency_window", changed)
    with pytest.raises(ValueError, match="source changed"):
        analyze_window(
            [path], BASE, ProcessingProfile(), tmp_path / "out", skip_empty=skip_empty
        )
    assert not (tmp_path / "out/result.json").exists()


def test_evidence_limit_is_explicit_failure(tmp_path, monkeypatch):
    import das_sentinel.whole_window as module

    monkeypatch.setattr(module, "MAX_EVIDENCE_BYTES", 1)
    with pytest.raises(ValueError, match="evidence exceeds"):
        analyze_window(
            [write(tmp_path / "a.h5")], BASE, ProcessingProfile(), tmp_path / "out"
        )
    assert not (tmp_path / "out/result.json").exists()


def test_grid_change_and_gap_plan_incomplete(tmp_path):
    a = write(tmp_path / "a.h5", -60, 120)
    b = write(tmp_path / "b.h5", 60, 120, channels=4)
    plan = next(
        p
        for p in plan_windows(inspect_sources([a, b]))
        if p["start_utc"] == BASE.isoformat()
    )
    assert plan["coverage"] == "incomplete"
    b = write(tmp_path / "gap.h5", 61, 120)
    plan = next(
        p
        for p in plan_windows(inspect_sources([a, b]))
        if p["start_utc"] == BASE.isoformat()
    )
    assert plan["coverage"] == "incomplete"


def test_excluded_signal_does_not_change_measurements(tmp_path):
    first = write(tmp_path / "first.h5")
    second = write(tmp_path / "second.h5")
    with h5py.File(second, "r+") as h:
        data = h["Acquisition/Raw[0]/RawData"]
        data[:, :4] = data[:, :4] * 1000
    profile = ProcessingProfile(device="cpu", scan_distance_start_m=400.0)
    a = analyze_window([first], BASE, profile, tmp_path / "a")
    b = analyze_window([second], BASE, profile, tmp_path / "b")
    assert a["features"] == b["features"]
    assert (a["candidate"], a["score"]) == (b["candidate"], b["score"])
    assert a["input_sha256"] != b["input_sha256"]


def test_measurement_keeps_only_used_metrics_without_refiltering(monkeypatch):
    import das_sentinel.whole_window as module
    from das_sentinel.whole_window import _measure

    rng = np.random.default_rng(42)
    context = rng.normal(size=(12000, 2))
    context[2995:3005] += 30
    def unexpected(*args, **kwargs):
        pytest.fail("measurement must not apply retired frequency filters")

    monkeypatch.setattr(module, "sosfiltfilt", unexpected)
    features = _measure(
        context[3000:9000],
        np.array([300.0, 500.0]),
        np.ones(2, bool),
        np.ones(2, bool),
    )
    assert set(features) == {
        "valid_channel_fraction", "active_channel_fraction",
        "active_contiguous_span_m", "lag_correlation_median",
    }


def test_measurement_empty_recognition_keeps_nulls_and_valid_fraction():
    from das_sentinel.whole_window import _measure

    assert _measure(np.zeros((3000, 3)), np.arange(3.), np.zeros(3, bool), np.zeros(3, bool)) == {
        "valid_channel_fraction": 0.0, "active_channel_fraction": None,
        "active_contiguous_span_m": None, "lag_correlation_median": None,
    }


@pytest.mark.parametrize("excluded", [False, True])
def test_measurement_preserves_signed_delayed_correlation_and_native_span(excluded):
    from das_sentinel.whole_window import _measure

    # Frozen pre-optimization values; gaps and anti-phase arrivals are intentional.
    data = np.random.default_rng(73).normal(0, .01, (3000, 8))
    pulse = np.sin(np.arange(100) * .3) * np.hanning(100)
    for channel in range(8):
        offset = 400 + channel * 9
        data[offset:offset + 100, channel] += pulse * (-1 if channel % 2 else 1)
    recognition = np.array([False] * 4 + [True] * 4) if excluded else np.ones(8, bool)
    result = _measure(data, np.array([-5., -1., 3., 8., 299., 499., 500., 700.]),
                      np.ones(8, bool), recognition)
    assert result["valid_channel_fraction"] == 1.
    assert result["active_channel_fraction"] == 1.
    assert result["active_contiguous_span_m"] == (401. if excluded else 705.)
    assert result["lag_correlation_median"] == pytest.approx(
        -0.9845265215007359 if excluded else -0.9838995409880766, abs=1e-14)


def test_artifact_hashing_does_not_read_entire_files(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path

    def unexpected(*args, **kwargs):
        pytest.fail("artifact hashing must stream files")

    monkeypatch.setattr(Path, "read_bytes", unexpected)
    result = analyze_window([write(tmp_path / "source.h5")], BASE, ProcessingProfile(), tmp_path / "out")
    for artifact in result["artifacts"]:
        with (tmp_path / "out" / artifact["path"]).open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == artifact["sha256"]


def test_complete_evidence_over_64_and_weak_groups_do_not_vote(tmp_path, monkeypatch):
    import das_sentinel.whole_window as module
    from das_sentinel.domain import Candidate
    from das_sentinel.screening import ScreeningResult

    def screen(block, *args):
        candidates = tuple(
            Candidate(
                "energy",
                block.sample_offset + 3000 + i,
                block.sample_offset + 3002 + i,
                block.sample_offset + 3001 + i,
                0,
                2,
                0.1,
            )
            for i in range(70)
        )
        return ScreeningResult(candidates, (), {})

    monkeypatch.setattr(module, "screen_block", screen)
    result = analyze_window(
        [write(tmp_path / "source.h5")], BASE, ProcessingProfile(), tmp_path / "out"
    )
    assert result["evidence_count"] == 140 and result["candidate"] is False
    assert len(json.loads((tmp_path / "out/evidence.json").read_text())) == 140
    import hashlib

    for artifact in result["artifacts"]:
        assert (
            hashlib.sha256(
                (tmp_path / "out" / artifact["path"]).read_bytes()
            ).hexdigest()
            == artifact["sha256"]
        )


def test_conditioning_overflow_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="30 seconds"):
        analyze_window([], BASE, ProcessingProfile(lta_seconds=40), tmp_path / "out")


def test_negative_native_grid_retains_all_channels(tmp_path):
    path = write(tmp_path / "negative.h5")
    with h5py.File(path, "r+") as h:
        h["Acquisition/Raw[0]"].attrs["StartLocusIndex"] = -2
    result = analyze_window([path], BASE, ProcessingProfile(), tmp_path / "out")
    arrays = np.load(tmp_path / "out/waveform.npz")
    assert result["shape"] == [8, 3000]
    np.testing.assert_array_equal(arrays["distance_m"], np.arange(-200, 600, 100))
    assert arrays["waveform"].shape == (8, 3000)
    assert result["read_sources"][0]["channel_start"] == 0
    assert result["read_sources"][0]["channel_stop"] == 8


def test_flat_main_cannot_trigger_from_context(tmp_path):
    path = write(tmp_path / "flat.h5")
    t = -60 + np.arange(18000) / 100
    pulse = np.sin(2 * np.pi * 8 * t) * np.exp(-(((t + 0.1) / 0.15) ** 2)) * (t < 0)
    with h5py.File(path, "r+") as h:
        h["Acquisition/Raw[0]/RawData"][:] = np.repeat(pulse[:, None], 8, axis=1)
    result = analyze_window(
        [path],
        BASE,
        ProcessingProfile(device="cpu", min_coincidence=2),
        tmp_path / "out",
    )
    assert result["recognition_channel_mask"] == [False] * 8
    assert result["candidate"] is False and result["evidence_count"] == 0


def test_empty_window_skip_precedes_measurement_and_all_output(tmp_path, monkeypatch):
    import das_sentinel.whole_window as module
    import das_sentinel.whole_window_plots as plots

    path = write(tmp_path / "flat.h5")
    with h5py.File(path, "r+") as h:
        h["Acquisition/Raw[0]/RawData"][:] = 0

    def unexpected(*args, **kwargs):
        pytest.fail("empty window reached feature measurement or visualization")

    monkeypatch.setattr(module, "_measure", unexpected)
    monkeypatch.setattr(module, "sosfiltfilt", unexpected)
    monkeypatch.setattr(plots, "create_visualization", unexpected)
    output = tmp_path / "artifacts" / "attempt-1"
    result = analyze_window(
        [path], BASE, ProcessingProfile(device="cpu"), output, skip_empty=True
    )
    assert result == {"skipped": True, "reason": "no_qualified_regions"}
    assert not output.parent.exists()


@pytest.mark.parametrize("qualified", [False, True])
def test_skip_uses_qualified_regions_not_candidate_count(tmp_path, monkeypatch, qualified):
    import das_sentinel.whole_window as module
    from das_sentinel.domain import Candidate, EventRecord
    from das_sentinel.screening import ScreeningResult

    def screen(block, *args):
        start = block.sample_offset + 3000
        candidate = Candidate("energy", start, start + 20, start + 10, 0, 2, 1.0)
        events = (
            (EventRecord("event", start, start + 20, start + 10, 0, 2, (candidate,)),)
            if qualified else ()
        )
        return ScreeningResult((candidate,), events, {})

    monkeypatch.setattr(module, "screen_block", screen)
    output = tmp_path / "out"
    result = analyze_window(
        [write(tmp_path / "source.h5")], BASE,
        ProcessingProfile(device="cpu", scan_distance_start_m=400.0),
        output, skip_empty=True,
    )
    assert result == {"skipped": True, "reason": "no_qualified_regions"}
    assert not output.exists()


def test_sixty_second_empty_window_keeps_artifacts_even_with_opt_in(tmp_path):
    path = write(tmp_path / "flat.h5")
    with h5py.File(path, "r+") as h:
        h["Acquisition/Raw[0]/RawData"][:] = 0
    output = tmp_path / "out"
    result = analyze_window(
        [path], BASE, ProcessingProfile(device="cpu"), output,
        window_seconds=60, skip_empty=True,
    )
    assert result["candidate"] is False
    assert result["detection_regions"] == []
    assert (output / "result.json").exists()
    assert len(result["artifacts"]) == 6


def test_actual_native_provenance_and_signature(tmp_path):
    from copy import deepcopy

    from das_sentinel.whole_window import compute_input_sha256

    result = analyze_window(
        [write(tmp_path / "source.h5")], BASE, ProcessingProfile(), tmp_path / "out"
    )
    source = result["read_sources"][0]
    assert (source["sample_start"], source["sample_stop"]) == (1999, 13001)
    assert datetime.fromisoformat(result["read_start_utc"]) == BASE - timedelta(
        seconds=40.01
    )
    assert datetime.fromisoformat(result["read_end_utc"]) == BASE + timedelta(
        seconds=70.01
    )
    assert datetime.fromisoformat(result["conditioning_start_utc"]) == BASE - timedelta(
        seconds=30
    )
    assert datetime.fromisoformat(result["conditioning_end_utc"]) == BASE + timedelta(
        seconds=60
    )
    changed = deepcopy(result)
    changed["read_sources"][0]["sample_start"] += 1
    assert compute_input_sha256(changed) != result["input_sha256"]


def test_site_timezone_reaches_both_plot_titles(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    titles = []
    original = Figure.savefig

    def capture(self, *args, **kwargs):
        titles.append(self.axes[0].get_title())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    result = analyze_window(
        [write(tmp_path / "source.h5")],
        BASE,
        ProcessingProfile(),
        tmp_path / "out",
        site_timezone="America/New_York",
    )
    assert result["start_utc"] == BASE.isoformat()
    assert len(titles) == 2 and all(
        "2023-09-26 23:02:00" in title and "-0400" in title for title in titles
    )
