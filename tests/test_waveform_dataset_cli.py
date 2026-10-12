import importlib
import json
from datetime import UTC, datetime, timedelta

import h5py
import numpy as np
from typer.testing import CliRunner

BASE = datetime(2026, 6, 16, 9, 0, tzinfo=UTC)
runner = CliRunner()


def _module():
    return importlib.import_module("das_sentinel.waveform_dataset")


def _source(path, duration=100):
    times = np.arange(duration * 100) / 100
    values = np.sin(2 * np.pi * times[:, None] * np.array([1, 2, 3]))
    with h5py.File(path, "w") as handle:
        raw = handle.create_group("Acquisition/Raw[0]")
        raw.create_dataset("RawData", data=values.astype("f4"))
        raw.create_dataset("RawDataTime", data=np.rint((BASE.timestamp() + times) * 1e6).astype("i8"))
        raw.attrs["OutputDataRate"] = 100
        raw.attrs["SpatialSamplingInterval"] = 4.0
        raw.attrs["StartLocusIndex"] = 0
        raw.attrs["RawDataUnit"] = "strain_rate"
    return path


def test_window_command_exports_loadable_full_distance_array_without_label(tmp_path):
    module = _module()
    sources = tmp_path / "sources"
    sources.mkdir()
    source = _source(sources / "source.h5")
    output = tmp_path / "dataset"
    result = runner.invoke(module.app, [
        "window", "--source-dir", str(sources), "--reference-source", str(source),
        "--output", str(output), "--view", "standard",
        "--start-utc", (BASE + timedelta(seconds=20)).isoformat(),
        "--suspended-end-m", "3", "--borehole-start-m", "6",
    ])
    assert result.exit_code == 0, result.output
    from das_sentinel.services.waveform_samples import load_sample
    arrays, metadata = load_sample(output / "sample")
    assert arrays["waveform"].shape == (3, 6000)
    np.testing.assert_array_equal(arrays["distance_m"], [0, 4, 8])
    np.testing.assert_array_equal(arrays["recognition_channel_mask"], [False, True, True])
    assert metadata["source_kind"] == "unlabelled_window"
    assert metadata["annotations"] == []
    assert metadata["supervised_eligible"] is False
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(report["succeeded"]) == 1
    assert report["failed"] == []
    # A repeat cannot silently replace this frozen sample.
    original = (output / "sample" / "metadata.json").read_bytes()
    repeated = runner.invoke(module.app, [
        "window", "--source-dir", str(sources), "--reference-source", str(source),
        "--output", str(output), "--start-utc", BASE.isoformat(),
    ])
    assert repeated.exit_code != 0
    assert (output / "sample" / "metadata.json").read_bytes() == original


def test_window_missing_real_context_reports_failure_without_array(tmp_path):
    module = _module()
    source = _source(tmp_path / "source.h5")
    output = tmp_path / "failed-window"
    result = runner.invoke(module.app, [
        "window", "--source-dir", str(tmp_path), "--reference-source", str(source),
        "--output", str(output), "--view", "standard", "--start-utc", BASE.isoformat(),
    ])
    assert result.exit_code == 1
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert not report["succeeded"]
    assert "上下文" in report["failed"][0]["reason"]
    assert not list(output.rglob("*.npz"))


def test_build_preserves_full_manifest_groups_and_neighbour_annotations(tmp_path):
    module = _module()
    source = _source(tmp_path / "source.h5")
    from das_sentinel.services.discovery import fingerprint
    grid_hash = module.grid_fingerprint(np.array([0, 4, 8]))
    targets = [{
        "event_id": f"event-{i}", "review_id": f"review-{i}", "label": label,
        "reviewed_at": BASE.isoformat(), "source_path": str(source),
        "source_fingerprint": fingerprint(source), "source_grid_sha256": grid_hash,
        "source_unit": "strain_rate", "source_sample_rate_hz": 100,
        "start_utc": (BASE + timedelta(seconds=48 + i)).isoformat(),
        "end_utc": (BASE + timedelta(seconds=49 + i)).isoformat(),
        "channel_start": 1, "channel_stop": 3, "legacy_features": {"rms": 2},
        "group": "frozen-full-manifest-group",
    } for i, label in enumerate(["earthquake", "traffic"])]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "site_id": "site-1", "targets": targets, "excluded": []}))
    output = tmp_path / "dataset"
    result = runner.invoke(module.app, [
        "build", "--manifest", str(manifest), "--source-dir", str(tmp_path),
        "--reference-source", str(source), "--output", str(output), "--limit", "1",
    ])
    assert result.exit_code == 0, result.output
    from das_sentinel.services.waveform_samples import load_sample
    arrays, metadata = load_sample(output / "event-0")
    assert metadata["group"] == "frozen-full-manifest-group"
    assert metadata["target_event_id"] == "event-0"
    assert {item["event_id"] for item in metadata["annotations"]} == {"event-0", "event-1"}
    assert len(json.loads((output / "manifest.json").read_text(encoding="utf-8"))["targets"]) == 2
    assert arrays["waveform"].shape == (3, 6000)


def test_window_requires_timezone_and_does_not_create_output(tmp_path):
    result = runner.invoke(_module().app, [
        "window", "--source-dir", str(tmp_path), "--reference-source", str(tmp_path / "x.h5"),
        "--output", str(tmp_path / "out"), "--start-utc", "2026-06-16T09:00:00",
    ])
    assert result.exit_code != 0
    assert not (tmp_path / "out").exists()


def test_build_rejects_labels_transferred_to_different_recording(tmp_path):
    module = _module()
    original_dir, substitute_dir = tmp_path / "reviewed", tmp_path / "substitute"
    original_dir.mkdir()
    substitute_dir.mkdir()
    original = _source(original_dir / "reviewed.h5")
    substitute = _source(substitute_dir / "different.h5")
    with h5py.File(substitute, "r+") as handle:
        handle["Acquisition/Raw[0]/RawData"][:] *= 2
    from das_sentinel.services.discovery import fingerprint
    target = {
        "event_id": "reviewed-event", "review_id": 1, "label": "earthquake",
        "reviewed_at": BASE.isoformat(), "source_path": str(original),
        "source_fingerprint": fingerprint(original),
        "source_grid_sha256": module.grid_fingerprint(np.array([0, 4, 8])),
        "source_unit": "strain_rate", "source_sample_rate_hz": 100,
        "start_utc": (BASE + timedelta(seconds=48)).isoformat(),
        "end_utc": (BASE + timedelta(seconds=49)).isoformat(),
        "channel_start": 1, "channel_stop": 3, "legacy_features": {}, "group": "day-1",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "site_id": "site-1", "targets": [target], "excluded": []}), encoding="utf-8")
    output = tmp_path / "dataset"
    result = runner.invoke(module.app, [
        "build", "--manifest", str(manifest), "--source-dir", str(substitute_dir),
        "--reference-source", str(original), "--output", str(output),
    ])
    assert result.exit_code == 1, result.output
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert not report["succeeded"]
    assert "人工标注" in report["failed"][0]["reason"]
    assert not list(output.rglob("*.npz"))


def test_low_frequency_target_without_sampled_annotation_is_not_supervised(tmp_path):
    module = _module()
    source = _source(tmp_path / "long.h5", duration=1300)
    from das_sentinel.services.discovery import fingerprint
    target = {
        "event_id": "short-event", "review_id": 1, "label": "earthquake",
        "reviewed_at": BASE.isoformat(), "source_path": str(source),
        "source_fingerprint": fingerprint(source),
        "source_grid_sha256": module.grid_fingerprint(np.array([0, 4, 8])),
        "source_unit": "strain_rate", "source_sample_rate_hz": 100,
        "start_utc": (BASE + timedelta(seconds=650.01)).isoformat(),
        "end_utc": (BASE + timedelta(seconds=650.02)).isoformat(),
        "channel_start": 1, "channel_stop": 3, "legacy_features": {}, "group": "day-1",
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "site_id": "site-1", "targets": [target], "excluded": []}), encoding="utf-8")
    output = tmp_path / "dataset"
    result = runner.invoke(module.app, [
        "build", "--manifest", str(manifest), "--source-dir", str(tmp_path),
        "--reference-source", str(source), "--output", str(output), "--view", "low_frequency",
    ])
    assert result.exit_code == 1, result.output
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert not report["succeeded"]
    assert "有效标注" in report["failed"][0]["reason"]
    assert not list(output.rglob("*.npz"))
