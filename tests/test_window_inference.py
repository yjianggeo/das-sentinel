import hashlib
import importlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest


def runtime():
    assert importlib.util.find_spec("das_sentinel.window_inference") is not None, "online window inference missing"
    return importlib.import_module("das_sentinel.window_inference")


@pytest.fixture
def candidate(tmp_path):
    torch = pytest.importorskip("torch")
    from das_sentinel.classifiers.window_resnet import ResidualWindowEncoder
    distance = np.arange(100, dtype='<f8') * 4.1
    waveform = tmp_path / 'normalized.npz'
    np.savez(waveform, waveform=np.arange(300000, dtype=np.float32).reshape(100, 3000),
             distance_m=distance, relative_seconds=np.arange(3000)/100,
             valid_channel_mask=np.ones(100, bool), recognition_channel_mask=distance >= 300,
             region_id=np.where(distance < 299, 0, np.where(distance < 499, 1, 2)))
    grid = {'version': 'native-channel-union-v1', 'start_locus': 0, 'spacing_m': 4.1,
            'channels': 100, 'grid_sha256': hashlib.sha256(distance.tobytes()).hexdigest()}
    row = {'window_id': 'window', 'input_sha256': 'input', 'window_seconds': 30, 'unit': 'strain',
           'waveform': str(waveform), 'waveform_sha256': hashlib.sha256(waveform.read_bytes()).hexdigest()}
    manifest = tmp_path / "manifest.json"
    site = str(uuid4())
    manifest.write_text(json.dumps({"site_id": site}))
    folder = tmp_path / "candidate"
    folder.mkdir()
    torch.manual_seed(42)
    torch.save(ResidualWindowEncoder("waveform").state_dict(), folder / "weights.pt")
    metadata = {"schema_version": 1, "task": "earthquake_including_blast_vs_noise",
                "candidate": "residual_waveform", "window_seconds": 30, "sample_rate_hz": 100,
                "unit": "strain", "training_grid": grid, "distance_start_m": 300.0,
                "input_channels": int((np.arange(100) * 4.1 >= 300).sum()),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "weights_sha256": hashlib.sha256((folder / "weights.pt").read_bytes()).hexdigest(),
                "metrics": {"threshold": .2, "selected_epoch": 7},
                "preprocess": "whole-window-signed-waveform-v1-30s"}
    (folder / "model.json").write_text(json.dumps(metadata))
    return folder, manifest, site, row, row, metadata


def test_published_prediction_exactly_replays_offline_model(candidate, tmp_path):
    import torch
    from das_sentinel.window_benchmark import candidate_inputs, load_candidate

    api = runtime()
    folder, manifest, site, raw, normalized, metadata = candidate
    models = tmp_path / "models"
    published = api.publish_model(folder, manifest, models, site, "300 m CNN")
    assert api.read_active(models, site) == published
    with np.load(normalized["waveform"]) as arrays, torch.inference_mode():
        expected = load_candidate(folder)(*candidate_inputs(arrays, metadata)).sigmoid().item()
    request = {"window_id": "window", "analysis_id": "analysis", "input_sha256": raw["input_sha256"],
               "model": published, "model_folder": str(api.deployment_folder(models, site, published["id"])),
               "row": raw}
    result = api.predict_request(request)
    assert result["status"] == "succeeded"
    assert result["score"] == pytest.approx(expected, abs=1e-7)
    assert result["label"] == ("present" if expected >= .2 else "absent")
    assert result["model_id"] == published["id"]
    assert result["analysis_id"] == "analysis" and result["input_sha256"] == raw["input_sha256"]
    assert result["threshold"] == .2


def test_publication_validates_site_hashes_and_keeps_previous_model(candidate, tmp_path):
    api = runtime()
    folder, manifest, site, *_ = candidate
    with pytest.raises(ValueError, match="site"):
        api.publish_model(folder, manifest, tmp_path / "models", str(uuid4()), "wrong site")
    first = api.publish_model(folder, manifest, tmp_path / "models", site, "first")
    second = api.publish_model(folder, manifest, tmp_path / "models", site, "second")
    assert second["previous_id"] == first["id"]
    assert api.deployment_folder(tmp_path / "models", site, first["id"]).is_dir()
    (folder / "weights.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        api.publish_model(folder, manifest, tmp_path / "models", site, "corrupt")
    assert api.read_active(tmp_path / "models", site)["id"] == second["id"]


def test_prediction_rejects_wrong_units_source_changes_and_modified_bundle(candidate, tmp_path):
    api = runtime()
    folder, manifest, site, raw, *_ = candidate
    models = tmp_path / "models"
    published = api.publish_model(folder, manifest, models, site, "model")
    deployed = api.deployment_folder(models, site, published["id"])
    request = {"window_id": "window", "analysis_id": "analysis", "input_sha256": raw["input_sha256"],
               "model": published, "model_folder": str(deployed), "row": raw | {"unit": "wrong"}}
    with pytest.raises(ValueError, match="unit"):
        api.predict_request(request)
    request["row"] = raw
    Path(raw["waveform"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        api.predict_request(request)
    (deployed / "model.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        api.read_active(models, site)


def test_no_active_model_and_path_traversal_are_safe(tmp_path):
    api = runtime()
    assert api.read_active(tmp_path, str(uuid4())) is None
    with pytest.raises(ValueError):
        api.read_active(tmp_path, "../escape")
