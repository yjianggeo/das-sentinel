import importlib
import io
import json

import numpy as np
import pytest
from das_sentinel.window_inference import file_sha, predict_request

from tests.test_window_inference import candidate as candidate  # noqa: PLC0414


def module():
    assert importlib.util.find_spec("das_sentinel.model_scan_inference"), "resident predictor missing"
    return importlib.import_module("das_sentinel.model_scan_inference")


def test_resident_replays_reference_without_reloading(candidate, monkeypatch):
    api = module()
    folder, _, _, row, _, meta = candidate
    snapshot = {"id": "fixed", "name": "fixed", "metadata_sha256": file_sha(folder / "model.json"),
                "weights_sha256": meta["weights_sha256"], "threshold": .2}
    request = {"model_folder": str(folder), "model": snapshot, "row": row,
               "window_id": row["window_id"], "analysis_id": "analysis", "input_sha256": row["input_sha256"]}
    expected = predict_request(request)
    predictor = api.ResidentPredictor(folder, snapshot)
    def cannot_reload(*args, **kwargs):
        raise AssertionError("weights must stay loaded")
    monkeypatch.setattr("das_sentinel.window_benchmark.load_candidate", cannot_reload)
    with np.load(row["waveform"], allow_pickle=False) as arrays:
        for _ in range(2):
            actual = predictor.predict(arrays)
            assert actual["score"] == pytest.approx(expected["score"], abs=1e-7)
            assert actual["label"] == expected["label"]


def test_stream_rejects_bad_hash_path_and_continues(candidate, tmp_path):
    api = module()
    folder, _, _, row, _, meta = candidate
    snapshot = {"id": "fixed", "name": "fixed", "metadata_sha256": file_sha(folder / "model.json"),
                "weights_sha256": meta["weights_sha256"], "threshold": .2}
    request = {"request_id": "one", "path": row["waveform"], "sha256": "a" * 64}
    stream = io.StringIO("{truncated\n" + json.dumps(request) + "\n" + json.dumps(
        request | {"request_id": "two", "sha256": row["waveform_sha256"]}) + "\n")
    output = io.StringIO()
    api.serve(api.ResidentPredictor(folder, snapshot), tmp_path, stream, output)
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [r["status"] for r in replies] == ["failed", "failed", "succeeded"]
    assert replies[-1]["request_id"] == "two"
    assert "score" not in replies[1]
    outside = io.StringIO(json.dumps(request | {"path": str(tmp_path.parent / "outside.npz")}) + "\n")
    output = io.StringIO()
    api.serve(api.ResidentPredictor(folder, snapshot), tmp_path, outside, output)
    assert json.loads(output.getvalue())["status"] == "failed"


@pytest.mark.parametrize("error,expected_kind", [
    (RuntimeError("device inference failure"), "runtime"),
    (ValueError("invalid waveform"), None),
])
def test_stream_preserves_inference_failure_kind(error, expected_kind, tmp_path):
    api = module()
    path = tmp_path / "input.npz"
    np.savez(path, waveform=np.ones((2, 3)))
    request = {"request_id": "one", "path": str(path), "sha256": file_sha(path)}
    class Predictor:
        def predict(self, arrays):
            raise error
    output = io.StringIO()
    api.serve(Predictor(), tmp_path, io.StringIO(json.dumps(request) + "\n"), output)
    reply = json.loads(output.getvalue())
    assert reply["status"] == "failed"
    assert reply["error_code"] == "inference_failed"
    assert reply["error_message"] == str(error)
    assert reply["request_id"] == "one"
    assert reply.get("error_kind") == expected_kind
    assert "score" not in reply


@pytest.mark.parametrize("bad_input", ["hash", "path", "format"])
def test_stream_invalid_input_is_not_a_runtime_failure(bad_input, tmp_path):
    api = module()
    spool = tmp_path / "spool"
    spool.mkdir()
    path = (tmp_path if bad_input == "path" else spool) / "input.npz"
    if bad_input == "format":
        path.write_bytes(b"invalid npz")
    else:
        np.savez(path, waveform=np.ones((2, 3)))
    request = {"request_id": "one", "path": str(path),
               "sha256": "a" * 64 if bad_input == "hash" else file_sha(path)}
    class Predictor:
        def predict(self, arrays):
            raise AssertionError("invalid input must not reach the model")
    output = io.StringIO()
    api.serve(Predictor(), spool, io.StringIO(json.dumps(request) + "\n"), output)
    reply = json.loads(output.getvalue())
    assert reply["status"] == "failed"
    assert reply["error_code"] == "inference_failed"
    assert reply["request_id"] == "one"
    assert "error_kind" not in reply
    assert "score" not in reply
