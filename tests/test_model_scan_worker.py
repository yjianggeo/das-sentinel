import importlib
import io
import json
import queue
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import numpy as np
import pytest
from das_sentinel.worker.main import WorkerClaimLostError, WorkerFatalProtocolError


def module():
    assert importlib.util.find_spec("das_sentinel.worker.model_scan_main"), "model scan worker missing"
    return importlib.import_module("das_sentinel.worker.model_scan_main")


@pytest.mark.parametrize("url,expected", [
    ("http://127.0.0.1:18000", False), ("http://localhost:18000", False),
    ("http://[::1]:18000", False), ("https://sentinel.example", True),
])
def test_local_api_avoids_system_proxy_discovery(url, expected, monkeypatch):
    api = module()
    captured = {}
    def client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(api.httpx, "Client", client)
    api.HttpxModelScanClient(url, "test-token").close()
    assert captured.get("trust_env", True) is expected


def test_negative_submits_receipt_without_creating_artifacts(tmp_path, monkeypatch):
    api = module()
    requests = []
    job = {"id": str(uuid4()), "attempt_count": 1, "kind": "infer", "status": "running", "site_id": str(uuid4()),
           "payload": {"windows": [{"start_utc": "2023-09-27T00:00:00+00:00", "sources": []}]},
           "config": {"root": str(tmp_path), "model": {"id": "frozen", "threshold": .04}}}
    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path.rsplit("/", 1)[-1], body))
        if request.url.path.endswith("/claim"):
            return httpx.Response(200, json=job)
        return httpx.Response(200, json={"status": "negative"})
    def read(*args, **kwargs):
        return {"waveform": np.ones((2, 3000))}, {"model_input_sha256": "a" * 64}
    def forbidden(*args, **kwargs):
        raise AssertionError("negative must not create artifacts")
    monkeypatch.setattr(api, "read_scan_input", read)
    monkeypatch.setattr(api, "checked_metadata", lambda *args: {})
    monkeypatch.setattr(api, "deployment_folder", lambda *args: tmp_path)
    monkeypatch.setattr("das_sentinel.whole_window.analyze_window", forbidden)
    predictor = SimpleNamespace(predict=lambda arrays: {"status": "succeeded", "score": .01,
                                "model_id": "frozen", "label": "absent", "threshold": .04})
    settings = SimpleNamespace(data_root=tmp_path, app_data_dir=tmp_path / "app")
    client = api.HttpxModelScanClient("http://test", "test-token", transport=httpx.MockTransport(handler))
    outcome = api.ModelScanWorker(settings, client, "worker", predictor=predictor).run_once()
    assert outcome.status == "succeeded"
    result = next(body for suffix, body in requests if suffix == "prediction")["result"]
    assert result["score"] == .01 and result["model_id"] == "frozen"
    assert not (settings.app_data_dir / "window-artifacts").exists()
    client.close()


@pytest.mark.parametrize("status,error", [(403, WorkerFatalProtocolError), (409, WorkerClaimLostError)])
def test_worker_http_failures_are_not_noise(status, error):
    api = module()
    client = api.HttpxModelScanClient("http://test", "test", transport=httpx.MockTransport(
        lambda request: httpx.Response(status, json={"detail": "claim_lost"})))
    with pytest.raises(error):
        client.heartbeat(str(uuid4()), "worker", 1)
    client.close()


def test_resident_process_closes_input_and_exits_cleanly(tmp_path, monkeypatch):
    import subprocess
    import sys
    api = module()
    original = subprocess.Popen
    def helper(args, **kwargs):
        return original([sys.executable, "-u", "-c", "import sys; list(sys.stdin)"], **kwargs)
    monkeypatch.setattr(api.subprocess, "Popen", helper)
    process = api.InferenceProcess(tmp_path, {"id": "test"}, tmp_path / "spool", sys.executable)
    process.close()
    assert process.process.returncode == 0
    assert not process.reader.is_alive()


@pytest.mark.parametrize("response,error,message", [
    ({"request_id": "request", "status": "failed", "error_code": "inference_failed",
      "error_kind": "runtime", "error_message": "device inference failure"}, RuntimeError, "device inference failure"),
    ({"request_id": "request", "status": "failed", "error_code": "inference_failed",
      "error_message": "invalid waveform"}, ValueError, "invalid waveform"),
    ({"request_id": "other", "status": "failed", "error_code": "inference_failed",
      "error_kind": "runtime", "error_message": "device inference failure"}, RuntimeError, "response ID mismatch"),
])
def test_resident_process_preserves_failure_class(response, error, message, tmp_path, monkeypatch):
    api = module()
    monkeypatch.setattr(api, "uuid4", lambda: SimpleNamespace(hex="request"))
    process = api.InferenceProcess.__new__(api.InferenceProcess)
    process.request_root = tmp_path
    process.process = SimpleNamespace(stdin=io.StringIO())
    process.responses = queue.Queue()
    process.responses.put(json.dumps(response))
    with pytest.raises(error, match=message):
        process.predict({"waveform": np.ones((2, 3))})
    assert not (tmp_path / "request.npz").exists()


@pytest.fixture
def synthetic_inference_children(monkeypatch):
    import subprocess
    import sys
    api = module()
    monkeypatch.setenv("PYTHONPATH", str(Path(api.__file__).resolve().parents[2]))
    original = subprocess.Popen
    children = []
    script = """
import sys
from das_sentinel.model_scan_inference import serve

class Predictor:
    def predict(self, arrays):
        failure = int(arrays["failure"][0])
        if failure == 1:
            raise RuntimeError("device inference failure")
        if failure == 2:
            raise ValueError("invalid waveform")
        return {"status": "succeeded", "score": .01, "threshold": .04,
                "label": "absent", "model_id": "frozen"}

serve(Predictor(), sys.argv[1], sys.stdin, sys.stdout)
"""
    def helper(args, **kwargs):
        spool = args[args.index("--spool-root") + 1]
        child = original([sys.executable, "-u", "-c", script, spool], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(api.subprocess, "Popen", helper)
    return children


@pytest.mark.parametrize("failure,expected_children,error_message", [
    (1, 2, "device inference failure"), (2, 1, "invalid waveform"),
])
def test_failed_window_preserves_receipt_and_next_window_succeeds(
        failure, expected_children, error_message, tmp_path, monkeypatch, synthetic_inference_children):
    api = module()
    requests, reads = [], []
    starts = ["2023-09-27T00:00:00+00:00", "2023-09-27T00:00:30+00:00", "2023-09-27T00:01:00+00:00"]
    job = {"id": str(uuid4()), "attempt_count": 1, "kind": "infer", "status": "running", "site_id": str(uuid4()),
           "payload": {"windows": [{"start_utc": start, "sources": []} for start in starts]},
           "config": {"root": str(tmp_path), "model": {"id": "frozen", "threshold": .04}}}
    def handler(request):
        body = json.loads(request.content)
        requests.append((request.url.path.rsplit("/", 1)[-1], body))
        if request.url.path.endswith("/claim"):
            return httpx.Response(200, json=job)
        return httpx.Response(200, json={})
    def read(plan, *args, **kwargs):
        reads.append(plan["start_utc"])
        return {"waveform": np.ones((2, 3)), "failure": np.array([failure if len(reads) == 1 else 0])}, {
            "model_input_sha256": "a" * 64}
    monkeypatch.setattr(api, "read_scan_input", read)
    monkeypatch.setattr(api, "checked_metadata", lambda *args: {})
    monkeypatch.setattr(api, "deployment_folder", lambda *args: tmp_path)
    settings = SimpleNamespace(data_root=tmp_path, app_data_dir=tmp_path / "app")
    client = api.HttpxModelScanClient("http://test", "test-token", transport=httpx.MockTransport(handler))
    worker = api.ModelScanWorker(settings, client, "worker")
    try:
        outcome = worker.run_once()
        assert outcome.status == "succeeded"
        results = [body["result"] for suffix, body in requests if suffix == "prediction"]
        assert results == [
            {"start_utc": starts[0], "status": "failed", "error_code": "input_or_inference_failed",
             "error_message": error_message},
            {"start_utc": starts[1], "status": "succeeded", "model_id": "frozen", "score": .01,
             "model_input_sha256": "a" * 64},
            {"start_utc": starts[2], "status": "succeeded", "model_id": "frozen", "score": .01,
             "model_input_sha256": "a" * 64},
        ]
        assert reads == starts
        assert len(synthetic_inference_children) == expected_children
        if failure == 1:
            assert synthetic_inference_children[0].returncode == 0
        assert synthetic_inference_children[-1].poll() is None
    finally:
        worker.close()
        client.close()
    assert all(child.returncode == 0 for child in synthetic_inference_children)


def test_cancelled_claim_does_not_start_inference(tmp_path, synthetic_inference_children):
    api = module()
    requests = []
    job = {"id": str(uuid4()), "attempt_count": 1, "kind": "infer", "status": "cancelled", "site_id": str(uuid4()),
           "payload": {"windows": []}, "config": {"root": str(tmp_path), "model": {"id": "frozen", "threshold": .04}}}
    def handler(request):
        requests.append(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(200, json=job)
    settings = SimpleNamespace(data_root=tmp_path, app_data_dir=tmp_path / "app")
    client = api.HttpxModelScanClient("http://test", "test-token", transport=httpx.MockTransport(handler))
    worker = api.ModelScanWorker(settings, client, "worker")
    try:
        assert worker.run_once().status == "cancelled"
        assert requests == ["claim"]
        assert synthetic_inference_children == []
    finally:
        worker.close()
        client.close()


@pytest.mark.parametrize("matched", [True, False])
def test_materialization_reads_npz_once_for_signature_and_prediction(matched, tmp_path, monkeypatch):
    import hashlib
    from collections import Counter

    api = module()
    expected = {"waveform": np.arange(6000, dtype="f4").reshape(2, 3000),
        "distance_m": np.array([0, 100], dtype="<f8"), "relative_seconds": np.arange(3000) / 100,
        "valid_channel_mask": np.array([True, True]), "recognition_channel_mask": np.array([True, True]),
        "region_id": np.array([0, 0], dtype="i1"), "annotation": np.array(["extra array"]),
        "calibration": np.array([1.25], dtype="f8")}
    path = tmp_path / "normalized.npz"
    np.savez(path, **expected)
    signature = "55acec0b888de57cb412b02ba725924015712cd0e1fc0ced901597e261addac8"
    result = {"input_sha256": "raw-input", "read_sources": [], "unit": "strain_rate", "grid_sha256": "grid",
        "artifacts": [{"kind": "waveform", "path": "waveform.npz", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    output = tmp_path / "window-artifacts" / "analysis" / "attempt-1"
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result))
    job = {"analysis": {"id": "analysis", "attempt_count": 1, "window_id": "window", "reuse": True},
        "payload": {"plan": {"start_utc": "2023-09-27T00:00:00+00:00"},
                     "model_input_sha256": signature if matched else "0" * 64},
        "config": {"model": {"id": "model"}}}
    monkeypatch.setattr(api, "normalize_prediction_row", lambda *args: {"waveform": str(path)})
    reads, supplied = Counter(), []
    original = np.lib.npyio.NpzFile.__getitem__
    def counted(self, key):
        reads[key] += 1
        return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", counted)
    def predict(arrays):
        supplied.append(arrays)
        spool = io.BytesIO()
        np.savez(spool, **arrays)
        spool.seek(0)
        with np.load(spool, allow_pickle=False) as serialized:
            monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", original)
            assert serialized.files == list(expected)
            for key, value in expected.items():
                np.testing.assert_array_equal(serialized[key], value)
        return {"score": .75}
    worker = api.ModelScanWorker(SimpleNamespace(app_data_dir=tmp_path), None, "worker",
        predictor=SimpleNamespace(predict=predict))
    if not matched:
        with pytest.raises(ValueError, match="input_mismatch"):
            worker._materialize(job, {}, tmp_path, tmp_path / "cache")
        assert supplied == []
        return
    binding = worker._materialize(job, {}, tmp_path, tmp_path / "cache")
    assert binding == {"analysis_id": "analysis", "input_sha256": "raw-input",
                          "model_input_sha256": signature, "score": .75}
    assert reads == Counter({key: 1 for key in expected})
    assert isinstance(supplied[0], dict)
