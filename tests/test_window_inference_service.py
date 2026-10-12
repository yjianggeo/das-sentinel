import hashlib
import importlib
import json
import threading
import time
from uuid import uuid4

import pytest
from das_sentinel.persistence import session_factory
from das_sentinel.persistence.window_models import WindowAnalysis, WindowReview
from das_sentinel.services.event_windows import enqueue_windows
from das_sentinel.window_inference import normalize_prediction_row, save_json
from sqlalchemy import func, select

from tests import test_event_windows as original
from tests.test_window_grid import sample

context = original.context


def module():
    assert importlib.util.find_spec('das_sentinel.services.window_inference') is not None, 'prediction service missing'
    return importlib.import_module('das_sentinel.services.window_inference')


def publish_fixture(root, site):
    mid = str(uuid4())
    folder = root / 'window-models' / site / 'deployments' / mid
    folder.mkdir(parents=True)
    (folder / 'weights.pt').write_bytes(b'test weight')
    meta = {'schema_version': 1, 'task': 'earthquake_including_blast_vs_noise', 'candidate': 'residual_waveform',
            'window_seconds': 30, 'sample_rate_hz': 100, 'distance_start_m': 300,
            'preprocess': 'whole-window-signed-waveform-v1-30s', 'unit': 'strain',
            'weights_sha256': hashlib.sha256(b'test weight').hexdigest(), 'metrics': {'threshold': .04}}
    save_json(folder / 'model.json', meta)
    model = {'id': mid, 'site_id': site, 'name': '300 m CNN', 'threshold': .04, 'distance_start_m': 300,
             'metadata_sha256': hashlib.sha256((folder / 'model.json').read_bytes()).hexdigest(),
             'weights_sha256': meta['weights_sha256']}
    save_json(folder / 'deployment.json', model)
    save_json(root / 'window-models' / site / 'active.json', {'id': mid})
    return model


def successful(request_path, output):
    request = json.loads(request_path.read_text(encoding='utf8'))
    save_json(output, {key: request[key] for key in ('window_id', 'analysis_id', 'input_sha256')} | {
        'status': 'succeeded', 'model_id': request['model']['id'], 'model_name': request['model']['name'],
        'score': .7, 'threshold': .04, 'label': 'present', 'distance_start_m': 300})


def wait_result(service, wid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = service.prediction(wid)
        if value['status'] != 'pending':
            return value
        time.sleep(.01)
    pytest.fail('prediction did not finish')


def setup_window(context):
    session, site, profile, windows = context
    planned = original.plan() | {'end_utc': '2023-09-27T00:00:30+00:00'}
    analysis = enqueue_windows(session, site.id, profile.id, [planned])[0]
    windows.claim('worker')
    result = original.make_result(windows, analysis)
    analysis.status = 'succeeded'
    analysis.input_sha256 = result['input_sha256']
    analysis.result_json = json.dumps(result)
    session.commit()
    return analysis.window_id, analysis.id, site.id, windows.artifact_root.parent


def test_prediction_is_deduplicated_cached_and_does_not_write_labels(context, monkeypatch):
    api = module()
    wid, aid, site, root = setup_window(context)
    publish_fixture(root, site)
    monkeypatch.setattr(api, 'normalize_prediction_row', lambda row, meta, root: row)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def runner(request, output):
        calls.append(request)
        entered.set()
        assert release.wait(5)
        successful(request, output)

    service = api.WindowInferenceService(session_factory(context[0].bind), root, runner=runner)
    try:
        assert service.prediction(wid)['status'] == 'pending'
        assert entered.wait(3)
        assert service.prediction(wid)['status'] == 'pending'
        release.set()
        result = wait_result(service, wid)
        assert result['label'] == 'present' and result['analysis_id'] == aid
        assert service.prediction(wid) == result
        assert len(calls) == 1
        assert context[0].scalar(select(func.count()).select_from(WindowReview)) == 0
        assert service.prediction(wid, analysis_id=str(uuid4()))['status'] == 'stale'
    finally:
        release.set()
        service.close()


def test_model_switch_and_input_change_never_reuse_old_prediction(context, monkeypatch):
    api = module()
    wid, aid, site, root = setup_window(context)
    first = publish_fixture(root, site)
    monkeypatch.setattr(api, 'normalize_prediction_row', lambda row, meta, root: row)
    service = api.WindowInferenceService(session_factory(context[0].bind), root, runner=successful)
    try:
        assert wait_result(service, wid)['model_id'] == first['id']
        second = publish_fixture(root, site)
        assert wait_result(service, wid)['model_id'] == second['id']
        analysis = context[0].get(WindowAnalysis, aid)
        analysis.input_sha256 = 'b' * 64
        context[0].commit()
        assert wait_result(service, wid)['input_sha256'] == 'b' * 64
    finally:
        service.close()


def test_failed_inference_is_not_noise_and_explicit_retry_recovers(context, monkeypatch):
    api = module()
    wid, _, site, root = setup_window(context)
    publish_fixture(root, site)
    monkeypatch.setattr(api, 'normalize_prediction_row', lambda row, meta, root: row)

    def broken(request, output):
        raise RuntimeError('isolated interpreter failed')

    service = api.WindowInferenceService(session_factory(context[0].bind), root, runner=broken)
    try:
        result = wait_result(service, wid)
        assert result['status'] == 'failed' and 'label' not in result
        service.runner = successful
        assert service.prediction(wid, retry=True)['status'] == 'pending'
        assert wait_result(service, wid)['status'] == 'succeeded'
    finally:
        service.close()


def test_api_normalization_uses_native_coordinates_model_range_and_fixed_grid(tmp_path):
    import numpy as np

    row, wave = sample(tmp_path, 'input', 100, spacing=4.1, nominal=4)
    distance = np.arange(102, dtype='<f8') * 4.1
    meta = {'unit': 'strain', 'distance_start_m': 300,
            'training_grid': {'version': 'native-channel-union-v1', 'start_locus': 0,
                              'channels': 102, 'spacing_m': 4.1,
                              'grid_sha256': hashlib.sha256(distance.tobytes()).hexdigest()}}
    prepared = normalize_prediction_row(row, meta, tmp_path / 'grid')
    with np.load(prepared['waveform']) as arrays:
        np.testing.assert_array_equal(arrays['distance_m'], distance)
        np.testing.assert_array_equal(arrays['waveform'][:100], wave)
        assert not arrays['valid_channel_mask'][-2:].any()
        assert not arrays['recognition_channel_mask'][distance < 300].any()
        assert arrays['recognition_channel_mask'][74:100].all()
    with pytest.raises(ValueError, match='unit'):
        normalize_prediction_row(row | {'unit': 'wrong'}, meta, tmp_path / 'grid')
