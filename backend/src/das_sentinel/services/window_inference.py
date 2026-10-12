"""Background predictions bound to immutable window inputs and published models."""

import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from das_sentinel.persistence.window_models import EventWindow, WindowAnalysis
from das_sentinel.services.event_windows import WindowError, WindowService
from das_sentinel.window_inference import (
    checked_metadata,
    deployment_folder,
    normalize_prediction_row,
    read_active,
    save_json,
)

logger = logging.getLogger(__name__)


class WindowInferenceService:
    def __init__(self, session_factory, app_data_dir, *, runner=None):
        self.session_factory = session_factory
        self.root = Path(app_data_dir)
        self.models = self.root / 'window-models'
        self.cache = self.root / 'window-predictions'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='window-inference')
        self.lock = threading.Lock()
        self.pending = set()
        self.closed = False
        self.process = None
        self.runner = runner or self._subprocess

    def active(self, site_id):
        return read_active(self.models, site_id)

    def _request(self, wid, analysis_id=None, input_sha256=None):
        with self.session_factory() as session:
            window = session.get(EventWindow, wid)
            if window is None:
                raise WindowError('window_not_found', 404)
            model = self.active(window.site_id)
            if model is None:
                return {'status': 'inactive'}
            analysis = session.get(WindowAnalysis, window.latest_analysis_id)
            if analysis is None:
                return {'status': 'unavailable'}
            if ((analysis_id is not None and analysis_id != window.latest_analysis_id)
                    or (input_sha256 is not None and input_sha256 != analysis.input_sha256)):
                return {'status': 'stale'}
            windows = WindowService(session, self.root / 'window-artifacts')
            if (window.window_seconds != 30 or analysis.status != 'succeeded'
                    or not windows._current(analysis)):
                return {'status': 'unavailable'}
            result = json.loads(analysis.result_json)
            if not result.get('read_sources'):
                return {'status': 'unavailable'}
            path, _ = windows.artifact(wid, 'waveform', analysis.id)
            waveform = next(a for a in result['artifacts'] if a['kind'] == 'waveform')
            row = {'window_id': wid, 'input_sha256': analysis.input_sha256, 'window_seconds': 30,
                   'sources': result['read_sources'], 'unit': result['unit'], 'grid_sha256': result['grid_sha256'],
                   'recognition_range_m': result.get('recognition_range_m', {'start': None, 'stop': None}),
                   'waveform': str(path), 'waveform_sha256': waveform['sha256']}
            return {'window_id': wid, 'analysis_id': analysis.id, 'input_sha256': analysis.input_sha256,
                    'model': model, 'model_folder': str(deployment_folder(self.models, window.site_id, model['id'])),
                    'row': row}

    @staticmethod
    def _binding(request):
        return {key: request[key] for key in ('window_id', 'analysis_id', 'input_sha256')} | {
            'model_id': request['model']['id']}

    @classmethod
    def _validate_result(cls, result, request):
        if any(result.get(key) != value for key, value in cls._binding(request).items()):
            raise ValueError('prediction result binding mismatch')
        if result['status'] == 'failed':
            if 'label' in result or 'score' in result:
                raise ValueError('failed prediction cannot have a label')
            return
        score = result.get('score')
        threshold = request['model']['threshold']
        if (result['status'] != 'succeeded' or isinstance(score, bool)
                or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1
                or result.get('threshold') != threshold
                or result.get('label') != ('present' if score >= threshold else 'absent')):
            raise ValueError('invalid prediction result')

    def prediction(self, wid, *, analysis_id=None, input_sha256=None, retry=False):
        request = self._request(wid, analysis_id, input_sha256)
        if 'status' in request:
            return request
        binding = self._binding(request)
        key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
        folder = self.cache / key
        folder.mkdir(exist_ok=True)
        cached = folder / 'prediction.json'
        with self.lock:
            if key in self.pending:
                return binding | {'status': 'pending', 'model_name': request['model']['name']}
            if cached.exists():
                result = json.loads(cached.read_text(encoding='utf8'))
                self._validate_result(result, request)
                if not (retry and result['status'] == 'failed'):
                    return result
            if self.closed:
                return {'status': 'unavailable'}
            if key not in self.pending:
                self.pending.add(key)
                self.pool.submit(self._run, key, folder, request)
        return binding | {'status': 'pending', 'model_name': request['model']['name']}

    def _run(self, key, folder, request):
        try:
            metadata = checked_metadata(request['model_folder'], request['model'])
            request['row'] = normalize_prediction_row(request['row'], metadata,
                self.root / 'window-inference-inputs' / request['model']['id'])
            request_path, response_path = folder / 'request.json', folder / 'response.json'
            save_json(request_path, request)
            response_path.unlink(missing_ok=True)
            self.runner(request_path, response_path)
            result = json.loads(response_path.read_text(encoding='utf8'))
            self._validate_result(result, request)
            if result['status'] != 'succeeded':
                raise ValueError('inference process did not succeed')
        except Exception:  # Persist a failed prediction without failing the scan or inventing a class.
            logger.exception('Window inference failed for %s', request['window_id'])
            result = self._binding(request) | {'status': 'failed', 'error': '模型识别失败，可重试识别。'}
        finally:
            with self.lock:
                try:
                    if 'result' in locals():
                        save_json(folder / 'prediction.json', result)
                finally:
                    self.pending.discard(key)

    def _subprocess(self, request_path, output_path):
        python = os.environ.get('DAS_INFERENCE_PYTHON') or os.environ.get('DAS_TRAINING_PYTHON') or sys.executable
        with self.lock:
            if self.closed:
                raise RuntimeError('inference service closing')
            process = subprocess.Popen(
                [python, '-m', 'das_sentinel.window_inference', 'predict',
                 '--request', str(request_path), '--output', str(output_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8',
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            self.process = process
        try:
            _, error = process.communicate(timeout=120)
            if process.returncode:
                raise RuntimeError(f'inference process failed: {error[-2000:]}')
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        finally:
            with self.lock:
                self.process = None

    def close(self):
        with self.lock:
            self.closed = True
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
        self.pool.shutdown(wait=True, cancel_futures=True)
