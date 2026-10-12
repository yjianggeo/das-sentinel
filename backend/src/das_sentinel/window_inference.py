"""Immutable whole-window model publication and isolated Torch inference."""

import argparse
import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'{uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf8')
    temporary.replace(path)


def identifier(value):
    if str(UUID(value)) != value:
        raise ValueError('noncanonical identifier')
    return value


def deployment_folder(root, site_id, model_id):
    return Path(root) / identifier(site_id) / 'deployments' / identifier(model_id)


def checked_metadata(folder, model=None):
    folder = Path(folder)
    if model is not None and file_sha(folder / 'model.json') != model['metadata_sha256']:
        raise ValueError('model metadata hash mismatch')
    meta = json.loads((folder / 'model.json').read_text(encoding='utf8'))
    if (meta.get('schema_version') != 1 or meta.get('task') != 'earthquake_including_blast_vs_noise'
            or meta.get('candidate') != 'residual_waveform' or meta.get('window_seconds') != 30
            or meta.get('sample_rate_hz') != 100 or meta.get('distance_start_m') != 300
            or meta.get('preprocess') != 'whole-window-signed-waveform-v1-30s'):
        raise ValueError('unsupported online model input protocol')
    threshold = meta['metrics']['threshold']
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('invalid model threshold')
    if file_sha(folder / 'weights.pt') != meta['weights_sha256']:
        raise ValueError('model weights hash mismatch')
    if model is not None and (meta['weights_sha256'] != model['weights_sha256']
                              or threshold != model['threshold']):
        raise ValueError('deployment metadata mismatch')
    return meta


def read_active(root, site_id):
    site = Path(root) / identifier(site_id)
    pointer = site / 'active.json'
    if not pointer.exists():
        return None
    model_id = json.loads(pointer.read_text(encoding='utf8'))['id']
    folder = deployment_folder(root, site_id, model_id)
    model = json.loads((folder / 'deployment.json').read_text(encoding='utf8'))
    if model['id'] != model_id or model['site_id'] != site_id:
        raise ValueError('deployment site mismatch')
    checked_metadata(folder, model)
    return model


def publish_model(source, manifest_path, root, site_id, name):
    from das_sentinel.window_benchmark import load_candidate

    identifier(site_id)
    if not name.strip() or len(name) > 160:
        raise ValueError('invalid model name')
    source, root = Path(source), Path(root)
    meta = checked_metadata(source)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf8'))
    if manifest['site_id'] != site_id:
        raise ValueError('model training site mismatch')
    if file_sha(manifest_path) != meta['manifest_sha256']:
        raise ValueError('model manifest hash mismatch')
    load_candidate(source)  # Also validates the actual architecture/state dictionary.
    previous = read_active(root, site_id)
    model_id = str(uuid4())
    folder = deployment_folder(root, site_id, model_id)
    folder.mkdir(parents=True)
    for filename in ('model.json', 'weights.pt'):
        shutil.copyfile(source / filename, folder / filename)
        if file_sha(source / filename) != file_sha(folder / filename):
            raise ValueError('published artifact hash mismatch')
    model = {'id': model_id, 'site_id': site_id, 'name': name, 'status': 'active',
             'published_at': datetime.now(UTC).isoformat(),
             'previous_id': previous['id'] if previous else None,
             'metadata_sha256': file_sha(folder / 'model.json'), 'weights_sha256': meta['weights_sha256'],
             'manifest_sha256': meta['manifest_sha256'], 'threshold': meta['metrics']['threshold'],
             'selected_epoch': meta['metrics']['selected_epoch'], 'distance_start_m': 300,
             'window_seconds': 30, 'sample_rate_hz': 100, 'input_channels': meta['input_channels']}
    save_json(folder / 'deployment.json', model)
    save_json(root / site_id / 'active.json', {'id': model_id})
    return model


def normalize_prediction_row(row, metadata, cache_root):
    from das_sentinel.window_grid import normalize_rows

    if row['unit'] != metadata['unit'] or row['window_seconds'] != 30:
        raise ValueError('model unit or window length mismatch')
    cache = Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    grid = metadata['training_grid']
    contract = {key: grid[key] for key in ('version', 'spacing_m', 'start_locus', 'channels')}
    grid_path = cache / 'grid.json'
    if grid_path.exists():
        if json.loads(grid_path.read_text(encoding='utf8')) != contract:
            raise ValueError('model normalization grid mismatch')
    else:
        save_json(grid_path, contract)
    selected = row | {'recognition_range_m': {'start': metadata['distance_start_m'], 'stop': None}}
    rows, actual = normalize_rows([selected], cache)
    if actual['grid_sha256'] != grid['grid_sha256']:
        raise ValueError('model source grid mismatch')
    return rows[0]


def predict_request(request):
    import numpy as np
    import torch

    from das_sentinel.window_benchmark import candidate_inputs, load_candidate

    meta = checked_metadata(request['model_folder'], request['model'])
    row = request['row']
    if row['unit'] != meta['unit'] or row['window_seconds'] != 30:
        raise ValueError('model unit or window length mismatch')
    if row['input_sha256'] != request['input_sha256'] or row['window_id'] != request['window_id']:
        raise ValueError('prediction input binding mismatch')
    if file_sha(row['waveform']) != row['waveform_sha256']:
        raise ValueError('waveform hash mismatch')
    torch.set_num_threads(2)
    model = load_candidate(request['model_folder'])
    with np.load(row['waveform'], allow_pickle=False) as arrays, torch.inference_mode():
        score = model(*candidate_inputs(arrays, meta)).sigmoid().item()
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('invalid model score')
    threshold = meta['metrics']['threshold']
    return {'status': 'succeeded', 'window_id': request['window_id'], 'analysis_id': request['analysis_id'],
            'input_sha256': request['input_sha256'], 'model_id': request['model']['id'],
            'model_name': request['model']['name'], 'score': score, 'threshold': threshold,
            'label': 'present' if score >= threshold else 'absent',
            'distance_start_m': meta['distance_start_m'], 'created_at': datetime.now(UTC).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    publish = commands.add_parser('publish')
    for flag in ('source', 'manifest', 'root', 'site-id', 'name'):
        publish.add_argument('--' + flag, required=True)
    predict = commands.add_parser('predict')
    predict.add_argument('--request', type=Path, required=True)
    predict.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'publish':
        print(json.dumps(publish_model(args.source, args.manifest, args.root, args.site_id, args.name)))
    else:
        save_json(args.output, predict_request(json.loads(args.request.read_text(encoding='utf8'))))


if __name__ == '__main__':
    main()
