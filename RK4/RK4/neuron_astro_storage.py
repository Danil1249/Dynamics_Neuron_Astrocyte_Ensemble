import json
from datetime import datetime

import numpy as np


CALCULATION_TEXT_FORMAT = 'neuron_astro_calculation_text'
CALCULATION_TEXT_VERSION = 2


def array_to_text_payload(value):
    try:
        arr = np.asarray(value)
        return {
            'dtype': str(arr.dtype),
            'shape': list(arr.shape),
            'data': arr.tolist(),
        }
    except Exception:
        return {
            'dtype': 'object',
            'shape': [],
            'data': value,
        }


def text_payload_to_array(payload):
    if not isinstance(payload, dict) or 'data' not in payload:
        return np.asarray([])
    dtype = str(payload.get('dtype', 'float64'))
    try:
        arr = np.asarray(payload.get('data', []), dtype=float if dtype != 'object' else object)
    except Exception:
        arr = np.asarray(payload.get('data', []))
    try:
        shape = payload.get('shape') or []
        if shape:
            arr = arr.reshape(tuple(int(item) for item in shape))
    except Exception:
        pass
    return arr


def dataset_to_text(dataset, record=None, app_version=''):
    record = record or {}
    kind = str(dataset.get('kind', ''))
    meta = dict(dataset.get('meta', {}) or {})
    meta.update({
        'tab_title': record.get('tab_title', meta.get('tab_title', '')),
        'title': record.get('title', meta.get('title', '')),
        'time': record.get('time', meta.get('time', '')),
        'method': record.get('method', meta.get('method', '')),
        'elapsed': record.get('elapsed', meta.get('elapsed', '')),
        'params': record.get('params', meta.get('params', '')),
        'kind': kind,
    })
    arrays = {}
    for key_name, value in dataset.items():
        if key_name in ('kind', 'meta'):
            continue
        arrays[key_name] = array_to_text_payload(value)
    payload = {
        'format': CALCULATION_TEXT_FORMAT,
        'version': CALCULATION_TEXT_VERSION,
        'app_version': app_version,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'meta': meta,
        'arrays': arrays,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_text_dataset(path):
    with open(path, 'r', encoding='utf-8-sig') as file:
        payload = json.load(file)
    if not isinstance(payload, dict) or payload.get('format') != CALCULATION_TEXT_FORMAT:
        raise ValueError('Это не текстовый файл расчёта этой программы.')
    dataset = {
        'kind': str((payload.get('meta') or {}).get('kind', '')),
        'meta': dict(payload.get('meta') or {}),
    }
    dataset['meta']['file_format'] = str(payload.get('format', ''))
    dataset['meta']['file_version'] = int(payload.get('version', 0) or 0)
    dataset['meta']['app_version'] = str(payload.get('app_version', ''))
    dataset['meta']['created_at'] = str(payload.get('created_at', ''))
    for key_name, value in (payload.get('arrays') or {}).items():
        dataset[key_name] = text_payload_to_array(value)
    return dataset
