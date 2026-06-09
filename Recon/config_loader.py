from copy import deepcopy
from pathlib import Path

import yaml


def _read_yaml(path):
    with open(path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError('configuration root must be a mapping: {}'.format(path))
    return config


def _validate_override_keys(defaults, overrides, prefix=''):
    for key, value in overrides.items():
        key_path = '{}.{}'.format(prefix, key) if prefix else str(key)
        if key not in defaults:
            raise KeyError('unknown configuration key: {}'.format(key_path))
        if isinstance(value, dict):
            if not isinstance(defaults[key], dict):
                raise TypeError(
                    'configuration key {} does not accept nested values'.format(
                        key_path
                    )
                )
            _validate_override_keys(defaults[key], value, key_path)


def _deep_merge(defaults, overrides):
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(config_path):
    config_path = Path(config_path).resolve()
    overrides = _read_yaml(config_path)
    default_name = overrides.pop('defaults', 'default.yaml')
    default_path = Path(default_name)
    if not default_path.is_absolute():
        default_path = config_path.parent / default_path

    defaults = _read_yaml(default_path)
    _validate_override_keys(defaults, overrides)
    return _deep_merge(defaults, overrides)
