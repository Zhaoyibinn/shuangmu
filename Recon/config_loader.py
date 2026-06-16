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
    default_name = overrides.pop('defaults', None)
    if config_path.name == 'default.yaml':
        if default_name is not None:
            raise ValueError('default.yaml cannot inherit another config')
        return overrides

    if default_name is None:
        default_name = 'default.yaml'
    default_path = (config_path.parent / default_name).resolve()
    expected_default_path = (config_path.parent / 'default.yaml').resolve()
    if default_path != expected_default_path:
        raise ValueError(
            'configuration {} may only inherit sibling default.yaml'.format(
                config_path
            )
        )

    defaults = _read_yaml(default_path)
    if 'defaults' in defaults:
        raise ValueError('default.yaml cannot inherit another config')
    _validate_override_keys(defaults, overrides)
    return _deep_merge(defaults, overrides)
