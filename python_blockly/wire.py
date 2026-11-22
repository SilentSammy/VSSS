"""Lossless JSON representation for approved API values (no object handles)."""
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

if __package__:
    from .api_registry import DERIVED_FIELDS
else:
    from api_registry import DERIVED_FIELDS


def encode(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if is_dataclass(value):
        result = {field.name: encode(getattr(value, field.name)) for field in fields(value)}
        result.update({name: encode(getattr(value, name)) for name in DERIVED_FIELDS if hasattr(value, name)})
        return {'$record': type(value).__name__, 'fields': result}
    if isinstance(value, Mapping):
        return {'$map': [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {'$tuple': [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    raise TypeError(f'API returned an unsupported value: {type(value).__name__}')
