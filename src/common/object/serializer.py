from __future__ import annotations

import json
import types as builtin_types
from collections.abc import Callable
from dataclasses import fields as dc_fields
from enum import Enum
from typing import Any, ClassVar, Union, get_args, get_origin, get_type_hints


_type_serializers: dict[type, Callable[[Any], Any]] = {}
_type_deserializers: dict[type, Callable[[Any], Any]] = {}


def register_type(type_cls: type, serializer: Callable[[Any], Any], deserializer: Callable[[Any], Any]) -> None:
    """Register custom serialization/deserialization handlers for a type.

    This allows domain types (e.g. Date, Term) to plug into the serialization
    framework without the common layer depending on them.
    """
    _type_serializers[type_cls] = serializer
    _type_deserializers[type_cls] = deserializer


def serialize_value(value: Any) -> Any:
    if value is None:
        return None
    # Enum check before primitives since IntEnum/StrEnum are int/str subclasses
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (str, int, float, bool)):
        return value
    for type_cls, fn in _type_serializers.items():
        if isinstance(value, type_cls):
            return fn(value)
    if isinstance(value, Serializable):
        return value.serialize()
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    if isinstance(value, set):
        return [serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if np.issubdtype(value.dtype, np.datetime64):
                return [str(v) for v in value]
            return value.tolist()
        if isinstance(value, (np.integer, np.bool_)):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.datetime64):
            return str(value)
    except ImportError:
        pass
    return str(value)


def deserialize_value(value: Any, type_hint: type | None) -> Any:
    if value is None:
        return None
    if type_hint is None:
        return value

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Union / Optional (X | None)
    if origin is Union or origin is builtin_types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if value is None:
            return None
        for t in non_none:
            try:
                return deserialize_value(value, t)
            except (TypeError, ValueError, KeyError):
                continue
        return value

    # list[X]
    if origin is list:
        elem_type = args[0] if args else None
        return [deserialize_value(v, elem_type) for v in value]

    # set[X]
    if origin is set:
        elem_type = args[0] if args else None
        return {deserialize_value(v, elem_type) for v in value}

    # dict[K, V]
    if origin is dict:
        key_type = args[0] if args else None
        val_type = args[1] if len(args) > 1 else None
        return {deserialize_value(k, key_type): deserialize_value(v, val_type) for k, v in value.items()}

    # tuple[X, ...]
    if origin is tuple:
        if args:
            return tuple(deserialize_value(v, args[min(i, len(args) - 1)]) for i, v in enumerate(value))
        return tuple(value)

    # Concrete types
    if isinstance(type_hint, type):
        # Custom deserializers
        for type_cls, fn in _type_deserializers.items():
            if issubclass(type_hint, type_cls):
                return fn(value)
        # Serializable subclass
        if issubclass(type_hint, Serializable) and isinstance(value, dict):
            return type_hint.deserialize(value)
        # Enum
        if issubclass(type_hint, Enum):
            return type_hint[value]
        # Primitives
        if type_hint in (str, int, float, bool):
            return type_hint(value)

    # numpy ndarray
    try:
        import numpy as np

        if (isinstance(type_hint, type) and issubclass(type_hint, np.ndarray)) or origin is np.ndarray:
            if isinstance(value, list) and value and isinstance(value[0], str):
                return np.array(value, dtype="datetime64[D]")
            return np.array(value)
    except ImportError:
        pass

    return value


class Serializable:
    """Mixin providing JSON serialization and deserialization for dataclasses."""

    _type_registry: ClassVar[dict[str, type[Serializable]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Serializable._type_registry[f"{cls.__module__}.{cls.__qualname__}"] = cls

    def serialize(self) -> dict[str, Any]:
        result: dict[str, Any] = {"__type__": f"{type(self).__module__}.{type(self).__qualname__}"}
        for f in dc_fields(self):
            if not f.init:
                continue
            result[f.name] = serialize_value(getattr(self, f.name))
        return result

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.serialize(), indent=indent)

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Serializable:
        type_key = data.get("__type__")
        target_cls = cls
        if type_key and type_key in Serializable._type_registry:
            target_cls = Serializable._type_registry[type_key]

        try:
            hints = get_type_hints(target_cls)
        except Exception:
            hints = {}

        init_fields = {f.name: f for f in dc_fields(target_cls) if f.init}
        kwargs: dict[str, Any] = {}
        for name, fld in init_fields.items():
            if name not in data:
                continue
            hint = hints.get(name)
            # Fallback: resolve raw field.type if get_type_hints missed it
            if hint is None and isinstance(fld.type, type):
                hint = fld.type
            kwargs[name] = deserialize_value(data[name], hint)

        return target_cls(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> Serializable:
        return cls.deserialize(json.loads(json_str))
