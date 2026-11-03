"""JSON serialisation for dataclasses, with stable type names and per-type versions.

A class opts in by inheriting :class:`Serializable` (or ``CommonObject``) **and** registering a
stable name::

    @serializable("Swap", version=1)
    @dataclass(kw_only=True)
    class Swap(Priceable): ...

The document carries ``{"__type__": "Swap", "__version__": 1, ...fields}``.  The name is the
contract: the class and its module can move or be renamed and old documents still load, because
:meth:`Serializable.deserialize` resolves the name through a registry, never a module path.

Versions move only when an old document can no longer be read by the current fields (a field
renamed, removed or re-interpreted; adding a field with a default is not a bump).  Each bump
ships a migration that rewrites the previous version's document, and the loader applies them in
sequence until the document matches the class::

    @upgrade("Swap", from_version=1)
    def _swap_1_to_2(doc: dict) -> dict:
        doc["funding"] = doc.pop("funding_id")
        return doc

Nested objects carry their own name and version, so bumping a leg does not bump the swap that
holds it.  See ``common/object/README.md`` for the rules and the fixture discipline.

Loading never needs the class: :func:`loads` / :func:`load` read the name off the document and
build whatever it describes, mirroring the ``json`` module (:func:`dumps` / :func:`dump` write).
"""

from __future__ import annotations

import json
import types as builtin_types
from collections.abc import Callable
from dataclasses import fields as dc_fields
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import IO, Any, Union, get_args, get_origin, get_type_hints

from common.registry import Registry

_type_serializers: dict[type, Callable[[Any], Any]] = {}
_type_deserializers: dict[type, Callable[[Any], Any]] = {}

_types: Registry[type] = Registry("Serializable")
_upgrades: dict[tuple[str, int], Callable[[dict[str, Any]], dict[str, Any]]] = {}

TYPE_KEY = "__type__"
VERSION_KEY = "__version__"


def register_type(type_cls: type, serializer: Callable[[Any], Any], deserializer: Callable[[Any], Any]) -> None:
    """Register custom serialization/deserialization handlers for a value type.

    This allows domain value types (e.g. Date, Term) to plug into the serialization framework
    without the common layer depending on them.  Value types carry no ``__type__``; they are
    recovered from the field's type hint.
    """
    _type_serializers[type_cls] = serializer
    _type_deserializers[type_cls] = deserializer


def serializable(name: str, *, version: int = 1):
    """Class decorator: register a :class:`Serializable` subclass under a stable name.

    Parameters
    ----------
    name : str
        The name written as ``__type__``.  It is forever: never reuse it for another class, never
        rename it.  The class and module are free to move.
    version : int, optional
        The document version this class writes and reads natively.  Bump it only with a matching
        :func:`upgrade` from the previous version.

    Raises
    ------
    TypeError
        If the class is not a ``Serializable``.
    ValueError
        If ``name`` is already registered to a different class.
    """

    def decorator(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, Serializable)):
            raise TypeError(f"@serializable({name!r}) needs a Serializable subclass, got {cls!r}")
        existing = _types.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"serialisable name {name!r} is already registered to {existing.__qualname__}")
        cls.__serial_name__ = name
        cls.__serial_version__ = int(version)
        _types.register(name, cls, overwrite=True)
        return cls

    return decorator


def upgrade(name: str, *, from_version: int):
    """Function decorator: register the migration of ``name`` documents from ``from_version`` to the next.

    The function takes the previous version's document (a dict, already a copy) and returns the
    next version's.  The loader chains them, so a document at version 1 reaching a class at
    version 4 passes through ``1 -> 2 -> 3 -> 4``.
    """

    def decorator(fn: Callable[[dict[str, Any]], dict[str, Any]]):
        key = (name, int(from_version))
        if key in _upgrades:
            raise ValueError(f"an upgrade for {name!r} from version {from_version} is already registered")
        _upgrades[key] = fn
        return fn

    return decorator


def registered_type(name: str) -> type | None:
    """The class registered under ``name``, or ``None``."""
    return _types.get(name)


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
            # Dtype-aware envelope — preserves dtype, shape, and datetime64 unit on round-trip.
            if np.issubdtype(value.dtype, np.datetime64):
                data: Any = value.astype(str).tolist()
            else:
                data = value.tolist()
            return {"__ndarray__": True, "dtype": str(value.dtype), "shape": list(value.shape), "data": data}
        if isinstance(value, (np.integer, np.bool_)):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.datetime64):
            return str(value)
    except ImportError:
        pass
    return str(value)


def _rebuild_ndarray(env: dict[str, Any]) -> Any:
    """Reconstruct a numpy array from the dtype-aware envelope written by serialize_value."""
    import numpy as np

    return np.array(env["data"], dtype=env["dtype"]).reshape(env["shape"])


def deserialize_value(value: Any, type_hint: type | None) -> Any:
    if value is None:
        return None

    # ── Marker-driven detection (fires regardless of type hint) ──────────────
    if isinstance(value, dict):
        # dtype-aware ndarray envelope
        if value.get("__ndarray__") is True:
            return _rebuild_ndarray(value)
        # Polymorphic Serializable — reconstruct whichever registered class was stored
        type_key = value.get(TYPE_KEY)
        if type_key and _types.has(type_key):
            return _types.get(type_key).deserialize(value)

    # When there's no type hint and the value is a container, recurse so that
    # nested markers (arrays / Serializables) are still found.
    if type_hint is None:
        if isinstance(value, list):
            return [deserialize_value(v, None) for v in value]
        if isinstance(value, dict):
            return {k: deserialize_value(v, None) for k, v in value.items()}
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

    # Bare (unparameterized) dict / list — recurse marker-driven so that nested
    # __ndarray__ envelopes and __type__ markers are still resolved.
    if type_hint is dict and isinstance(value, dict):
        return {k: deserialize_value(v, None) for k, v in value.items()}
    if type_hint is list and isinstance(value, list):
        return [deserialize_value(v, None) for v in value]

    # Concrete types
    if isinstance(type_hint, type):
        # numpy ndarray (hint-driven path; marker-driven path above handles untyped containers)
        try:
            import numpy as np

            if issubclass(type_hint, np.ndarray) or origin is np.ndarray:
                if isinstance(value, dict) and value.get("__ndarray__") is True:
                    return _rebuild_ndarray(value)
                return np.array(value)
        except ImportError:
            pass
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

    return value


class Serializable:
    """Mixin providing JSON serialization and deserialization for dataclasses.

    Subclasses must be registered with :func:`serializable`; serialising an unregistered class is
    an error, so a stable name is never invented from a module path.
    """

    __serial_name__: str
    __serial_version__: int

    @classmethod
    def _serial_info(cls) -> tuple[str, int]:
        # Declared on this class, not inherited: a subclass must register its own name.
        if "__serial_name__" not in cls.__dict__:
            raise TypeError(
                f"{cls.__qualname__} is Serializable but not registered; "
                f'decorate it with @serializable("{cls.__name__}", version=1)'
            )
        return cls.__dict__["__serial_name__"], cls.__dict__["__serial_version__"]

    def serialize(self) -> dict[str, Any]:
        name, version = type(self)._serial_info()
        result: dict[str, Any] = {TYPE_KEY: name, VERSION_KEY: version}
        for f in dc_fields(self):
            if not f.init:
                continue
            result[f.name] = serialize_value(getattr(self, f.name))
        return result

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.serialize(), indent=indent)

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Serializable:
        """Build the object a document describes.

        The document's ``__type__`` selects the class (any registered class may be loaded through
        ``Serializable.deserialize``); a document without one is read as ``cls``.  A document older
        than the class is upgraded step by step through the registered :func:`upgrade` functions.

        Raises
        ------
        KeyError
            If ``__type__`` names an unregistered class, or an upgrade step is missing.
        ValueError
            If the document was written by a newer version than the class knows.
        """
        type_key = data.get(TYPE_KEY)
        if type_key is None:
            target_cls = cls
        else:
            target_cls = _types.get(type_key)
            if target_cls is None:
                raise KeyError(f"unknown serialisable type {type_key!r}; register it with @serializable")
        name, current = target_cls._serial_info()
        version = int(data.get(VERSION_KEY, 1))
        if version > current:
            raise ValueError(f"{name} document is version {version}; this code reads up to version {current}")
        while version < current:
            step = _upgrades.get((name, version))
            if step is None:
                raise KeyError(f"no upgrade registered for {name!r} from version {version} to {version + 1}")
            data = step(dict(data))
            version += 1

        try:
            hints = get_type_hints(target_cls)
        except Exception:
            hints = {}

        init_fields = {f.name: f for f in dc_fields(target_cls) if f.init}
        kwargs: dict[str, Any] = {}
        for field_name, fld in init_fields.items():
            if field_name not in data:
                continue
            hint = hints.get(field_name)
            # Fallback: resolve raw field.type if get_type_hints missed it
            if hint is None and isinstance(fld.type, type):
                hint = fld.type
            kwargs[field_name] = deserialize_value(data[field_name], hint)

        return target_cls(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> Serializable:
        return cls.deserialize(json.loads(json_str))


# ---------------------------------------------------------------------------
# Module-level front door, mirroring ``json``: the document names its own class.
# ---------------------------------------------------------------------------
def dumps(obj: Serializable, *, indent: int | None = 2) -> str:
    """The JSON text of ``obj``'s document."""
    return obj.to_json(indent=indent)


def dump(obj: Serializable, fp: str | PathLike[str] | IO[str], *, indent: int | None = 2) -> None:
    """Write ``obj``'s document to a path or an open text file."""
    text = dumps(obj, indent=indent) + "\n"
    if isinstance(fp, (str, PathLike)):
        Path(fp).write_text(text)
    else:
        fp.write(text)


def loads(text: str) -> Serializable:
    """Build the object a JSON document describes; its ``__type__`` picks the class."""
    return Serializable.deserialize(json.loads(text))


def load(fp: str | PathLike[str] | IO[str]) -> Serializable:
    """Build the object the JSON document at a path, or in an open text file, describes."""
    if isinstance(fp, (str, PathLike)):
        return loads(Path(fp).read_text())
    return loads(fp.read())
