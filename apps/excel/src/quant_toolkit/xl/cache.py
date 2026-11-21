"""Content-addressed object handle cache.

A curve, instrument, or pricing program is far too big to live in a cell, so a UDF stores
it here and returns an opaque *handle* string (e.g. ``"Curve::8f2a1c"``). Other UDFs accept
that handle and look the object back up.

Handles are **content-addressed**: the handle is a digest of the inputs that built the
object, so identical inputs always map to the same handle. This matters in Excel — every
recalculation re-runs the UDF, and a random-id scheme would leak a new object on every
keystroke. Content addressing makes recalculation idempotent and naturally dedups shared
objects (one curve referenced by many cells is stored once).
"""
from __future__ import annotations

import hashlib

import numpy as np

# handle -> live python object. Process-local; rebuilt by recalculation, never persisted.
_STORE: dict[str, object] = {}

_SEP = "::"


def _digest(parts: tuple[object, ...]) -> str:
    """Stable 6-byte digest over the inputs that produced an object."""
    h = hashlib.blake2b(digest_size=6)
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
            h.update(str(p.dtype).encode())
            h.update(str(p.shape).encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()


def store(kind: str, obj: object, *key_parts: object) -> str:
    """Cache ``obj`` under a content-addressed handle and return the handle string.

    ``kind`` is a short type tag (``"Curve"``, ``"Swap"``, ``"Program"``, ``"Market"``)
    used both as the handle prefix and for type-checking on load. ``key_parts`` are the
    raw inputs that define the object — identical parts yield the same handle.
    """
    handle = f"{kind}{_SEP}{_digest(key_parts)}"
    _STORE[handle] = obj
    return handle


def load(handle: object, kind: str | None = None) -> object:
    """Resolve a handle back to its object.

    Raises a clear, Excel-friendly error if the handle is malformed, unknown (typically a
    stale reference whose creating cell needs recalculating), or of the wrong kind.
    """
    if not isinstance(handle, str) or _SEP not in handle:
        raise ValueError(f"{handle!r} is not a handle — pass the cell that built the object")
    if kind is not None and not handle.startswith(f"{kind}{_SEP}"):
        got = handle.split(_SEP, 1)[0]
        raise TypeError(f"expected a {kind} handle but got a {got} handle ({handle!r})")
    obj = _STORE.get(handle)
    if obj is None:
        raise KeyError(f"handle {handle!r} not found — recalculate the cell that creates it")
    return obj


def clear() -> int:
    """Drop every cached object. Returns the number removed."""
    n = len(_STORE)
    _STORE.clear()
    return n
