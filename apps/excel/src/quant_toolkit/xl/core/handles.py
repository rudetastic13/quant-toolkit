"""Handles for objects too big for a cell: ``"Swap;;;8f2a1c"``, ``"List;;;03be77"``.

A function stores its object here and returns the handle string; another function takes the
handle and gets the object back. We keep our own store instead of xlOil's object cache because
xlOil's references start with a non-printing marker character, which shows up as a stray glyph
in the cell.

Handles are content-addressed: the digest is taken over the inputs that built the object, so
recalculating a cell gives the same handle and does not leak a new object each time.

A date inside a list or dict is indistinguishable from a number once Excel hands it over, so
``Date;;;2026-06-01`` is a value tag rather than a stored object: ``resolve`` parses it back
into a ``Date``.
"""
from __future__ import annotations

import hashlib

import numpy as np

from quant_toolkit.finance.dates import Date

SEP = ";;;"
DATE_TAG = "Date"

# handle -> live object. Process-local; rebuilt by recalculation, never persisted.
_STORE: dict[str, object] = {}


def _digest(parts: tuple[object, ...]) -> str:
    """Short hash of the inputs that built an object.

    Parameters
    ----------
    parts : tuple of object
        Hashed through ``repr``, arrays through their nested lists.

    Returns
    -------
    str
        Twelve hex characters.
    """
    h = hashlib.blake2b(digest_size=6)
    for p in parts:
        h.update(repr(p.tolist() if isinstance(p, np.ndarray) else p).encode())
        h.update(b"\x00")
    return h.hexdigest()


def store(tag: str, obj: object, *key_parts: object) -> str:
    """Store an object and return its handle.

    Parameters
    ----------
    tag : str
        The kind of object, shown before the separator: ``"Swap"``, ``"List"``.
    obj : object
        The object to keep.
    *key_parts : object
        The inputs that identify the object. Identical parts give the same handle, so
        recalculating a cell replaces its object instead of adding another.

    Returns
    -------
    str
        ``<tag>;;;<digest>``.
    """
    handle = f"{tag}{SEP}{_digest(key_parts)}"
    _STORE[handle] = obj
    return handle


def load(handle: object, tag: str | None = None) -> object:
    """The object behind a handle.

    Parameters
    ----------
    handle : str
        A handle string from a cell.
    tag : str, optional
        The kind expected; checked against the handle's own tag.

    Returns
    -------
    object

    Raises
    ------
    ValueError
        If ``handle`` is not a handle string.
    TypeError
        If the handle's tag is not ``tag``.
    KeyError
        If nothing is stored under the handle (the cell that builds it has not recalculated
        in this session).
    """
    if not isinstance(handle, str) or SEP not in handle:
        raise ValueError(f"{handle!r} is not a handle: pass the cell that built the object")
    got = handle.split(SEP, 1)[0]
    if tag is not None and got != tag:
        raise TypeError(f"expected a {tag} handle, got a {got} handle ({handle!r})")
    try:
        return _STORE[handle]
    except KeyError:
        raise KeyError(f"handle {handle!r} not found: recalculate the cell that creates it") from None


def tag_date(date: Date) -> str:
    """A date as a value tag that survives being put in a list or dict.

    Parameters
    ----------
    date : Date

    Returns
    -------
    str
        ``Date;;;YYYY-MM-DD``.
    """
    return f"{DATE_TAG}{SEP}{date.to_str()}"


def resolve(value: object) -> object:
    """A cell value as the object it stands for.

    Parameters
    ----------
    value : object
        A cell value; numpy scalars are unwrapped first.

    Returns
    -------
    object
        The stored object for a handle, a ``Date`` for a date tag, otherwise ``value`` unchanged.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str) and SEP in value:
        tag, body = value.split(SEP, 1)
        return Date.from_str(body) if tag == DATE_TAG else load(value)
    return value


def flatten(values: object) -> list:
    """A scalar or an Excel range as a flat list of resolved values.

    Parameters
    ----------
    values : object or ndarray
        One cell value, or a 2-D range.

    Returns
    -------
    list
        Each cell through :func:`resolve`, row by row; a single-element list for a scalar.
    """
    if isinstance(values, np.ndarray):
        return [resolve(v) for v in values.ravel()]
    return [resolve(values)]
