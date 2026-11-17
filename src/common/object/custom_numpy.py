"""Numpy values in a JSON document: arrays and scalars, written and rebuilt.

Arrays and the two time scalars travel in a marker envelope that carries the dtype, so they are
recovered without a type hint (inside untyped containers too)::

    {"__ndarray__": true, "dtype": "float64", "shape": [2], "data": [1.0, 2.0]}
    {"__datetime64__": true, "dtype": "datetime64[D]", "value": "2025-04-03"}
    {"__timedelta64__": true, "dtype": "timedelta64[D]", "value": 5}

``datetime64`` data is ISO text; ``timedelta64`` data is its int64 count (NaT included), because
numpy's text form (``"5 days"``) does not parse back.  Other scalars become plain JSON numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NDARRAY_KEY = "__ndarray__"
DATETIME64_KEY = "__datetime64__"
TIMEDELTA64_KEY = "__timedelta64__"


def is_numpy(value: Any) -> bool:
    """Whether ``value`` is a numpy array or scalar."""
    return isinstance(value, (np.ndarray, np.generic))


def serialize_numpy(value: np.ndarray | np.generic) -> Any:
    """The JSON form of a numpy array or scalar.

    Parameters
    ----------
    value : ndarray or numpy scalar

    Returns
    -------
    dict, int, float or str
        An envelope for arrays, ``datetime64`` and ``timedelta64``; a plain number for integer,
        bool and floating scalars; ``str(value)`` for anything else.
    """
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.datetime64):
            data = value.astype(str).tolist()
        elif np.issubdtype(value.dtype, np.timedelta64):
            data = value.view(np.int64).tolist()
        else:
            data = value.tolist()
        return {NDARRAY_KEY: True, "dtype": str(value.dtype), "shape": list(value.shape), "data": data}
    # timedelta64 subclasses np.signedinteger, so it must be tested before np.integer
    if isinstance(value, np.timedelta64):
        return {TIMEDELTA64_KEY: True, "dtype": str(value.dtype), "value": int(value.view(np.int64))}
    if isinstance(value, np.datetime64):
        return {DATETIME64_KEY: True, "dtype": str(value.dtype), "value": str(value)}
    if isinstance(value, (np.integer, np.bool_)):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def is_numpy_envelope(value: dict[str, Any]) -> bool:
    """Whether a document dict is one of the envelopes :func:`serialize_numpy` writes."""
    return value.get(NDARRAY_KEY) is True or value.get(DATETIME64_KEY) is True or value.get(TIMEDELTA64_KEY) is True


def deserialize_numpy(envelope: dict[str, Any]) -> np.ndarray | np.generic:
    """Rebuild the array or scalar an envelope describes, dtype (and time unit) preserved."""
    if envelope.get(NDARRAY_KEY) is True:
        return np.array(envelope["data"], dtype=envelope["dtype"]).reshape(envelope["shape"])
    return np.array(envelope["value"], dtype=envelope["dtype"])[()]


def is_ndarray_hint(type_hint: Any, origin: Any) -> bool:
    """Whether a field hint is ``np.ndarray`` or ``NDArray[...]``."""
    return origin is np.ndarray or (isinstance(type_hint, type) and issubclass(type_hint, np.ndarray))
