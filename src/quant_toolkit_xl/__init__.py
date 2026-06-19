"""quant_toolkit_xl — thin Excel adapter over quant_toolkit.

Three layers, deliberately kept apart so Excel concerns never leak into the core lib:

- ``cache``        — content-addressed object handle store (curves/instruments/programs
                     don't fit in a cell; UDFs return an opaque handle string instead).
- ``marshalling``  — Excel value <-> domain type conversion (serials/datetimes -> Date,
                     ranges -> numpy arrays).
- ``api``          — pure logic functions (snake_case, no xlwings) that do the real work.
- ``functions``    — the camelCase ``q*`` UDFs Excel sees; thin wrappers over ``api`` that
                     turn exceptions into readable cell strings.

The ``q*`` names are the public Excel surface: ``qBuildCurve``, ``qCompileProgram``,
``qPrice``, etc. Treat their signatures as a stable contract — saved workbooks bind to them.
"""
from __future__ import annotations

from quant_toolkit_xl import api, cache, marshalling
from quant_toolkit_xl.functions import (
    qBuildCurve,
    qCompileProgram,
    qDiscountFactor,
    qMarket,
    qPrice,
    qSwap,
    qZeroRate,
)

__all__ = [
    "api",
    "cache",
    "marshalling",
    "qBuildCurve",
    "qCompileProgram",
    "qDiscountFactor",
    "qMarket",
    "qPrice",
    "qSwap",
    "qZeroRate",
]
