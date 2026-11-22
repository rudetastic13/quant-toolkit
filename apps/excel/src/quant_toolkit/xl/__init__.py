"""quant_toolkit.xl — thin Excel adapter (xlOil) over quant_toolkit.

Layers, deliberately kept apart so Excel concerns never leak into the core lib:

- ``cache``        — content-addressed object handle store (curves/instruments/programs
                     don't fit in a cell; functions return an opaque handle string instead).
- ``marshalling``  — Excel value <-> domain type conversion (serials/datetimes -> Date,
                     ranges -> numpy arrays).
- ``api``          — pure logic functions (snake_case, no Excel dependency) that do the real work.
- ``functions``    — the camelCase ``q*`` worksheet functions xlOil registers; thin wrappers
                     over ``api`` that turn exceptions into readable cell strings. The only
                     module that imports ``xloil`` (Windows-only), so it is not imported here:
                     xlOil loads ``quant_toolkit.xl.functions`` directly.

The ``q*`` names are the public Excel surface: ``qBuildCurve``, ``qCompileProgram``,
``qPrice``, etc. Treat their signatures as a stable contract — saved workbooks bind to them.
"""
from __future__ import annotations

from quant_toolkit.xl import api, cache, marshalling

__all__ = ["api", "cache", "marshalling"]
