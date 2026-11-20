"""Numba pricing and analytic-adjoint backend.

Importing the package registers the option kernels.  Machine-code compilation remains
lazy and cached; constructing a :class:`NumbaProgram` and making its first call creates one
dtype/layout signature that is reused for every subsequent portfolio length.
"""
from __future__ import annotations

from finance.pricing.engines.numba.kernels import bachelier_greeks, black_greeks
from finance.pricing.engines.numba.program import NumbaAdjointResult, NumbaProgram

__all__ = ["NumbaProgram", "NumbaAdjointResult", "bachelier_greeks", "black_greeks"]
