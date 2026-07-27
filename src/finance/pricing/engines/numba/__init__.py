"""Numba pricing and analytic-adjoint backend.

Importing the package registers the narrow kernel ABI.  Machine-code compilation remains
lazy and cached; constructing a :class:`NumbaProgram` and making its first call creates one
dtype/layout signature that is reused for every subsequent portfolio length.
"""
from __future__ import annotations

from finance.pricing.engines.numba.kernels import (
    averaged,
    bachelier,
    bachelier_greeks,
    black,
    black_greeks,
    compounded,
    dcf,
)
from finance.pricing.engines.numba.program import NumbaAdjointResult, NumbaProgram

__all__ = [
    "NumbaProgram",
    "NumbaAdjointResult",
    "dcf",
    "compounded",
    "averaged",
    "bachelier",
    "black",
    "bachelier_greeks",
    "black_greeks",
]
