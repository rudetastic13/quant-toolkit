"""Compatibility shim for the production Numba pricing program.

The prototype formerly lived here.  It has moved to
``finance.pricing.engines.numba.NumbaProgram`` so benchmarks and production now exercise the
same fused primal/adjoint implementation.
"""
from finance.pricing.engines.numba import NumbaAdjointResult, NumbaProgram


def prepare(program, market) -> NumbaProgram:
    return NumbaProgram.from_program(program, market)


NumbaInputs = NumbaProgram

__all__ = ["prepare", "NumbaInputs", "NumbaProgram", "NumbaAdjointResult"]
