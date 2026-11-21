"""Pricing engines: whole-portfolio programs behind ``program_registry``, narrow kernels behind ``kernel_registry``.

Every engine consumes the same columnar ``KernelInputs`` (see ``quant_toolkit.finance.pricing.kernels``):

- ``numpy``  — the reference ``NumpyProgram``; no optional dependency, no adjoint.
- ``numba``  — ``NumbaProgram``: curve geometry prepared once, one fused ``@njit`` pass, analytic adjoint.
- ``jax``    — ``JaxProgram``: autodiff validation oracle (a differentiable closure, not this protocol).

Usage::

    from quant_toolkit.finance.pricing.engines import program_class
    program = program_class(Backend.Numba)(inputs, market)
    result = program.reprice(market)

See :mod:`quant_toolkit.finance.pricing.engines.registry` for the contract and for why the registry is per program.
"""
from __future__ import annotations

from quant_toolkit.finance.pricing.engines.registry import (
    has_kernel,
    has_program,
    kernel,
    kernel_registry,
    program_class,
    program_registry,
)
from quant_toolkit.finance.pricing.types import Backend

# Import implementations for their registration side effects.
from quant_toolkit.finance.pricing.engines.numpy import option as _numpy_option  # noqa: E402,F401
from quant_toolkit.finance.pricing.engines.numpy.program import NumpyProgram  # noqa: E402

program_registry.register(Backend.Numpy, NumpyProgram, overwrite=True)

# Numba is an optional accelerator.  Register it when installed without making the base
# package depend on it; the actual machine-code compile is lazy on first use.
try:  # pragma: no cover - the no-numba branch is exercised only in minimal installations
    import numba as _numba_runtime  # noqa: E402,F401
except ImportError:
    pass
else:
    from quant_toolkit.finance.pricing.engines.numba import kernels as _numba_kernels  # noqa: E402,F401
    from quant_toolkit.finance.pricing.engines.numba.program import NumbaProgram  # noqa: E402

    program_registry.register(Backend.Numba, NumbaProgram, overwrite=True)

__all__ = ["program_registry", "kernel_registry", "kernel", "has_kernel", "has_program", "program_class"]
