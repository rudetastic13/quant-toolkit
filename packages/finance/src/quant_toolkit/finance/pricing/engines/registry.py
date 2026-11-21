"""The two registries behind the engines, in a leaf module so engines can register without import cycles.

``program_registry``
    ``Backend -> program class``.  A *program* is a whole-portfolio engine: built from the compiled
    ``KernelInputs`` (and the market, whose curve geometry it may prepare against), it reprices the
    book.  This is the seam for swapping engines to compare performance or to add a new one (Rust):
    register a class under a new ``Backend`` and ``PricingProgram`` dispatches to it.  Why programs
    and not per-kernel calls: the speed of the fast engines comes from fusing curve lookup, shaping
    and reduction into one pass, which no per-kernel seam can express.

``kernel_registry``
    ``(kernel_id, Backend) -> callable``.  Narrow, pure array kernels that pricers call at exactly
    their own granularity: today the Black and Bachelier option kernels (value and greeks), later
    lattices for American exercise.

Program protocol
----------------
``__init__(inputs: KernelInputs, market)``, ``compatible(market) -> bool`` (can the prepared state serve
this market, or must it be rebuilt), and ``reprice(market) -> KernelResult``.  ``value_and_grad(market)``
is optional: a program without it (numpy, a primal-only Rust engine) is risked by bump-and-reprice, or by
the Numba adjoint, since every engine implements the same model.
"""
from __future__ import annotations

from quant_toolkit.common.registry import Registry, create_factory

program_registry: Registry = Registry("PricingPrograms")
kernel_registry: Registry = Registry("PricingKernels")

# Convenience dispatcher: kernel((kernel_id, backend), *args, **kwargs)
kernel = create_factory(kernel_registry)


def has_kernel(kernel_id: str, backend) -> bool:
    """True if a kernel is registered for ``(kernel_id, backend)``."""
    return kernel_registry.has((kernel_id, backend))


def has_program(backend) -> bool:
    """True if a program class is registered for ``backend``."""
    return program_registry.has(backend)


def program_class(backend):
    """The program class registered for ``backend``.

    Raises
    ------
    NotImplementedError
        If no engine is registered (for Numba: numba is not installed).
    """
    cls = program_registry.get(backend)
    if cls is None:
        raise NotImplementedError(f"no pricing engine is registered for backend {backend.name}")
    return cls
