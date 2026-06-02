"""Pure numeric kernels and the registry that dispatches them by backend.

A *kernel* takes plain contiguous numpy arrays and returns numpy arrays — no curve
lookups, no instrument objects.  Because the boundary is just arrays, a numpy kernel
and a future numba/rust kernel are interchangeable behind one registry key
``(kernel_id, Backend)``.

Usage::

    from finance.pricing.engines import engine
    from finance.pricing.types import KERNEL_DCF, Backend
    pv = engine((KERNEL_DCF, Backend.Numpy), cash, df, sign, row_offsets)
"""
from __future__ import annotations

from common.registry import Registry, create_factory

# (kernel_id, Backend) -> callable(*arrays) -> arrays
engine_registry: Registry = Registry("PricingEngines")

# Convenience dispatcher: engine((kernel_id, backend), *args, **kwargs)
engine = create_factory(engine_registry)


def has_engine(kernel_id: str, backend) -> bool:
    """True if a kernel is registered for ``(kernel_id, backend)``."""
    return engine_registry.has((kernel_id, backend))


# Import implementations for their registration side effects.
from finance.pricing.engines.numpy import rates as _rates  # noqa: E402,F401
from finance.pricing.engines.numpy import dcf as _dcf      # noqa: E402,F401

__all__ = ["engine_registry", "engine", "has_engine"]
