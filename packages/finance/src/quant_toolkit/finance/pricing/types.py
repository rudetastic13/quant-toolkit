"""Shared enums and key constants for the pricing layer.

``Backend`` selects the implementation language/runtime.  Whole-portfolio engines register under it
(``program_registry``: ``Backend -> program class``); narrow kernels register under
``(kernel_id, Backend)`` (``kernel_registry``).  See ``quant_toolkit.finance.pricing.engines.registry``.
"""
from __future__ import annotations

from enum import IntEnum

from quant_toolkit.common.containers.enums import SupportedIntEnum


class Backend(SupportedIntEnum):
    """Numeric kernel backend: numpy (reference), numba (fused kernels and the analytic adjoint), jax (oracle).

    ``Jax`` is the autodiff backend: the same columnar ``KernelInputs`` repriced through a
    pure ``jax.numpy`` valuation, so risk comes from ``jax.grad``/``jacobian``/``hessian``
    instead of bump-and-reprice.  It is optional and lazily imported (the core path never
    imports jax); register its kernels only when the backend is actually requested.
    """

    Numpy = 1
    Numba = 2
    Jax = 4


class RateKind(IntEnum):
    """Per-flow rate-computation discriminator (the ``rate_kind`` column).

    The compiled form is columnar: every period carries its own ``RateKind``, so a leg
    can switch fixed<->float<->compounded arbitrarily across its life (step-ups, etc.)
    with no special machinery — the reprice groups flows by this code.
    """

    Fixed = 0
    Float = 1        # simple single-fixing (IBOR-style)
    Compounded = 2   # OIS daily compounding (SOFR)
    Averaged = 3     # arithmetic average of daily fixings


# -- narrow kernel ids (``kernel_registry`` keys are ``(kernel_id, Backend)``) ------
KERNEL_BACHELIER = "bachelier"   # normal-vol option model: value + greeks
KERNEL_BLACK = "black"           # lognormal-vol option model: value + greeks


__all__ = [
    "Backend",
    "KERNEL_BACHELIER",
    "KERNEL_BLACK",
]
