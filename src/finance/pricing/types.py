"""Shared enums and key constants for the pricing layer.

The engine registry is keyed by ``(kernel_id, Backend)``.  ``Backend`` selects the
implementation language/runtime; the *same* kernel id can have a numpy, numba, or
rust implementation registered under the same id with a different backend — the
swap is a one-line key change at the call site.  This mirrors the scalar-vs-numpy
two-registry pattern in ``finance.dates.term.term_math_registry``.
"""
from __future__ import annotations

from enum import IntEnum

from common.containers.enums import SupportedIntEnum


class Backend(SupportedIntEnum):
    """Numeric kernel backend. numpy is implemented; numba/rust are reserved seams.

    ``Jax`` is the autodiff backend: the same columnar ``KernelInputs`` repriced through a
    pure ``jax.numpy`` valuation, so risk comes from ``jax.grad``/``jacobian``/``hessian``
    instead of bump-and-reprice.  It is optional and lazily imported (the core path never
    imports jax); register its kernels only when the backend is actually requested.
    """

    Numpy = 1
    Numba = 2
    Rust = 3
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


class ProductKind(SupportedIntEnum):
    """Discriminator used to dispatch to a product pricer."""

    Swap = 1
    Bond = 2
    Future = 3
    FRA = 4
    Loan = 5
    Swaption = 6


# -- kernel ids -------------------------------------------------------------
KERNEL_DCF = "dcf"               # discounted-cashflow PV reducer
KERNEL_BACHELIER = "bachelier"   # normal-vol option model (reserved, not implemented)
KERNEL_BLACK = "black"           # lognormal-vol option model (reserved)

# -- rate-kernel kinds (also used as kernel ids in the engine registry) -----
RATE_FIXED = "rate_fixed"
RATE_FLOAT = "rate_float"          # single fixing per period (simple)
RATE_COMPOUNDED = "rate_compounded"  # OIS-style daily compounding (SOFR)
RATE_AVERAGED = "rate_averaged"      # arithmetic average of daily fixings


__all__ = [
    "Backend",
    "ProductKind",
    "KERNEL_DCF",
    "KERNEL_BACHELIER",
    "KERNEL_BLACK",
    "RATE_FIXED",
    "RATE_FLOAT",
    "RATE_COMPOUNDED",
    "RATE_AVERAGED",
]
