"""Pricing layer: market context bundling, compiled kernels, and product pricers."""
from finance.pricing.types import Backend, ProductKind, RateKind
from finance.pricing.results import PricingResult, CashflowReport
from finance.pricing.pricers import SwapPricer, PricingProgram

# NOTE: ``finance.pricing.risk`` (Sensitivities) is intentionally NOT re-exported here.
# It imports ``finance.markets.context``, which imports ``finance.pricing.conventions`` —
# so re-exporting risk from this package __init__ would create an import cycle. Import it
# directly:  from finance.pricing.risk import Sensitivities
# (Future cleanup: move ConventionSet/ConventionRegistry under finance.markets so the
#  markets layer no longer depends on finance.pricing at all.)

__all__ = [
    "Backend",
    "ProductKind",
    "RateKind",
    "PricingResult",
    "CashflowReport",
    "SwapPricer",
    "PricingProgram",
]
