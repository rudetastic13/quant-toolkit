"""Pricing layer: market context bundling, compiled kernels, and product pricers."""
from finance.pricing.types import Backend, ProductKind, RateKind
from finance.pricing.results import PricingResult, CashflowReport

# NOTE: ``finance.pricing.risk`` (Sensitivities) and ``finance.pricing.calibration``
# (CurveCalibrator) are intentionally NOT re-exported here. They import
# ``finance.markets.context``, which imports ``finance.pricing.conventions`` — so
# re-exporting them from this package __init__ would create an import cycle. Import them
# directly:  from finance.pricing.risk import Sensitivities
#            from finance.pricing.calibration import CurveCalibrator
# (Future cleanup: move ConventionSet/ConventionRegistry under finance.markets so the
#  markets layer no longer depends on finance.pricing at all.)
#
# SwapPricer / PricingProgram are imported lazily (PEP 562) for the same reason: the
# resolution builders import ``finance.pricing.conventions``, which runs this __init__;
# eagerly importing the pricer here would pull ``finance.instruments.resolution`` back in
# while it is still initialising. They remain available as ``from finance.pricing import
# SwapPricer``.


def __getattr__(name: str):
    if name in ("SwapPricer", "PricingProgram"):
        from finance.pricing import pricers

        return getattr(pricers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Backend",
    "ProductKind",
    "RateKind",
    "PricingResult",
    "CashflowReport",
    "SwapPricer",
    "PricingProgram",
]
