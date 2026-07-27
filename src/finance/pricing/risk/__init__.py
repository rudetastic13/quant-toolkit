"""Risk layers: universal bump-and-reprice and the Numba analytic adjoint."""
from finance.pricing.risk.sensitivities import Sensitivities, KeyRateLadder, bumped_curve

try:  # optional accelerator
    from finance.pricing.risk.adjoint import NumbaRisk
except ImportError:  # pragma: no cover - minimal installation without numba
    NumbaRisk = None

__all__ = ["Sensitivities", "KeyRateLadder", "bumped_curve", "NumbaRisk"]
