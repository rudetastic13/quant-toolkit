"""Risk layers: universal bump-and-reprice, the Numba analytic adjoint, and the RiskEngine
front door that dispatches between them per docs/engine_selection.md."""
from finance.pricing.risk.sensitivities import Sensitivities, KeyRateLadder, bumped_curve
from finance.pricing.risk.engine import RiskEngine, CurveRiskReport

try:  # optional accelerator
    from finance.pricing.risk.adjoint import NumbaRisk
except ImportError:  # pragma: no cover - minimal installation without numba
    NumbaRisk = None

__all__ = [
    "Sensitivities", "KeyRateLadder", "bumped_curve", "NumbaRisk", "RiskEngine", "CurveRiskReport",
]
