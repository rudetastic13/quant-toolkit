"""Risk layers: universal bump-and-reprice, the Numba analytic adjoint, and the RiskEngine
front door that dispatches between them per docs/engine_selection.md."""
from finance.pricing.risk.sensitivities import Sensitivities, KeyRateLadder
from finance.pricing.risk.pillar import PillarRisk, pillar_risk
from finance.pricing.risk.jacobian import template_jacobian, calibration_jacobian
from finance.pricing.risk.basis import TentBasis, LeastSquaresBasis, SchemeBasis
from finance.pricing.risk.template import (
    RiskTemplate, BoundRiskTemplate, QuoteRisk, TemplateError, TemplateLimits, TemplateDiagnostics,
)
from finance.pricing.risk.engine import RiskEngine, CurveRiskReport

try:  # optional accelerator
    from finance.pricing.risk.adjoint import NumbaRisk
except ImportError:  # pragma: no cover - minimal installation without numba
    NumbaRisk = None

__all__ = [
    "Sensitivities", "KeyRateLadder", "NumbaRisk", "RiskEngine", "CurveRiskReport",
    "PillarRisk", "pillar_risk", "template_jacobian", "calibration_jacobian",
    "RiskTemplate", "BoundRiskTemplate", "QuoteRisk", "TemplateError", "TemplateLimits", "TemplateDiagnostics",
    "TentBasis", "LeastSquaresBasis", "SchemeBasis",
]
