"""Curve calibration — build market curves from single-quote instruments and a solver.

Import directly (``from quant_toolkit.finance.pricing.calibration import CurveCalibrator``); this package
is *not* re-exported from ``quant_toolkit.finance.pricing``.
"""
from quant_toolkit.finance.pricing.calibration.solvers import (
    Bootstrapper,
    GlobalSolver,
    ResidualFn,
    Solver,
    SolverResult,
)
from quant_toolkit.finance.pricing.calibration.instruments import (
    CalibrationInstrument,
    DepositHelper,
    FraHelper,
    FuturesHelper,
    Quote,
    QuoteKind,
    SpreadHelper,
    SwapHelper,
    deposit_helper,
    fra_helper,
    futures_helper,
    spread_helper,
    swap_helper,
)
from quant_toolkit.finance.pricing.calibration.calibrator import (
    CalibrationResult,
    CurveCalibrator,
    CurveDefinition,
    LOCAL_INTERPOLATORS,
)

__all__ = [
    # solvers
    "ResidualFn",
    "SolverResult",
    "Solver",
    "GlobalSolver",
    "Bootstrapper",
    # instruments
    "QuoteKind",
    "Quote",
    "CalibrationInstrument",
    "DepositHelper",
    "FraHelper",
    "FuturesHelper",
    "SpreadHelper",
    "SwapHelper",
    "deposit_helper",
    "fra_helper",
    "futures_helper",
    "spread_helper",
    "swap_helper",
    # calibrator
    "CurveDefinition",
    "CalibrationResult",
    "CurveCalibrator",
    "LOCAL_INTERPOLATORS",
]
