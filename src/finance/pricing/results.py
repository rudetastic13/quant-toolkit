"""Pricing results — plain, immutable value objects (no curve, no mutable cache).

Deliberately simple: PV per instrument, PV per leg, and an optional cashflow table.
Sensitivities (DV01 / key-rate) are a *separate* layer that bump-reprices the compiled
portfolio — they are not methods here and not cached on any instrument.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArray = np.ndarray


@dataclass
class CashflowReport:
    """Per-flow detail for inspection/reporting (all arrays aligned, length F)."""

    pay_dates: np.ndarray      # datetime64[D]
    leg: np.ndarray            # leg index per flow
    notional: FloatArray
    rate: FloatArray
    period_frac: FloatArray
    df: FloatArray
    sign: FloatArray
    flow_pv: FloatArray


@dataclass
class PricingResult:
    """PV per instrument + per leg, with optional cashflow detail."""

    instrument_pv: FloatArray          # (n_instruments,)
    leg_pv: FloatArray                 # (n_legs,)
    cashflows: CashflowReport | None = None

    @property
    def pv(self) -> float:
        """Convenience for the single-instrument case."""
        if self.instrument_pv.shape[0] != 1:
            raise ValueError(f"pv is ambiguous for {self.instrument_pv.shape[0]} instruments; index instrument_pv")
        return float(self.instrument_pv[0])


__all__ = ["CashflowReport", "PricingResult"]
