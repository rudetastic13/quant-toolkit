"""Pricing results — plain, immutable value objects (no curve, no mutable cache).

Deliberately simple: PV per instrument, PV per leg, and an optional cashflow table.
Sensitivities (DV01 / key-rate) are a *separate* layer that bump-reprices the compiled
portfolio — they are not methods here and not cached on any instrument.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import numpy as np

FloatArray = np.ndarray


def read_measure(result: Any, fields: Mapping[str, str], name: str) -> Any:
    """The value of measure ``name`` on ``result``, or ``None`` when the product does not answer it.

    Parameters
    ----------
    result : object
        A product result.
    fields : mapping of str to str
        Measure name -> attribute path on ``result`` (dotted for nested, ``"greeks.vega"``).
    name : str

    Returns
    -------
    object or None
        ``None`` for a measure outside ``fields`` and for one that was not requested.

    Notes
    -----
    On-demand measures (risk ladders) are computed only when requested and live in
    ``result.extras``, which is read first.
    """
    extras = getattr(result, "extras", None)
    if extras and name in extras:
        return extras[name]
    path = fields.get(name)
    if path is None:
        return None
    value = result
    for part in path.split("."):
        value = getattr(value, part)
    return value


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
    extras: dict[str, Any] = field(default_factory=dict)   # on-demand measures, by request name

    fields: ClassVar[dict[str, str]] = {"pv": "instrument_pv", "leg_pvs": "leg_pv", "cashflows": "cashflows"}

    def get(self, measure: str):
        """Measure by request name; ``None`` if a leg product does not answer it or it was not requested."""
        return read_measure(self, self.fields, measure)

    @property
    def value(self) -> FloatArray:
        """PV per instrument, shape (n_instruments,): the number risk differentiates."""
        return self.instrument_pv

    @property
    def pv(self) -> float:
        """Convenience for the single-instrument case."""
        if self.instrument_pv.shape[0] != 1:
            raise ValueError(f"pv is ambiguous for {self.instrument_pv.shape[0]} instruments; index instrument_pv")
        return float(self.instrument_pv[0])


__all__ = ["CashflowReport", "PricingResult"]
