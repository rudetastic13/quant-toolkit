"""MarketContext — the single market-data container handed to pricers.

Composes the pieces that already exist (``CurveNamespace``, ``ConventionRegistry``,
``SinglePath`` fixings) plus the designed vol seam, behind one object with a small,
vectorized query API.  Pricers receive a ``MarketContext`` and never reach into the
namespaces directly, so swapping curves for a scenario is a single rebind.

This is the *impure* side of the pure/impure split: it does curve lookups and
turns them into plain numpy arrays the engine kernels consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finance.dates import Date, DayCountMethod, period_fractions
from finance.markets.curves import ZeroCurve
from finance.markets.paths.single_path import SinglePath
from finance.markets.curves import CurveNamespace
from finance.pricing.conventions import ConventionRegistry, default_registry

FloatArray = np.ndarray
DateArray = np.ndarray


@dataclass
class MarketContext:
    """Bundle of curves, fixings, vols and conventions as of a valuation date.

    Parameters
    ----------
    as_of_date : Date
        Valuation date. Observations strictly before this use realized ``fixings``
        when available; on/after it the curve projects.
    curves : CurveNamespace
        Versioned name -> ZeroCurve store (discount and projection curves).
    fixings : dict[str, SinglePath]
        Realized index history keyed by index/curve name (optional overlay).
    vols : VolNamespace | None
        Designed seam for option pricers; ``None`` for linear products.
    conventions : ConventionRegistry
        Defaults to the shared ``default_registry``.
    """

    as_of_date: Date
    curves: CurveNamespace
    fixings: dict[str, SinglePath] = field(default_factory=dict)
    vols: object | None = None
    conventions: ConventionRegistry = default_registry

    # -- discounting --------------------------------------------------------
    def discount(self, name: str) -> ZeroCurve:
        """Resolve the named discount/projection curve."""
        return self.curves.resolve(name)

    def discount_factor(self, name: str, dates: DateArray) -> FloatArray:
        """DF(t) for an array of datetime64[D] dates off the named curve."""
        return self.curves.resolve(name).discount_factor(dates)

    # -- projection ---------------------------------------------------------
    def forward_rate(self, name: str, starts: DateArray, ends: DateArray) -> FloatArray:
        """Continuously-compounded forward rates between paired dates (curve only)."""
        return self.curves.resolve(name).forward_rate(starts, ends)

    def project(
        self,
        name: str,
        starts: DateArray,
        ends: DateArray,
        day_count: DayCountMethod = DayCountMethod.Actual360,
    ) -> FloatArray:
        """Simple (money-market) index rates over each [start, end] observation.

        ``simple = (DF(start)/DF(end) - 1) / tau`` with ``tau`` the day-count
        fraction — this is the per-observation overnight rate whose compounded
        product telescopes back to the curve's DF ratio.  Observations strictly
        before ``as_of_date`` are overlaid from realized ``fixings`` when present.
        """
        curve = self.curves.resolve(name)
        df_s = curve.discount_factor(starts)
        df_e = curve.discount_factor(ends)
        tau = period_fractions(day_count, starts, ends)
        # guard zero-length windows (tau==0) — leave as 0 rate
        out = np.zeros_like(tau, dtype=np.float64)
        nz = tau > 0
        out[nz] = (df_s[nz] / df_e[nz] - 1.0) / tau[nz]

        fix = self.fixings.get(name)
        if fix is not None:
            as_of = np.datetime64(self.as_of_date.to_str(), "D")
            past = starts < as_of
            if past.any():
                out[past] = fix.get_value(starts[past])
        return out

    # -- convention helper --------------------------------------------------
    def convention(self, currency: str, index_name: str):
        """Resolve the ConventionSet for a (currency, index) pair."""
        return self.conventions.get(currency, index_name)


__all__ = ["MarketContext"]
