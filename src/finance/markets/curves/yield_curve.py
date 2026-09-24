"""Convention-aware market curve: an origin date, a pure :class:`ZeroCurve`, and an index.

This is the only layer that sees dates.  ``ZeroCurve`` works on float day offsets; every
date-based query here converts through :func:`dates_to_x` against ``origin``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from common.math.interpolation import Interpolator, Linear
from finance.conventions import ConventionRegistry, MarketConventions, RateIndex, default_registry
from finance.dates import Term
from finance.markets.curves._curve_impl.zero_curve import ZeroCurve
from finance.markets.curves.types import CurveSpace, RateExtrapolator
from finance.markets.fixings import HistoricalFixings

DateArray = NDArray[np.datetime64]
FloatArray = NDArray[np.float64]


def dates_to_x(origin: np.datetime64, dates: DateArray) -> FloatArray:
    """Float day offsets of ``dates`` from ``origin`` (any datetime64 unit; cast to days)."""
    days = np.asarray(dates).astype("datetime64[D]", copy=False)
    return (days.view(np.int64) - origin.astype("datetime64[D]").astype(np.int64)).astype(np.float64)


@dataclass(frozen=True)
class YieldCurve:
    """Bind an origin date and mathematical discount state to the conventions of one index.

    ``ZeroCurve`` is unaware of dates, currency, calendars, fixing rules or coupon conventions.
    ``YieldCurve`` is the market object registered in ``MarketContext`` and supplies those
    definitions to ``RateGenerator``.
    """

    origin: np.datetime64
    zero_curve: ZeroCurve
    conventions: MarketConventions
    historical_fixings: HistoricalFixings | None = None

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        origin: np.datetime64,
        zero_curve: ZeroCurve,
        *,
        currency: str,
        index_name: str,
        registry: ConventionRegistry = default_registry,
        historical_fixings: HistoricalFixings | None = None,
    ) -> YieldCurve:
        """Compose an origin and zero curve with the registered ``(currency, index)`` definition."""
        return cls(
            origin=np.datetime64(origin, "D"),
            zero_curve=zero_curve,
            conventions=registry.get(currency, index_name),
            historical_fixings=historical_fixings,
        )

    @classmethod
    def build(
        cls,
        node_dates: DateArray,
        dfs: FloatArray,
        *,
        currency: str,
        index_name: str,
        space: CurveSpace = CurveSpace.LogDF,
        interpolator: Interpolator = Linear(),
        extrapolation: RateExtrapolator = RateExtrapolator.FlatForward,
        registry: ConventionRegistry = default_registry,
        historical_fixings: HistoricalFixings | None = None,
    ) -> YieldCurve:
        """Dates-in convenience: the first node is the origin; the rest become the ``ZeroCurve`` pillars."""
        node_dates = np.asarray(node_dates).astype("datetime64[D]")
        origin = node_dates[0]
        zero_curve = ZeroCurve(
            dates_to_x(origin, node_dates),
            dfs,
            space=space,
            interpolator=interpolator,
            extrapolation=extrapolation,
        )
        return cls.from_registry(
            origin,
            zero_curve,
            currency=currency,
            index_name=index_name,
            registry=registry,
            historical_fixings=historical_fixings,
        )

    def with_zero_curve(self, zero_curve: ZeroCurve) -> YieldCurve:
        """Same origin, definition and fixings with replacement mathematical curve state."""
        return YieldCurve(
            origin=self.origin,
            zero_curve=zero_curve,
            conventions=self.conventions,
            historical_fixings=self.historical_fixings,
        )

    def with_historical_fixings(self, historical_fixings: HistoricalFixings | None) -> YieldCurve:
        """Same curve state and definition with replacement fixing history."""
        return YieldCurve(
            origin=self.origin,
            zero_curve=self.zero_curve,
            conventions=self.conventions,
            historical_fixings=historical_fixings,
        )

    # -- identity -----------------------------------------------------------------------

    @property
    def index(self) -> RateIndex:
        return self.conventions.index

    @property
    def name(self) -> str:
        return f"{self.index.currency.upper()}.{self.index.name.upper()}"

    # -- dates <-> x ----------------------------------------------------------------------

    def to_x(self, dates: DateArray) -> FloatArray:
        return dates_to_x(self.origin, dates)

    @property
    def node_dates(self) -> DateArray:
        """Pillar dates, origin first."""
        return self.origin + self.zero_curve.x.astype(np.int64).astype("timedelta64[D]")

    @property
    def max_date(self) -> np.datetime64:
        return self.node_dates[-1]

    def node_index(self, at: Term | np.datetime64) -> int:
        """Index of the pillar at ``origin + at`` (a ``Term``) or at the date ``at``.

        Raises ``ValueError`` when no pillar falls on that date; use it to place a
        :class:`~common.math.interpolation.Mixed` switch on a node.
        """
        target = np.datetime64(self.origin + at if isinstance(at, Term) else at, "D")
        nodes = self.node_dates
        idx = int(np.searchsorted(nodes, target))
        if idx >= nodes.size or nodes[idx] != target:
            raise ValueError(f"no curve pillar on {target}; pillars are {nodes.tolist()}.")
        return idx

    # -- queries by date ------------------------------------------------------------------

    def discount_factor(self, dates: DateArray) -> FloatArray:
        return self.zero_curve.discount_factor(self.to_x(dates))

    def log_discount_factor(self, dates: DateArray) -> FloatArray:
        return self.zero_curve.log_discount_factor(self.to_x(dates))

    def zero_rate(self, dates: DateArray) -> FloatArray:
        """Act/365 continuously compounded zero rate from ``origin`` to each date."""
        return self.zero_curve.zero_rate(self.to_x(dates))

    def instantaneous_forward(self, dates: DateArray) -> FloatArray:
        return self.zero_curve.instantaneous_forward(self.to_x(dates))

    def __repr__(self) -> str:
        return f"YieldCurve(name={self.name!r}, origin={self.origin}, zero_curve={self.zero_curve!r})"


__all__ = ["YieldCurve", "dates_to_x"]
