"""Convention-aware market curve: an origin date, a pure :class:`DiscountCurve`, and a key.

This is the only layer that sees dates.  Discount curves (``ZeroCurve``, ``CompositeCurve``)
work on float day offsets; every date-based query here converts through :func:`dates_to_x`
against ``origin``.

A ``YieldCurve`` is registered under its :class:`~finance.conventions.CurveKey`.  A
projection key (``USD.SOFR.1D``) carries the index family's conventions so
``RateGenerator`` knows the projection day count and fixing rules; a discount key
(``USD.STDCSA``, ``USD.UST``) carries none — nothing resets off it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from common.math.interpolation import Interpolator, Linear
from finance.conventions import ConventionRegistry, CurveKey, IndexFamily, MarketConventions, default_registry
from finance.dates import Term
from finance.markets.curves._curve_impl.discount_curve import DiscountCurve
from finance.markets.curves._curve_impl.zero_curve import ZeroCurve
from finance.markets.curves.types import CurveSpace, RateExtrapolator
from finance.markets.fixings import HistoricalFixings

DateArray = NDArray[np.datetime64]
FloatArray = NDArray[np.float64]


def dates_to_x(origin: np.datetime64, dates: DateArray) -> FloatArray:
    """Float day offsets of ``dates`` from ``origin`` (any datetime64 unit; cast to days)."""
    days = np.asarray(dates).astype("datetime64[D]", copy=False)
    return (days.view(np.int64) - origin.astype("datetime64[D]").astype(np.int64)).astype(np.float64)


def _resolve_key(key: CurveKey | str, registry: ConventionRegistry) -> tuple[CurveKey, MarketConventions | None]:
    """Parse ``key`` and look its conventions up: a projection key's family must publish its tenor."""
    parsed = CurveKey.parse(key)
    if parsed.is_discount:
        return parsed, None
    conventions = registry.get(parsed)
    conventions.index(parsed.tenor)  # raises KeyError when the family does not publish the tenor
    return parsed, conventions


@dataclass(frozen=True)
class YieldCurve:
    """Bind an origin date and mathematical discount state to a curve key.

    A ``DiscountCurve`` is unaware of dates, currency, calendars, fixing rules or coupon
    conventions.  ``YieldCurve`` is the market object registered in ``MarketContext`` and
    supplies those definitions to ``RateGenerator``.  The field is still called ``zero_curve``
    although any ``DiscountCurve`` (a spread composite over another index's atom, say) can sit there.

    Parameters
    ----------
    origin : datetime64[D]
    zero_curve : DiscountCurve
    key : CurveKey or str
        ``'USD.SOFR.1D'`` (projection) or ``'USD.STDCSA'`` (discount).
    conventions : MarketConventions or None
        Required for a projection key (its family must publish the key's tenor); must be
        ``None`` for a discount key.
    historical_fixings : HistoricalFixings or None
    """

    origin: np.datetime64
    zero_curve: DiscountCurve
    key: CurveKey
    conventions: MarketConventions | None = None
    historical_fixings: HistoricalFixings | None = None

    def __post_init__(self) -> None:
        key = CurveKey.parse(self.key)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "origin", np.datetime64(self.origin, "D"))
        if key.is_projection:
            if self.conventions is None:
                raise ValueError(f"projection curve {key} needs the conventions of its index family")
            self.conventions.index(key.tenor)
        elif self.conventions is not None:
            raise ValueError(f"discount curve {key} carries no index conventions")

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        origin: np.datetime64,
        zero_curve: DiscountCurve,
        *,
        key: CurveKey | str,
        registry: ConventionRegistry = default_registry,
        historical_fixings: HistoricalFixings | None = None,
    ) -> YieldCurve:
        """Compose an origin and discount curve with a key, looking conventions up in ``registry``.

        ``key`` is ``'USD.SOFR.1D'``, ``'USD.CME-TERM-SOFR.3M'`` (projection) or ``'USD.STDCSA'``
        (discount).
        """
        parsed, conventions = _resolve_key(key, registry)
        return cls(
            origin=np.datetime64(origin, "D"),
            zero_curve=zero_curve,
            key=parsed,
            conventions=conventions,
            historical_fixings=historical_fixings,
        )

    @classmethod
    def build(
        cls,
        node_dates: DateArray,
        dfs: FloatArray,
        *,
        key: CurveKey | str,
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
            key=key,
            registry=registry,
            historical_fixings=historical_fixings,
        )

    def with_zero_curve(self, zero_curve: DiscountCurve) -> YieldCurve:
        """Same origin, key, conventions and fixings with replacement mathematical curve state."""
        return YieldCurve(
            origin=self.origin,
            zero_curve=zero_curve,
            key=self.key,
            conventions=self.conventions,
            historical_fixings=self.historical_fixings,
        )

    def with_historical_fixings(self, historical_fixings: HistoricalFixings | None) -> YieldCurve:
        """Same curve state and key with replacement fixing history."""
        return YieldCurve(
            origin=self.origin,
            zero_curve=self.zero_curve,
            key=self.key,
            conventions=self.conventions,
            historical_fixings=historical_fixings,
        )

    def with_key(self, key: CurveKey | str, *, registry: ConventionRegistry = default_registry) -> YieldCurve:
        """The same origin, curve state and fixings registered under another key."""
        parsed, conventions = _resolve_key(key, registry)
        return YieldCurve(
            origin=self.origin,
            zero_curve=self.zero_curve,
            key=parsed,
            conventions=conventions,
            historical_fixings=self.historical_fixings,
        )

    # -- identity -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The key as text: ``'USD.SOFR.1D'``, ``'USD.STDCSA'``."""
        return str(self.key)

    @property
    def family(self) -> IndexFamily | None:
        """The projected index family for a projection key; ``None`` for a discount key."""
        return None if self.conventions is None else self.conventions.family

    # -- dates <-> x ----------------------------------------------------------------------

    def to_x(self, dates: DateArray) -> FloatArray:
        return dates_to_x(self.origin, dates)

    @property
    def node_dates(self) -> DateArray:
        """Pillar dates, origin first (for a composite, the union of its components' pillars)."""
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
