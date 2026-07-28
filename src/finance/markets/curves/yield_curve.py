
"""Convention-aware market curve composed over a pure :class:`ZeroCurve`."""
from __future__ import annotations

from dataclasses import dataclass

from finance.conventions import ConventionRegistry, MarketConventions, RateIndex, default_registry
from finance.markets.fixings import HistoricalFixings
from finance.markets.curves._curve_impl.zero_curve import ZeroCurve


@dataclass(frozen=True)
class YieldCurve:
    """Bind mathematical discount state to the conventions of one market index.

    ``ZeroCurve`` remains unaware of currency, index calendars, fixing rules, or coupon
    conventions. ``YieldCurve`` is the market object registered in ``MarketContext`` and
    supplies those definitions to ``RateGenerator``.
    """

    zero_curve: ZeroCurve
    conventions: MarketConventions
    historical_fixings: HistoricalFixings | None = None

    @classmethod
    def from_registry(
        cls,
        zero_curve: ZeroCurve,
        *,
        currency: str,
        index_name: str,
        registry: ConventionRegistry = default_registry,
        historical_fixings: HistoricalFixings | None = None,
    ) -> YieldCurve:
        """Compose a zero curve with the registered ``(currency, index)`` definition."""
        return cls(
            zero_curve=zero_curve,
            conventions=registry.get(currency, index_name),
            historical_fixings=historical_fixings,
        )

    @property
    def index(self) -> RateIndex:
        return self.conventions.index

    @property
    def name(self) -> str:
        return f"{self.index.currency.upper()}.{self.index.name.upper()}"

    def with_zero_curve(self, zero_curve: ZeroCurve) -> YieldCurve:
        """Return the same market definition with replacement mathematical curve state."""
        return YieldCurve(
            zero_curve=zero_curve,
            conventions=self.conventions,
            historical_fixings=self.historical_fixings,
        )

    def with_historical_fixings(self, historical_fixings: HistoricalFixings | None) -> YieldCurve:
        """Return the same curve state/definition with replacement fixing history."""
        return YieldCurve(
            zero_curve=self.zero_curve,
            conventions=self.conventions,
            historical_fixings=historical_fixings,
        )

    def discount_factor(self, dates):
        return self.zero_curve.discount_factor(dates)

    def log_discount_factor(self, dates):
        return self.zero_curve.log_discount_factor(dates)

    def zero_rate(self, dates):
        return self.zero_curve.zero_rate(dates)

    def __repr__(self) -> str:
        return f"YieldCurve(name={self.name!r}, zero_curve={self.zero_curve!r})"


__all__ = ["YieldCurve"]
