"""MarketContext — registered yield curves, fixings, vols, and conventions.

The market resolves market objects. Financial projection and par-rate logic deliberately
lives in :class:`finance.markets.rate_generator.RateGenerator`, not on the market or the
underlying mathematical :class:`ZeroCurve`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from finance.conventions import ConventionRegistry, default_registry
from finance.dates import Date
from finance.markets.curves import CurveNamespace, YieldCurve, ZeroCurve
from finance.markets.paths.single_path import SinglePath


@dataclass
class MarketContext:
    """The market-data container handed to rate generators and pricers.

    ``curves`` contains convention-aware ``YieldCurve`` objects. The context performs
    lookup and immutable scenario rebinding only; it does not generate forward rates.
    """

    as_of_date: Date
    curves: CurveNamespace
    fixings: dict[str, SinglePath] = field(default_factory=dict)
    vols: object | None = None
    conventions: ConventionRegistry = default_registry

    def with_curve(self, curve: YieldCurve) -> MarketContext:
        """Return a new market with ``curve`` bound under its canonical name."""
        namespace = CurveNamespace()
        snapshot = self.curves.snapshot()
        for name, (bound, _version) in snapshot.items():
            namespace.bind(curve if name == curve.name else bound)
        if curve.name not in snapshot:
            namespace.bind(curve)
        return MarketContext(
            as_of_date=self.as_of_date,
            curves=namespace,
            fixings=self.fixings,
            vols=self.vols,
            conventions=self.conventions,
        )

    def yield_curve(self, name: str) -> YieldCurve:
        """Resolve a registered convention-aware yield curve."""
        return self.curves.resolve(name)

    def zero_curve(self, name: str) -> ZeroCurve:
        """Resolve the mathematical discount state of a registered yield curve."""
        return self.yield_curve(name).zero_curve

    def convention(self, currency: str, index_name: str):
        """Resolve the registered market conventions for ``(currency, index)``."""
        return self.conventions.get(currency, index_name)


__all__ = ["MarketContext"]
