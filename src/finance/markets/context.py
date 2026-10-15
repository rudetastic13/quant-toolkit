"""MarketContext — registered yield curves, vols, and conventions.

The market resolves market objects. Financial projection and par-rate logic deliberately
lives in :class:`finance.markets.rate_generator.RateGenerator`, not on the market or the
underlying mathematical :class:`DiscountCurve`.

Curves are registered under curve keys (:class:`finance.conventions.CurveKey`): projection
keys ``CCY.INDEX.TENOR`` and discount keys ``CCY.NAME``.  A market must register every key
its trades name — in particular the discount key a trade's ``funding_id`` resolves to
(``USD.STDCSA``); discounting at SOFR is the SOFR curve object bound under that key as well.
"""

from __future__ import annotations

from dataclasses import dataclass

from finance.conventions import ConventionRegistry, CurveKey, default_registry
from finance.dates import Date
from finance.markets.curves import CurveNamespace, DiscountCurve, YieldCurve


@dataclass
class MarketContext:
    """The market-data container handed to rate generators and pricers.

    ``curves`` contains convention-aware ``YieldCurve`` objects. The context performs
    lookup and immutable scenario rebinding only. Historical fixings travel with their
    associated yield curve; the context itself does not generate rates.
    """

    as_of_date: Date
    curves: CurveNamespace
    vols: object | None = None
    conventions: ConventionRegistry = default_registry

    def with_curve(self, curve: YieldCurve, name: str | None = None) -> MarketContext:
        """Return a new market with ``curve`` bound under ``name`` (default: its own key).

        Rebinding a curve under its own key says "this curve moved": every other key holding
        the replaced object follows it (``USD.STDCSA`` moves with ``USD.SOFR.ON`` when it held
        the same object), and a key whose ``YieldCurve`` merely shares the replaced discount
        curve (same ``zero_curve`` object under its own key) receives the new discount curve
        and keeps its key.  An explicit ``name`` says "this key now means that curve" and
        changes that key only.
        """
        target = curve.name if name is None else str(name)
        snapshot = self.curves.snapshot()
        replaced = snapshot[target][0] if name is None and target in snapshot else None
        namespace = CurveNamespace()
        for bound_name, (bound, _version) in snapshot.items():
            if bound_name == target or (replaced is not None and bound is replaced):
                new = curve
            elif replaced is not None and bound.zero_curve is replaced.zero_curve:
                new = bound.with_zero_curve(curve.zero_curve)
            else:
                new = bound
            namespace.bind(new, name=bound_name)
        if target not in snapshot:
            namespace.bind(curve, name=target)
        return MarketContext(
            as_of_date=self.as_of_date,
            curves=namespace,
            vols=self.vols,
            conventions=self.conventions,
        )

    def yield_curve(self, name: str) -> YieldCurve:
        """Resolve a registered convention-aware yield curve."""
        return self.curves.resolve(name)

    def zero_curve(self, name: str) -> DiscountCurve:
        """Resolve the mathematical discount state of a registered yield curve (any ``DiscountCurve``)."""
        return self.yield_curve(name).zero_curve

    def curve_owner(self, name: str) -> str:
        """The key that owns the curve object bound under ``name``.

        When one discount-curve object sits under several keys a projection key owns it
        (``USD.SOFR.ON`` over ``USD.STDCSA``); otherwise the first key bound.  Engines lay
        parameters out per owner, so two keys on one object share one parameter block.
        """
        names = self.curve_aliases(name)
        for candidate in names:
            if CurveKey.parse(candidate).is_projection:
                return candidate
        return names[0]

    def curve_aliases(self, name: str) -> tuple[str, ...]:
        """Every key whose curve is the same discount-curve object as ``name``'s, in binding order."""
        target = self.zero_curve(name)
        return tuple(key for key in self.curves if self.curves.resolve(key).zero_curve is target)

    def convention(self, currency: str, index_name: str):
        """Resolve the registered market conventions for ``(currency, index)``."""
        return self.conventions.get(currency, index_name)
