"""YieldCurveGroup — one currency's curves by key, plus the FX rate that translates them.

A group is a ``Mapping[str, YieldCurve]``.  Every name is a curve key in the group's currency
(``USD.SOFR.1D`` projects, ``USD.STDCSA`` discounts) and a curve's role is read off the key
grammar of the names it is registered under — nothing is declared.  ``add`` registers a curve
under its own key and, optionally, under the funding names it discounts under as well::

    usd = YieldCurveGroup("USD", default_funding="STDCSA")
    usd.add(sofr, funding_names=["USD.STDCSA"])   # projects as USD.SOFR.1D, discounts as USD.STDCSA
    usd.add(cme)                                  # key USD.CMECSA: discounts only
    usd.add(ftp)                                  # key USD.XXX-FTP.1D: projects only

Each name holds its own :class:`YieldCurve` view over the shared ``zero_curve``: a funding name
carries no index conventions and no fixings, so an alias never behaves like an index.
:class:`~finance.markets.context.MarketContext` holds one group per currency, validates origins
and ownership, and resolves ``market["USD.SOFR.1D"]`` through the group.  ``fx_rate`` is the
value of one unit of the group's currency in the market's base currency; it only translates a
PV and is never a pricing parameter.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType

from finance.conventions import OVERNIGHT, CurveKey
from finance.dates import Term
from finance.markets.curves import YieldCurve


class YieldCurveGroup(Mapping[str, YieldCurve]):
    """One currency's curves, by key.

    Parameters
    ----------
    currency : str
        ISO code; every key registered must be in it.
    fx_rate : float, optional
        Value of one unit of ``currency`` in the market's base currency (USD per GBP for a GBP
        group under a USD base).  Default 1.0, which is right for the base group only.
    default_funding : str or None, optional
        The funding slot ``discount()`` resolves when none is named, e.g. ``"STDCSA"``.
    """

    __slots__ = ("_currency", "_fx_rate", "_default_funding", "_views", "_frozen")

    def __init__(
        self,
        currency: str,
        *,
        fx_rate: float = 1.0,
        default_funding: str | None = None,
    ) -> None:
        self._currency = str(currency).strip().upper()
        if not float(fx_rate) > 0.0:
            raise ValueError(f"fx_rate must be positive, got {fx_rate!r}")
        self._fx_rate = float(fx_rate)
        self._default_funding = None if default_funding is None else str(default_funding).strip().upper()
        self._views: dict[str, YieldCurve] | MappingProxyType = {}
        self._frozen = False

    # -- identity -------------------------------------------------------------------------

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def fx_rate(self) -> float:
        return self._fx_rate

    @property
    def default_funding(self) -> str | None:
        return self._default_funding

    @property
    def frozen(self) -> bool:
        return self._frozen

    def _name(self, key: CurveKey | str) -> str:
        parsed = CurveKey.parse(key)
        if parsed.currency != self._currency:
            raise ValueError(f"'{parsed}' is not a {self._currency} key")
        return str(parsed)

    # -- Mapping --------------------------------------------------------------------------

    def __getitem__(self, key: str) -> YieldCurve:
        view = self._views.get(key)
        if view is None:
            view = self._views.get(self._name(key))
            if view is None:
                raise KeyError(f"no {self._currency} curve registered under '{key}'")
        return view

    def __iter__(self) -> Iterator[str]:
        return iter(self._views)

    def __len__(self) -> int:
        return len(self._views)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, (str, CurveKey)):
            return False
        try:
            return self._name(key) in self._views
        except ValueError:
            return False

    # -- registration ---------------------------------------------------------------------

    def add(self, yield_curve: YieldCurve, *, funding_names: Iterable[CurveKey | str] = ()) -> None:
        """Register ``yield_curve`` under its own key and under each of ``funding_names``.

        Parameters
        ----------
        yield_curve : YieldCurve
            Stored as given under ``yield_curve.name`` (its conventions and fixings included).
        funding_names : iterable of str or CurveKey, optional
            Discount keys in this currency (``"USD.STDCSA"``) the same ``zero_curve`` discounts
            under; each gets a bare view with no conventions and no fixings.

        Raises
        ------
        TypeError
            If the group is frozen.
        ValueError
            If a key is in another currency, or a funding name is a projection key.
        """
        if self._frozen:
            raise TypeError("this YieldCurveGroup is frozen")
        if not isinstance(yield_curve, YieldCurve):
            raise TypeError(f"expected a YieldCurve, got {type(yield_curve).__name__}")
        own = self._name(yield_curve.key)
        aliases: dict[str, YieldCurve] = {}
        for funding in funding_names:
            key = CurveKey.parse(funding)
            name = self._name(key)
            if key.is_projection:
                raise ValueError(f"funding name '{name}' is a projection key; a discount name is 'CCY.NAME'")
            aliases[name] = YieldCurve(origin=yield_curve.origin, zero_curve=yield_curve.zero_curve, key=key)
        self._views[own] = yield_curve
        self._views.update(aliases)

    def projection(self, index: str, tenor: Term | str = OVERNIGHT) -> YieldCurve:
        """The curve projecting ``index`` at ``tenor``: ``projection("SOFR")`` is ``USD.SOFR.1D``."""
        return self[str(CurveKey.projection(self._currency, index, tenor))]

    def discount(self, funding: str | None = None) -> YieldCurve:
        """The curve discounting under the funding slot ``funding`` (``"STDCSA"``), or the default.

        Raises
        ------
        ValueError
            If ``funding`` is omitted and the group has no ``default_funding``.
        """
        slot = funding if funding is not None else self._default_funding
        if slot is None:
            raise ValueError(f"the {self._currency} group has no default funding; name one")
        return self[str(CurveKey.discount(self._currency, slot))]

    # -- copies ---------------------------------------------------------------------------

    def _freeze(self) -> None:
        if not self._frozen:
            self._views = MappingProxyType(dict(self._views))
            self._frozen = True

    def _rebound(self, views: Mapping[str, YieldCurve]) -> YieldCurveGroup:
        """A frozen copy with the same currency, FX rate and default, holding ``views``."""
        out = YieldCurveGroup(self._currency, fx_rate=self._fx_rate, default_funding=self._default_funding)
        out._views = MappingProxyType(dict(views))
        out._frozen = True
        return out

    def __repr__(self) -> str:
        state = "frozen" if self._frozen else "mutable"
        return f"YieldCurveGroup({self._currency!r}, fx_rate={self._fx_rate:g}, curves={list(self._views)!r}, {state})"
