"""Convention resolution — the small amount of glue that lets a trader fill ~3 fields.

This is the deliberate inverse of the "25 stringly-typed constructor args" approach: the
instrument carries a currency + index, and the conventions (frequencies, day counts, BDC,
roll, calendar, spot lag) are looked up from the shared ``ConventionRegistry``.  No curve
lives here — resolution produces a *contract*, never anything tied to market data.

Curve names follow the one grammar in :mod:`finance.conventions.keys`: a floating leg
projects off ``CCY.INDEX.TENOR`` (:func:`curve_name`), cashflows discount off ``CCY.NAME``
(:func:`funding_curve_name`).  There is no alias from a funding id to an index: a market that
discounts at SOFR registers the SOFR curve under ``USD.STDCSA`` itself.
"""

from __future__ import annotations

from finance.conventions import ConventionRegistry, CurveKey, MarketConventions, default_registry, split_index_name
from finance.dates import BDC, Date, Term, add_business_days, add_term
from finance.dates.term import TermType

STDCSA = "STDCSA"
"""The default funding id: the market-standard (cleared / standard CSA) discount curve."""


def curve_name(currency: str, index_name: str, tenor: Term | str | None = None) -> str:
    """Projection curve key, e.g. ``('usd', 'sofr') -> 'USD.SOFR.ON'``.

    Parameters
    ----------
    currency : str
    index_name : str
        Family name, optionally with its tenor (``"CME-TERM-SOFR.3M"``).  A full key whose
        currency matches is accepted too.
    tenor : Term, str or None
        Overrides a missing tenor; a bare family name defaults to overnight.

    Returns
    -------
    str
    """
    family, embedded = _strip_currency(currency, index_name)
    family, embedded = split_index_name(family, embedded)
    if tenor is not None and embedded is not None:
        raise ValueError(f"tenor given twice: in {index_name!r} and as {tenor!r}")
    chosen = tenor if tenor is not None else embedded
    return str(CurveKey.projection(currency, family, "ON" if chosen is None else chosen))


def _strip_currency(currency: str, index_name: str) -> tuple[str, None]:
    slots = str(index_name).strip().upper().split(".")
    if len(slots) == 3:
        if slots[0] != currency.upper():
            raise ValueError(f"index {index_name!r} names currency {slots[0]} but the instrument is in {currency}")
        return ".".join(slots[1:]), None
    return ".".join(slots), None


def funding_curve_name(currency: str, funding_id: str = STDCSA) -> str:
    """Discount curve key: literally ``CCY.FUNDING_ID``, e.g. ``('USD', 'STDCSA') -> 'USD.STDCSA'``.

    A funding id is one slot (``STDCSA``, ``UST``, ``USDSTDCSA``); nothing resets off a
    discount curve, so a tenor is rejected.
    """
    if "." in funding_id:
        raise ValueError(
            f"funding id {funding_id!r} must be one slot (a discount curve has no tenor); "
            "register the index curve under the discount key instead"
        )
    return str(CurveKey.discount(currency, funding_id))


def resolve_conventions(
    currency: str, index_name: str, registry: ConventionRegistry = default_registry
) -> MarketConventions:
    """Look up the MarketConventions bundle for the family ``index_name`` names."""
    family, _ = _strip_currency(currency, index_name)
    return registry.get(currency, family)


def roll_spot(as_of: Date, spot_lag: Term, calendar: str) -> Date:
    """Spot/effective date = ``as_of`` advanced by the spot lag to a good business day.

    Business-day lags go through the calendar-aware path; calendar-tenor lags add the term
    then adjust Following.  A zero lag is a no-op.
    """
    if spot_lag.term_length == 0:
        return as_of
    if spot_lag.term_type == TermType.BusinessDays:
        eff = add_business_days(as_of.to_numpy(), spot_lag.term_length, BDC.Following, calendar)
    else:
        eff = add_term(as_of.to_numpy(), spot_lag, BDC.Following, calendar)
    return Date.from_numpy(eff)
