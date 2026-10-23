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

from finance.conventions import CurveKey
from finance.dates import BDC, Date, Term, add_business_days, add_term
from finance.dates.term import TermType

STDCSA = "STDCSA"
"""The default funding id: the market-standard (cleared / standard CSA) discount curve."""


def curve_name(currency: str, index_name: str) -> str:
    """The projection curve key an instrument in ``currency`` names, normalised: ``'USD.SOFR.1D'``.

    ``index_name`` is the full key (``CCY.INDEX.TENOR``); it must be a projection key in the
    instrument's currency.

    Raises
    ------
    ValueError
        If it is not a projection key, or names another currency.
    """
    key = CurveKey.parse(index_name)
    if key.is_discount:
        raise ValueError(f"index {index_name!r} is a discount key; a floating leg projects off 'CCY.INDEX.TENOR'")
    if key.currency != currency.strip().upper():
        raise ValueError(f"index {index_name!r} names currency {key.currency} but the instrument is in {currency}")
    return str(key)


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
