"""Convention resolution — the small amount of glue that lets a trader fill ~3 fields.

This is the deliberate inverse of the "25 stringly-typed constructor args" approach: the
instrument carries a currency + index, and the conventions (frequencies, day counts, BDC,
roll, calendar, spot lag) are looked up from the shared ``ConventionRegistry``.  No curve
lives here — resolution produces a *contract*, never anything tied to market data.
"""
from __future__ import annotations

from finance.dates import Date, Term, BDC, add_business_days, add_term
from finance.dates.term import TermType
from finance.conventions import ConventionRegistry, MarketConventions, default_registry


def curve_name(currency: str, index_name: str) -> str:
    """Canonical curve key, e.g. ('usd','sofr') -> 'USD.SOFR'."""
    return f"{currency.upper()}.{index_name.upper()}"


# Funding (discount) curve aliases.  A ``funding_id`` is the label a trade carries to say
# *which* curve it is collateralised/discounted on; ``STDCSA`` ("standard CSA") is the
# market-standard OIS discount curve, which resolves per-currency to that ccy's OIS index.
# A simple per-ccy map for now — graduates to a registry / on-curve alias later (the
# ``FundingIndex`` stub in finance.conventions is the eventual home).
STDCSA = "STDCSA"
_FUNDING_ALIASES: dict[str, dict[str, str]] = {
    STDCSA: {"USD": "SOFR", "EUR": "ESTR"},
}


def funding_curve_name(currency: str, funding_id: str = STDCSA) -> str:
    """Resolve a funding id to a discount curve name, e.g. ('USD','STDCSA') -> 'USD.SOFR'.

    An aliased id (``STDCSA``) maps per-currency to its OIS index; an unrecognised id is
    treated as a concrete index name and passed straight through to :func:`curve_name`
    (so ``funding_id='SOFR'`` or ``'FEDFUNDS'`` also works).
    """
    aliases = _FUNDING_ALIASES.get(funding_id.upper())
    if aliases is None:
        return curve_name(currency, funding_id)
    try:
        index = aliases[currency.upper()]
    except KeyError:
        raise KeyError(f"no '{funding_id}' funding alias for currency '{currency}'") from None
    return curve_name(currency, index)


def resolve_conventions(
    currency: str, index_name: str, registry: ConventionRegistry = default_registry
) -> MarketConventions:
    """Look up the MarketConventions bundle for a (currency, index) pair."""
    return registry.get(currency, index_name)


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


__all__ = ["curve_name", "funding_curve_name", "resolve_conventions", "roll_spot", "STDCSA"]
