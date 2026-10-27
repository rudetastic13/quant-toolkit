"""The bundled index families, registered into the shared ConventionRegistry on package import.

Families are all the curve layer needs: a projection key (``USD.SOFR.1D``) resolves to its
family here, with no instrument layer loaded.  The leg and product conventions of these same
families are registered beside them by ``finance.instruments.conventions.definitions``.

Calendar note
-------------
Only the ``no_holidays`` calendar is registered today (see ``finance.dates.calendars``).  SOFR
really settles on SIFMA; until a SIFMA calendar is registered we use ``no_holidays`` (weekday
mask, no holiday set).  Swap the calendar fields once a SIFMA calendar exists.
"""

from __future__ import annotations

from finance.conventions.convention_registry import default_registry
from finance.conventions.keys import OVERNIGHT
from finance.conventions.rate_index import IndexFamily
from finance.dates import DayCountMethod, Term
from finance.dates.enums import FixingType
from finance.dates.term.term_type import TermType


def _usd_overnight_family(name: str) -> IndexFamily:
    """A USD overnight index: Act/360, published next business day, one tenor (``1D``)."""
    return IndexFamily(
        currency="USD",
        name=name,
        day_count_method=DayCountMethod.Actual360,
        fixing_calendar="no_holidays",
        publication_lag=Term(1, TermType.BusinessDays),
        tenors=(OVERNIGHT,),
    )


# USD.SOFR.1D — the standard USD overnight index.
# USD.FEDFUND.1D — the projection index of a SOFR-discounted basis trade: a Fed-Funds-vs-fixed
#   swap projects off it and discounts off the market's USD.STDCSA curve.
# USD.UST-OTR.1D — the Treasury curve as an overnight index so RateGenerator.spread reads the
#   SOFR/GV spread in SOFR's basis; USD.UST-OTR is its discount key (yield_curve_group_plan.md §5).
# USD.XXX-FTP.1D — funds-transfer-pricing index (placeholder name): bank debt as a credit spread
#   over USD.UST-OTR.1D.
for _name in ("SOFR", "FEDFUND", "UST-OTR", "XXX-FTP"):
    default_registry.register_family(_usd_overnight_family(_name), overwrite=True)

# USD.CME-TERM-SOFR.{1M,3M,6M,12M} — a forward-looking term family, Act/360.  The rate is set at
# the start of each period but is the term rate over it, so its projection window is the accrual
# period itself: FixingType.Arrears in this library's reset-window vocabulary.
default_registry.register_family(
    IndexFamily(
        currency="USD",
        name="CME-TERM-SOFR",
        day_count_method=DayCountMethod.Actual360,
        fixing_calendar="no_holidays",
        fixing_type=FixingType.Arrears,
        tenors=tuple(Term.from_str(t) for t in ("1M", "3M", "6M", "12M")),
    ),
    overwrite=True,
)
