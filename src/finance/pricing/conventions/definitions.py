"""Static market-convention definitions, registered into the shared ConventionRegistry.

Importing this module has the side effect of registering the bundled conventions
(via the ``@register_convention`` decorator).  ``conventions/__init__.py`` imports it
so the definitions are available as soon as the package is loaded.

Calendar note
-------------
Only the ``no_holidays`` calendar is registered today (see
``finance.dates.calendars``).  SOFR really settles on SIFMA; until a SIFMA calendar
is registered we use ``no_holidays`` (weekday mask, no holiday set).  Swap the
``calendar`` field once a SIFMA calendar exists.
"""
from __future__ import annotations

from finance.dates import Term, DayCountMethod, BDC, Roll, Frequency
from finance.dates.term import TermType
from finance.pricing.conventions.convention_set import ConventionSet
from finance.pricing.conventions.convention_registry import register_convention


@register_convention("USD", "SOFR", overwrite=True)
def _usd_sofr() -> ConventionSet:
    """Standard USD SOFR OIS conventions (compounded-in-arrears, annual pay)."""
    return ConventionSet(
        day_count_method=DayCountMethod.Actual360,
        payment_frequency=Frequency.Annually,
        reset_frequency=Frequency.Daily,
        business_day_convention=BDC.ModifiedFollowing,
        roll_convention=Roll.Empty,
        calendar="no_holidays",
        spot_lag=Term(2, TermType.BusinessDays),
        fixed_day_count_method=DayCountMethod.Actual360,
    )
