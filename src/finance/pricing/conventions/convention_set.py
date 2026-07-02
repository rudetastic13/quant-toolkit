"""ConventionSet — immutable market convention descriptor for a (currency, index) pair."""
from __future__ import annotations

from dataclasses import dataclass, replace

from finance.dates import Term
from finance.dates.enums import BDC, DayCountMethod, FixingType, Frequency, Roll


@dataclass(frozen=True)
class ConventionSet:
    """
    Full set of market conventions for a rate instrument leg.

    All fields are required; factories are responsible for populating from
    the ConventionRegistry before constructing instruments.
    """

    day_count_method: DayCountMethod
    payment_frequency: Frequency
    reset_frequency: Frequency
    business_day_convention: BDC
    roll_convention: Roll
    calendar: str
    spot_lag: Term
    fixed_day_count_method: DayCountMethod | None = None
    fixing_type: FixingType = FixingType.Arrears

    def override(self, **kwargs) -> ConventionSet:
        """Return a new ConventionSet with the given fields replaced."""
        return replace(self, **kwargs)
