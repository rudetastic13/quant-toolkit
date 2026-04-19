"""ConventionSet — immutable market convention descriptor for a (currency, index) pair."""
from __future__ import annotations

from dataclasses import dataclass

from finance.dates import Term
from finance.dates.enums import BDC, DayCountMethod, Frequency, Roll


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

    def override(self, **kwargs) -> ConventionSet:
        """Return a new ConventionSet with the given fields replaced."""
        fields = {
            "day_count_method": self.day_count_method,
            "payment_frequency": self.payment_frequency,
            "reset_frequency": self.reset_frequency,
            "business_day_convention": self.business_day_convention,
            "roll_convention": self.roll_convention,
            "calendar": self.calendar,
            "spot_lag": self.spot_lag,
            "fixed_day_count_method": self.fixed_day_count_method,
        }
        fields.update(kwargs)
        return ConventionSet(**fields)
