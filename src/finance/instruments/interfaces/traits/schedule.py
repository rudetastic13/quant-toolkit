
from __future__ import annotations

from typing import Protocol, runtime_checkable

from finance.dates import Date, Term, Frequency, DayCountMethod, BDC, Roll, Direction


@runtime_checkable
class HasScheduleParams(Protocol):
    """Trait for anything that generates a periodic schedule."""

    @property
    def effective(self) -> Date: ...

    @property
    def maturity(self) -> Date: ...

    @property
    def payment_frequency(self) -> Frequency: ...

    @property
    def day_count_method(self) -> DayCountMethod: ...

    @property
    def business_day_convention(self) -> BDC: ...

    @property
    def roll_convention(self) -> Roll: ...

    @property
    def direction(self) -> Direction: ...

    @property
    def pay_calendar(self) -> str: ...

    @property
    def payment_delay(self) -> Term | None: ...


@runtime_checkable
class HasStubDates(Protocol):
    """Trait for instruments that may have front/back stubs."""

    @property
    def first_regular_accrual_date(self) -> Date | None: ...

    @property
    def last_regular_accrual_date(self) -> Date | None: ...
