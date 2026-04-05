from __future__ import annotations

from typing import Protocol, runtime_checkable

from finance.dates import Date, DayCountMethod
from finance.instruments.interfaces.traits import HasNotional


@runtime_checkable
class FRA(HasNotional, Protocol):
    """Forward Rate Agreement: single-period, off-balance-sheet.

    No schedule generation needed. Settlement based on the difference
    between the agreed rate and the reference index fixing.
    """

    @property
    def effective(self) -> Date: ...

    @property
    def maturity(self) -> Date: ...

    @property
    def day_count_method(self) -> DayCountMethod: ...

    @property
    def coupon_rate(self) -> float: ...

    @property
    def rate_index(self) -> str: ...
