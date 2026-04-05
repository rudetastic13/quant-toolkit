from __future__ import annotations

from typing import Protocol, runtime_checkable

from finance.dates import Date
from finance.instruments.interfaces.traits import HasNotional


@runtime_checkable
class Future(HasNotional, Protocol):
    """Exchange-traded interest rate future.

    Standardized contract. Price = 100 - implied rate.
    Daily margin settlement, no principal exchange.
    """

    @property
    def maturity(self) -> Date: ...

    @property
    def rate_index(self) -> str: ...

    @property
    def coupon_rate(self) -> float: ...
