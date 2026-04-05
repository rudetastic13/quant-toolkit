from __future__ import annotations

from typing import Protocol, runtime_checkable

from finance.dates import Term, Frequency
from finance.instruments.enums import CouponType


@runtime_checkable
class HasFixedRate(Protocol):
    """Trait for instruments with a fixed coupon rate."""

    @property
    def coupon_type(self) -> CouponType: ...

    @property
    def coupon_rate(self) -> float: ...


@runtime_checkable
class HasFloatingRate(Protocol):
    """Trait for instruments with a floating rate index."""

    @property
    def coupon_type(self) -> CouponType: ...

    @property
    def rate_index(self) -> str: ...

    @property
    def spread(self) -> float: ...

    @property
    def reset_frequency(self) -> Frequency: ...

    @property
    def rate_lookback(self) -> Term | None: ...

    @property
    def rate_lockout(self) -> Term | None: ...

    @property
    def rate_calendar(self) -> str | None: ...


@runtime_checkable
class HasRateBounds(Protocol):
    """Trait for instruments with caps/floors on rate."""

    @property
    def cap(self) -> float | None: ...

    @property
    def floor(self) -> float | None: ...

    @property
    def index_floor(self) -> float | None: ...
