from __future__ import annotations

from typing import Protocol, runtime_checkable

from finance.instruments.interfaces.traits import (
    HasScheduleParams,
    HasStubDates,
    HasNotional,
    HasFixedRate,
)


@runtime_checkable
class SwapLeg(HasScheduleParams, HasStubDates, HasNotional, HasFixedRate, Protocol):
    """Single leg of an interest rate swap.

    Structurally similar to a bond leg but without principal exchange.
    The coupon_type field determines whether this is a fixed or floating leg.
    """
    ...


@runtime_checkable
class Swap(Protocol):
    """Interest rate swap: composition of two legs.

    Typically one fixed leg and one floating leg.
    Each leg independently defines its own schedule, day count, and rate parameters.
    """

    @property
    def pay_leg(self) -> SwapLeg: ...

    @property
    def receive_leg(self) -> SwapLeg: ...
