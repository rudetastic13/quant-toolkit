from __future__ import annotations

from typing import Protocol, runtime_checkable

from finance.instruments.interfaces.traits import (
    HasScheduleParams,
    HasStubDates,
    HasNotional,
    HasAmortization,
    HasFixedRate,
)


@runtime_checkable
class Loan(HasScheduleParams, HasStubDates, HasNotional, HasAmortization, HasFixedRate, Protocol):
    """Loan instrument: fixed or floating rate, with amortization.

    The coupon_type field determines whether the loan is fixed or floating.
    Floating-rate loans will additionally satisfy HasFloatingRate at runtime.
    """
    ...
