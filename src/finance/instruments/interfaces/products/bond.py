from __future__ import annotations

from typing import Protocol, runtime_checkable

from finance.instruments.interfaces.traits import (
    HasScheduleParams,
    HasStubDates,
    HasNotional,
    HasFixedRate,
)


@runtime_checkable
class Bond(HasScheduleParams, HasStubDates, HasNotional, HasFixedRate, Protocol):
    """Fixed-rate bullet instrument: Treasuries, FHLB advances, Long Term Debt.

    Principal returned at maturity (bullet).
    """
    ...
