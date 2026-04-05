from __future__ import annotations

from typing import Protocol, runtime_checkable

from finance.dates import Date
from finance.instruments.enums import AmortizationType


@runtime_checkable
class HasNotional(Protocol):
    """Trait for instruments that carry a principal balance."""

    @property
    def currency(self) -> str: ...

    @property
    def notional(self) -> float: ...


@runtime_checkable
class HasAmortization(Protocol):
    """Trait for instruments where the notional amortizes over time."""

    @property
    def amortization_type(self) -> AmortizationType: ...

    @property
    def original_notional(self) -> float | None: ...

    @property
    def amortization_start(self) -> Date | None: ...

    @property
    def amortization_end(self) -> Date | None: ...
