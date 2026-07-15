"""Rate index definitions — the anchor conventions every product hangs off.

A ``RateIndex`` describes the index itself, independent of any product traded on it:
its accrual basis, where fixings come from, and how they apply.  Product conventions
(``SwapConventions``, ``DepositConventions``, ...) reference the index for anything
index-derived — most importantly the day count, which is the *projection* basis the
curve kernels use to turn discount factors into forward rates.  A float leg's accrual
day count defaults to the index's, so accrual and projection align by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from finance.dates import Term
from finance.dates.enums import DayCountMethod, FixingType


@dataclass(frozen=True, kw_only=True)
class RateIndex:
    """A rate index (SOFR, ESTR, FEDFUND, ...) and its intrinsic conventions."""

    currency: str
    name: str
    day_count_method: DayCountMethod
    fixing_calendar: str
    fixing_type: FixingType = FixingType.Arrears
    # informational until the kernels consume it: SOFR publishes the prior
    # business day's rate the next morning
    publication_lag: Term | None = None
    # None = overnight index; a Term (e.g. "3M") marks a term/IBOR-style index
    tenor: Term | None = None

    @property
    def is_overnight(self) -> bool:
        return self.tenor is None

    @property
    def label(self) -> str:
        """The label instruments carry in their ``rate_index`` field, e.g. 'USD SOFR'."""
        return f"{self.currency} {self.name}"


@dataclass
class FundingIndex:
    """Class representing a funding index/CSA discount basis, such as SOFR-OIS discounting."""
    currency: str
    name: str
    _config: dict = field(default_factory=dict, init=False)

    @property
    def config(self) -> dict:
        return self._config

    def __getitem__(self, key: str):
        return self._config[key]
