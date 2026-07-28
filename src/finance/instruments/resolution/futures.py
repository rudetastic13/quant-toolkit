"""SOFR futures contracts and contract-month date resolution."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finance.dates import Date, DayCountMethod
from finance.instruments.enums import CouponType

_MONTH_CODES = {code: month for month, code in enumerate("FGHJKMNQUVXZ", start=1)}
_QUARTERLY = frozenset({3, 6, 9, 12})


def imm_date(year: int, month: int) -> Date:
    """Third Wednesday of a quarterly contract month."""
    if month not in _QUARTERLY:
        raise ValueError("IMM month must be March, June, September, or December")
    first = np.datetime64(f"{year:04d}-{month:02d}-01", "D")
    # numpy weekday: 1970-01-01 was Thursday; Monday=0 after the +3 shift.
    weekday = int((first.astype(np.int64) + 3) % 7)
    first_wednesday = first + np.timedelta64((2 - weekday) % 7, "D")
    return Date.from_numpy(first_wednesday + np.timedelta64(14, "D"))


def next_quarterly_imm(date: Date) -> Date:
    """Quarterly IMM immediately following ``date``."""
    year = date.year
    candidates = [(year, m) for m in (3, 6, 9, 12)] + [(year + 1, 3)]
    for y, m in candidates:
        candidate = imm_date(y, m)
        if candidate > date:
            return candidate
    raise AssertionError("unreachable")


def _contract_month(month: str | tuple[int, int]) -> tuple[int, int]:
    if isinstance(month, tuple):
        year, number = int(month[0]), int(month[1])
    else:
        label = month.strip().upper()
        if "-" in label:
            year_text, month_text = label.split("-", 1)
            year, number = int(year_text), int(month_text)
        else:
            if len(label) < 2 or label[0] not in _MONTH_CODES:
                raise ValueError("contract month must look like 'Z26', '2026-12', or (2026, 12)")
            number = _MONTH_CODES[label[0]]
            yy = int(label[1:])
            year = 2000 + yy if yy < 100 else yy
    if not 1 <= number <= 12:
        raise ValueError(f"invalid contract month {number}")
    return year, number


def _first_business_day(year: int, month: int) -> Date:
    first = np.datetime64(f"{year:04d}-{month:02d}-01", "D")
    return Date.from_numpy(np.busday_offset(first, 0, roll="forward"))


@dataclass(frozen=True, kw_only=True)
class SofrFuture:
    """Resolved SR1/SR3 future, independent of curves and market state."""

    contract: str
    ref_start: Date
    ref_end: Date
    coupon_type: CouponType
    day_count_method: DayCountMethod = DayCountMethod.Actual360
    currency: str = "USD"
    index_name: str = "SOFR"
    funding_id: str = "STDCSA"
    point_value: float = 25.0
    contracts: float = 1.0
    entry_price: float | None = None
    convexity: float = 0.0

    @property
    def rate_kind(self):
        """Pricing discriminator, imported lazily to keep contracts above the pricing layer."""
        from finance.pricing.types import RateKind

        if self.coupon_type == CouponType.GeometricAveraged:
            return RateKind.Compounded
        return RateKind.Averaged

    @classmethod
    def from_contract_month(
        cls,
        *,
        contract: str,
        month: str | tuple[int, int],
        as_of: Date,
        currency: str = "USD",
        rate_index: str = "SOFR",
        funding_id: str = "STDCSA",
        point_value: float | None = None,
        contracts: float = 1.0,
        entry_price: float | None = None,
        convexity: float = 0.0,
    ) -> "SofrFuture":
        code = contract.upper()
        year, number = _contract_month(month)
        if code == "SR3":
            if number not in _QUARTERLY:
                raise ValueError("SR3 trades quarterly IMM months H/M/U/Z")
            start = imm_date(year, number)
            end = next_quarterly_imm(start)
            coupon_type = CouponType.GeometricAveraged
            default_point_value = 25.0
        elif code == "SR1":
            start = _first_business_day(year, number)
            next_year, next_month = (year + 1, 1) if number == 12 else (year, number + 1)
            # The final fixing accrues through the first business day of the next month.
            end = _first_business_day(next_year, next_month)
            coupon_type = CouponType.ArithmeticAveraged
            default_point_value = 41.666666666666664
        else:
            raise ValueError("supported SOFR futures are 'SR1' and 'SR3'")
        if end <= as_of:
            raise ValueError("future reference period has already ended as of valuation date")
        return cls(
            contract=code,
            ref_start=start,
            ref_end=end,
            coupon_type=coupon_type,
            currency=currency,
            index_name=rate_index,
            funding_id=funding_id,
            point_value=default_point_value if point_value is None else float(point_value),
            contracts=float(contracts),
            entry_price=None if entry_price is None else float(entry_price),
            convexity=float(convexity),
        )


__all__ = ["SofrFuture", "imm_date", "next_quarterly_imm"]
