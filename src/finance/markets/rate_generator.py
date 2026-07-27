"""Convention-aware rate generation over registered yield curves."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable

import numpy as np

from finance.dates import DayCountMethod, period_fractions

if TYPE_CHECKING:
    from finance.instruments.resolution import Swap
    from finance.markets.context import MarketContext

FloatArray = np.ndarray
DateArray = np.ndarray


def _check_windows(starts: DateArray, ends: DateArray) -> tuple[DateArray, DateArray]:
    starts = np.asarray(starts).astype("datetime64[D]")
    ends = np.asarray(ends).astype("datetime64[D]")
    if starts.ndim != 1 or starts.shape != ends.shape:
        raise ValueError("starts and ends must be equally sized 1-D date arrays.")
    if np.isnat(starts).any() or np.isnat(ends).any():
        raise ValueError("rate windows cannot contain NaT.")
    if np.any(ends < starts):
        raise ValueError("rate window ends cannot precede starts.")
    return starts, ends


def _discount_pairs(curve, starts: DateArray, ends: DateArray) -> tuple[FloatArray, FloatArray]:
    """Evaluate shared date endpoints once and scatter them back into paired windows."""
    count = starts.size
    if count == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty
    all_dates = np.concatenate([starts, ends])
    # Daily overnight grids are dense over a bounded range. Factorizing their integer
    # ordinals with a bucket table avoids the O(M log M) sort in np.unique for large books.
    ordinals = all_dates.view(np.int64)
    lower = int(ordinals.min())
    span = int(ordinals.max()) - lower + 1
    if span <= 4 * ordinals.size:
        offsets = ordinals - lower
        seen = np.zeros(span, dtype=np.bool_)
        seen[offsets] = True
        unique_offsets = np.flatnonzero(seen)
        dates = (unique_offsets + lower).view("datetime64[D]")
        codes = np.empty(span, dtype=np.intp)
        codes[unique_offsets] = np.arange(unique_offsets.size)
        inverse = codes[offsets]
    else:
        dates, inverse = np.unique(all_dates, return_inverse=True)
    dfs = curve.discount_factor(dates)[inverse]
    return dfs[:count], dfs[count:]


@dataclass(frozen=True)
class RateGenerator:
    """Generate market rates from yield curves, conventions, and realized fixings.

    ``ZeroCurve`` supplies only discount state. This class owns the financial meaning of
    ratios of those discount factors: simple index rates, continuous interval forwards,
    compounded/averaged observation rates, and par swap rates.
    """

    market: MarketContext

    def simple_rate(
        self,
        curve_name: str,
        starts: DateArray,
        ends: DateArray,
        day_count: DayCountMethod | None = None,
    ) -> FloatArray:
        """Simple index rate over each window, overlaying realized fixings first."""
        starts, ends = _check_windows(starts, ends)
        yield_curve = self.market.yield_curve(curve_name)
        basis = yield_curve.index.day_count_method if day_count is None else day_count
        tau = period_fractions(basis, starts, ends)
        rates = np.zeros_like(tau, dtype=np.float64)

        fixing_path = self.market.fixings.get(curve_name)
        as_of = np.datetime64(self.market.as_of_date.to_str(), "D")
        realized = starts < as_of if fixing_path is not None else np.zeros(starts.size, dtype=np.bool_)
        if realized.any():
            rates[realized] = fixing_path.get_value(starts[realized])

        projected = ~realized & (tau > 0.0)
        if projected.any():
            df_start, df_end = _discount_pairs(
                yield_curve.zero_curve,
                starts[projected],
                ends[projected],
            )
            rates[projected] = (df_start / df_end - 1.0) / tau[projected]
        return rates

    def continuous_forward_rate(
        self,
        curve_name: str,
        starts: DateArray,
        ends: DateArray,
    ) -> FloatArray:
        """Continuously-compounded interval forward on the curve's Act/365 clock."""
        starts, ends = _check_windows(starts, ends)
        days = (ends.astype(np.int64) - starts.astype(np.int64)).astype(np.float64)
        rates = np.zeros(days.size, dtype=np.float64)
        nonzero = days > 0.0
        if nonzero.any():
            curve = self.market.zero_curve(curve_name)
            log_start = curve.log_discount_factor(starts[nonzero])
            log_end = curve.log_discount_factor(ends[nonzero])
            rates[nonzero] = -(log_end - log_start) / (days[nonzero] / 365.0)
        return rates

    def compounded_rate(
        self,
        curve_name: str,
        starts: DateArray,
        ends: DateArray,
        weights: FloatArray,
        offsets: np.ndarray,
    ) -> FloatArray:
        """Geometrically compounded index rates reduced over an observation grid."""
        rates = self.simple_rate(curve_name, starts, ends)
        weights, offsets = self._check_observations(rates, weights, offsets)
        growth = np.expm1(np.add.reduceat(np.log1p(rates * weights), offsets))
        return growth / np.add.reduceat(weights, offsets)

    def averaged_rate(
        self,
        curve_name: str,
        starts: DateArray,
        ends: DateArray,
        weights: FloatArray,
        offsets: np.ndarray,
    ) -> FloatArray:
        """Accrual-weighted arithmetic index rates over an observation grid."""
        rates = self.simple_rate(curve_name, starts, ends)
        weights, offsets = self._check_observations(rates, weights, offsets)
        return np.add.reduceat(rates * weights, offsets) / np.add.reduceat(weights, offsets)

    @staticmethod
    def _check_observations(
        rates: FloatArray,
        weights: FloatArray,
        offsets: np.ndarray,
    ) -> tuple[FloatArray, np.ndarray]:
        weights = np.asarray(weights, dtype=np.float64)
        offsets = np.asarray(offsets, dtype=np.intp)
        if weights.shape != rates.shape:
            raise ValueError("observation weights and rates must share shape.")
        if offsets.ndim != 1 or offsets.size == 0 or offsets[0] != 0:
            raise ValueError("offsets must be a non-empty 1-D array beginning at zero.")
        if np.any(np.diff(offsets) <= 0) or offsets[-1] >= rates.size:
            raise ValueError("offsets must define non-empty observation segments.")
        return weights, offsets

    def par_rate_from_program(
        self,
        program,
        *,
        fixed_legs: Iterable[int],
        floating_legs: Iterable[int],
    ) -> float:
        """Par rate from a program whose fixed coupons are compiled at a unit rate."""
        leg_pv = program.reprice(self.market).leg_pv
        fixed_pv = float(sum(leg_pv[index] for index in fixed_legs))
        floating_pv = float(sum(leg_pv[index] for index in floating_legs))
        if abs(fixed_pv) < 1e-16:
            raise ZeroDivisionError("unit fixed-leg annuity is zero; par rate is undefined.")
        return -floating_pv / fixed_pv

    def par_swap_rate(self, swap: Swap) -> float:
        """Generate the par fixed rate for a resolved fixed-versus-floating swap."""
        from finance.instruments.enums import CouponType
        from finance.pricing.pricers import SwapPricer

        legs = []
        fixed: list[int] = []
        floating: list[int] = []
        for index, leg in enumerate(swap):
            if leg.coupon_type == CouponType.Fixed:
                legs.append(replace(leg, coupon_rate=1.0))
                fixed.append(index)
            else:
                legs.append(leg)
                floating.append(index)
        if not fixed or not floating:
            raise ValueError("par_swap_rate requires at least one fixed and one floating leg.")
        unit_swap = replace(swap, receive_leg=legs[0], pay_leg=legs[1])
        program = SwapPricer().compile([unit_swap])
        return self.par_rate_from_program(program, fixed_legs=fixed, floating_legs=floating)


__all__ = ["RateGenerator"]
