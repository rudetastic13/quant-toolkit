"""Observation grid: the Tier-2 fixing grid that feeds the compounded/averaged kernels.

Fills the slot the schedule reserved but never built.  The whole leg's daily fixings are
laid out as ONE continuous, sorted boundary set; per-period membership is recovered with
``np.searchsorted`` (no per-period Python loop).  Build order:

    1. boundaries = business days  UNION  accrual boundaries   (continuous, tiling)
    2. offsets    = searchsorted(accrual_ends, sub_starts)      (period membership)
    3. weights    = day-count on the PRE-SHIFT accrual sub-periods (the weight basis)
    4. read window = the rate-observation window, per the lookback style

Including the accrual boundaries in step 1 makes every sub-period fall inside exactly one
accrual period.

Lookback styles (the weight basis is always the pre-shift accrual sub-period; the styles
differ in *which window's rate* is observed and *how it is weighted*):

- ``Lookback`` (ISDA, default): the rate is the overnight rate observed at the date shifted
  back ``lookback`` business days, ``[dᵢ-L, next_bd(dᵢ-L)]``; the weight is the REAL
  sub-period ``dcf(dᵢ, dᵢ₊₁)``.  Does not telescope for L>0 (the lookback basis).
- ``ObservationShift``: both the rate window and the weight come from the shifted window
  ``[dᵢ-L, dᵢ₊₁-L]``; telescopes against the curve shifted by L.  (The leg cashflow still
  accrues over the real ``period_frac``.)

``lockout`` freezes the tail sub-periods' rate window at the cutoff
(``accrual_end`` minus ``lockout`` business days); weights are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from finance.dates import BDC, DayCountMethod, is_good_bd, add_business_days, period_fractions


class LookbackStyle(IntEnum):
    """How the lookback shift relates the rate-observation window to the weight."""

    Lookback = 0          # ISDA: rate at shifted date (overnight), weight = real sub-period
    ObservationShift = 1  # rate & weight both from the shifted window


@dataclass
class ObservationGrid:
    """Flat Tier-2 arrays for a single leg (M fixings, N accrual periods).

    read_starts/read_ends : (M,) datetime64[D]  rate-observation window (post shift/lockout)
    weights               : (M,) float64        accrual weight per fixing
    offsets               : (N,) intp           start index of each accrual period
    sub_starts/sub_ends   : (M,) datetime64[D]  pre-shift accrual sub-periods (weight basis)
    """

    read_starts: np.ndarray
    read_ends: np.ndarray
    weights: np.ndarray
    offsets: np.ndarray
    sub_starts: np.ndarray
    sub_ends: np.ndarray

    @property
    def n_obs(self) -> int:
        return int(self.read_starts.shape[0])


def build_observation_grid(
    accrual_starts: np.ndarray,
    accrual_ends: np.ndarray,
    calendar: str,
    day_count: DayCountMethod = DayCountMethod.Actual360,
    lookback: int = 0,
    lockout: int = 0,
    lookback_style: LookbackStyle = LookbackStyle.Lookback,
) -> ObservationGrid:
    """Build the in-arrears daily observation grid for a leg."""
    effective = accrual_starts[0]
    maturity = accrual_ends[-1]

    # 1. continuous boundary set: business days UNION accrual boundaries.  All inputs are
    #    sorted, dense datetime64[D] over [effective, maturity], so mark a day-offset bool
    #    mask and read it back — sorted & unique with no union1d sort.
    all_days = np.arange(effective, maturity + np.timedelta64(1, "D"), dtype="datetime64[D]")
    mark = is_good_bd(all_days, calendar)
    day0 = all_days.view(np.int64)[0]
    mark[accrual_starts.view(np.int64) - day0] = True
    mark[accrual_ends.view(np.int64) - day0] = True
    boundaries = all_days[mark]

    sub_starts = boundaries[:-1]
    sub_ends = boundaries[1:]

    # 2. period membership + reduceat offsets (no per-period loop)
    period_idx = np.searchsorted(accrual_ends, sub_starts, side="right")
    n_periods = accrual_ends.shape[0]
    offsets = np.searchsorted(period_idx, np.arange(n_periods), side="left").astype(np.intp)

    # 3 & 4. weights (pre-shift basis) and the rate-observation window, per style
    if lookback_style == LookbackStyle.ObservationShift:
        read_starts = add_business_days(sub_starts, -lookback, BDC.Preceding, calendar) if lookback else sub_starts.copy()
        read_ends = add_business_days(sub_ends, -lookback, BDC.Preceding, calendar) if lookback else sub_ends.copy()
        weights = period_fractions(day_count, read_starts, read_ends)  # shifted weights
    else:  # ISDA Lookback: overnight rate at the shifted date; real-period weight
        read_starts = add_business_days(sub_starts, -lookback, BDC.Preceding, calendar) if lookback else sub_starts.copy()
        read_ends = add_business_days(read_starts, 1, BDC.Following, calendar)
        weights = period_fractions(day_count, sub_starts, sub_ends)  # real weights

    # lockout: freeze the tail sub-periods' rate window at the cutoff
    if lockout:
        cutoff = add_business_days(accrual_ends, -lockout, BDC.Preceding, calendar)
        locked = sub_starts > cutoff[period_idx]
        lock_start = cutoff[period_idx]
        if lookback:
            lock_start = add_business_days(lock_start, -lookback, BDC.Preceding, calendar)
        lock_end = add_business_days(lock_start, 1, BDC.Following, calendar)
        read_starts = np.where(locked, lock_start, read_starts)
        read_ends = np.where(locked, lock_end, read_ends)

    return ObservationGrid(
        read_starts=read_starts,
        read_ends=read_ends,
        weights=weights,
        offsets=offsets,
        sub_starts=sub_starts,
        sub_ends=sub_ends,
    )


__all__ = ["ObservationGrid", "build_observation_grid", "LookbackStyle"]
