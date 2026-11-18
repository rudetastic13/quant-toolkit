"""Observation grid: the Tier-2 fixing grid that feeds the compounded/averaged kernels.

Fills the slot the schedule reserved but never built.  The whole leg's daily fixings are
laid out as ONE continuous, sorted boundary set; per-period membership is recovered with
``np.searchsorted`` (no per-period Python loop).  Build order:

    1. boundaries = business days  UNION  accrual boundaries   (continuous, tiling)
    2. offsets    = searchsorted(accrual_ends, interest_starts) (period membership)
    3. fixing window = where the rate is pulled from, per the lookback style
    4. weights    = day-count on the interest sub-period (Lookback) or the fixing window
                    (ObservationShift)

Each row has two date windows, named as the ARRC conventions name them: the *interest* window
``[interest_start, interest_end)`` is the slice of the accrual period the interest is applied
for, and the *fixing* window is where the rate is pulled from.  They coincide until a lookback
or a lockout moves the fixing window.

Including the accrual boundaries in step 1 makes every sub-period fall inside exactly one
accrual period.

Lookback styles (they differ in *which window's rate* is observed and *how it is weighted*):

- ``Lookback`` (ISDA, default): the rate is the overnight rate observed at the date shifted
  back ``lookback`` business days, ``[dᵢ-L, next_bd(dᵢ-L)]``; the weight is the REAL
  sub-period ``dcf(dᵢ, dᵢ₊₁)``.  Does not telescope for L>0 (the lookback basis).
- ``ObservationShift``: both the rate window and the weight come from the shifted window
  ``[dᵢ-L, dᵢ₊₁-L]``; telescopes against the curve shifted by L.  (The leg cashflow still
  accrues over the real ``period_frac``.)

``lockout`` freezes the tail sub-periods' rate window at the cutoff, the ISDA Lockout Date:
``accrual_end`` minus ``lockout`` business days.  The cutoff fixing is therefore used ``lockout``
times in total (``lockout - 1`` days are frozen; ``lockout=1`` freezes nothing), and the rows are
never merged: each locked day keeps its own weight and its own compounding step.

Conventions follow the 2006 ISDA Definitions compounding provisions (OIS Compounding,
Compounding with Lookback / Observation Period Shift / Lockout).  A period whose first day is
not a business day carries one extra leading sub-period that reads the preceding business
day's fixing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from finance.dates import BDC, DayCountMethod, is_good_bd, add_business_days, period_fractions


class LookbackStyle(IntEnum):
    """How the lookback shift relates the fixing window to the weight."""

    Lookback = 0          # ISDA: rate at shifted date (overnight), weight = real sub-period
    ObservationShift = 1  # rate & weight both from the shifted window


@dataclass
class ObservationGrid:
    """Flat Tier-2 arrays for a single leg (M fixings, N accrual periods).

    fixing_starts/fixing_ends     : (M,) datetime64[D]  where the rate is pulled from (post lookback/lockout)
    weights                       : (M,) float64        accrual weight per fixing
    offsets                       : (N,) intp           start index of each accrual period
    interest_starts/interest_ends : (M,) datetime64[D]  the days the interest is applied for (pre-shift)
    """

    fixing_starts: np.ndarray
    fixing_ends: np.ndarray
    weights: np.ndarray
    offsets: np.ndarray
    interest_starts: np.ndarray
    interest_ends: np.ndarray

    @property
    def n_obs(self) -> int:
        return int(self.fixing_starts.shape[0])


def build_observation_grid(
    accrual_starts: np.ndarray,
    accrual_ends: np.ndarray,
    calendar: str,
    day_count: DayCountMethod = DayCountMethod.Actual360,
    lookback: int = 0,
    lockout: int = 0,
    lookback_style: LookbackStyle = LookbackStyle.Lookback,
    sub_boundaries: np.ndarray | None = None,
) -> ObservationGrid:
    """Build the in-arrears observation grid for a leg.

    ``sub_boundaries`` selects the intra-period fixing granularity: ``None`` (default) uses
    every business day — the classic daily RFR grid — while an explicit sorted date array
    (e.g. a monthly reset schedule inside quarterly accrual periods) uses those dates as the
    sub-period boundaries instead.  Accrual boundaries are always unioned in, so every
    sub-period falls inside exactly one accrual period either way.
    """
    effective = accrual_starts[0]
    maturity = accrual_ends[-1]

    # 1. continuous boundary set: sub-period boundaries UNION accrual boundaries.  All
    #    inputs are sorted, dense datetime64[D] over [effective, maturity], so mark a
    #    day-offset bool mask and read it back — sorted & unique with no union1d sort.
    all_days = np.arange(effective, maturity + np.timedelta64(1, "D"), dtype="datetime64[D]")
    day0 = all_days.view(np.int64)[0]
    if sub_boundaries is None:
        mark = is_good_bd(all_days, calendar)
    else:
        mark = np.zeros(all_days.shape[0], dtype=bool)
        inside = sub_boundaries[(sub_boundaries >= effective) & (sub_boundaries <= maturity)]
        mark[inside.view(np.int64) - day0] = True
    mark[accrual_starts.view(np.int64) - day0] = True
    mark[accrual_ends.view(np.int64) - day0] = True
    boundaries = all_days[mark]

    interest_starts = boundaries[:-1]
    interest_ends = boundaries[1:]

    # 2. period membership + reduceat offsets (no per-period loop)
    period_idx = np.searchsorted(accrual_ends, interest_starts, side="right")
    n_periods = accrual_ends.shape[0]
    offsets = np.searchsorted(period_idx, np.arange(n_periods), side="left").astype(np.intp)

    # 3 & 4. the fixing window and the weights, per style
    if lookback_style == LookbackStyle.ObservationShift:
        fixing_starts = add_business_days(interest_starts, -lookback, BDC.Preceding, calendar) if lookback else interest_starts.copy()
        fixing_ends = add_business_days(interest_ends, -lookback, BDC.Preceding, calendar) if lookback else interest_ends.copy()
        weights = period_fractions(day_count, fixing_starts, fixing_ends)  # shifted weights
    else:  # ISDA Lookback: overnight rate at the shifted date; real-period weight
        # Rolled Preceding even with no lookback: a period whose first day is not a business day
        # reads the preceding business day's fixing (ISDA: r + 1 business days before day 1).
        fixing_starts = add_business_days(interest_starts, -lookback, BDC.Preceding, calendar)
        fixing_ends = add_business_days(fixing_starts, 1, BDC.Following, calendar)
        weights = period_fractions(day_count, interest_starts, interest_ends)  # real weights

    # lockout: freeze the tail sub-periods' rate window at the cutoff (the ISDA Lockout Date).
    # Counted back from the last business day in the period, so the cutoff fixing is used
    # ``lockout`` times whether or not the period end is a business day.
    if lockout:
        last_bd = add_business_days(accrual_ends - np.timedelta64(1, "D"), 0, BDC.Preceding, calendar)
        cutoff = add_business_days(last_bd, -(lockout - 1), BDC.Preceding, calendar)
        locked = interest_starts > cutoff[period_idx]
        lock_start = cutoff[period_idx]
        if lookback:
            lock_start = add_business_days(lock_start, -lookback, BDC.Preceding, calendar)
        lock_end = add_business_days(lock_start, 1, BDC.Following, calendar)
        fixing_starts = np.where(locked, lock_start, fixing_starts)
        fixing_ends = np.where(locked, lock_end, fixing_ends)

    return ObservationGrid(
        fixing_starts=fixing_starts,
        fixing_ends=fixing_ends,
        weights=weights,
        offsets=offsets,
        interest_starts=interest_starts,
        interest_ends=interest_ends,
    )
