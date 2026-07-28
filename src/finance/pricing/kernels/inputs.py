"""KernelInputs — the columnar (struct-of-arrays) compiled form the engine reprices.

Everything is per-flow columns, concatenated across every leg of every instrument in the
portfolio.  The unit is the *flow* (one accrual period), NOT the leg — which is what makes
arbitrary piecewise coupons (fixed -> float -> fixed step-ups, N switches) free: a switch
is just a change in the ``rate_kind``/``fixed_rate`` column, and the reprice groups flows
by ``rate_kind`` (a handful of vectorized kernel calls regardless of population size).

This is the *lowered* form: ``CouponEvent``/``CouponSchedule`` remains the authored,
symbolic source of truth on the instrument; the compiler lowers it into these columns
(one-way, source -> IR), so there is no second source of truth (and no ``RateSpec``).

Sentinels (branchless shaping): ``cap = +inf`` means uncapped, ``floor``/``index_floor =
-inf`` means none, ``proj_curve = -1`` means no projection curve (fixed legs).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FloatArray = np.ndarray
IntArray = np.ndarray
DateArray = np.ndarray

NO_CAP = np.inf
NO_FLOOR = -np.inf
NO_CURVE = -1


# ---------------------------------------------------------------------------
# Notional providers — notionals are not always static (amortization).
# ---------------------------------------------------------------------------
class NotionalProvider:
    """Produces the per-period notional array for a leg.

    Precomputable providers implement ``materialize(n_periods) -> (n_periods,)`` (one-shot,
    no rate dependence — the fast path).  Rate-dependent providers leave it unimplemented
    and are time-stepped by the pricer via ``step``; ``is_rate_dependent`` selects the path.
    """

    is_rate_dependent: bool = False

    def materialize(self, n_periods: int) -> FloatArray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class StaticNotional(NotionalProvider):
    """Constant notional across all periods (bullet)."""

    amount: float

    def materialize(self, n_periods: int) -> FloatArray:
        return np.full(int(n_periods), float(self.amount), dtype=np.float64)


@dataclass
class ScheduledNotional(NotionalProvider):
    """Precomputed per-period notional (straight-line, custom schedule, etc.)."""

    schedule: FloatArray

    def materialize(self, n_periods: int) -> FloatArray:
        sched = np.asarray(self.schedule, dtype=np.float64)
        if sched.shape[0] != n_periods:
            raise ValueError(f"notional schedule has {sched.shape[0]} periods, expected {n_periods}")
        return sched


@dataclass
class RateDependentNotional(NotionalProvider):
    """Notional whose paydown depends on realized rates (level-pay, prepayment).

    DESIGNED seam: no one-shot ``materialize``; the pricer time-steps it across periods
    (vectorized across the population) — the MSR/balance-sheet hook.
    """

    original_notional: float
    is_rate_dependent: bool = True

    def step(self, t: int, prev_notional: FloatArray, prev_rate: FloatArray) -> FloatArray:  # pragma: no cover
        raise NotImplementedError(
            "RateDependentNotional time-stepping is a designed seam; implement in the pricer."
        )


# ---------------------------------------------------------------------------
# Columnar compiled inputs.
# ---------------------------------------------------------------------------
@dataclass
class KernelInputs:
    """Portfolio-wide columns.  F = total flows, L = legs, P = compounded/averaged periods."""

    # -- per-flow (F) -------------------------------------------------------
    pay_dates: DateArray         # datetime64[D]
    period_frac: FloatArray
    sign: FloatArray             # +/- per flow (receive/pay)
    notional: FloatArray         # materialized (static/scheduled fast path)
    rate_kind: IntArray          # RateKind codes (int8)
    fixed_rate: FloatArray
    spread: FloatArray
    index_floor: FloatArray      # NO_FLOOR = none
    cap: FloatArray              # NO_CAP = none
    floor: FloatArray            # NO_FLOOR = none
    margin: IntArray             # MarginTreatment codes
    discount_curve: IntArray     # index into curve_names
    proj_curve: IntArray         # index into curve_names; NO_CURVE for fixed
    reset_starts: DateArray      # datetime64[D] — simple float window start (NaT if n/a)
    reset_ends: DateArray        # datetime64[D] — simple float window end

    # -- leg / instrument structure ----------------------------------------
    leg_offsets: IntArray        # (L,) start flow index per leg (reduceat boundaries)
    leg_instrument: IntArray     # (L,) instrument index per leg
    n_instruments: int

    # -- global observation grid (compounded/averaged flows), M fixings -----
    obs_starts: DateArray        # (M,) datetime64[D]
    obs_ends: DateArray          # (M,)
    obs_w: FloatArray            # (M,)
    obs_offsets: IntArray        # (P,) start index of each compounded/avg period in the obs arrays
    obs_flow: IntArray           # (P,) global flow index for each of those periods
    obs_proj_curve: IntArray     # (P,) projection curve index for each period

    # -- name table ---------------------------------------------------------
    curve_names: list[str] = field(default_factory=list)

    @property
    def n_flows(self) -> int:
        return int(self.pay_dates.shape[0])

    @property
    def n_legs(self) -> int:
        return int(self.leg_offsets.shape[0])

    @property
    def n_obs_periods(self) -> int:
        return int(self.obs_offsets.shape[0])


@dataclass
class KernelResult:
    """Engine output."""

    instrument_pv: FloatArray            # (n_instruments,)
    leg_pv: FloatArray | None = None     # (L,)
    flow_pv: FloatArray | None = None    # (F,)
    rate: FloatArray | None = None       # (F,) realized per-period rate (handy for reports)
    df: FloatArray | None = None         # (F,) payment discount factors; expired flows are zero


__all__ = [
    "NotionalProvider",
    "StaticNotional",
    "ScheduledNotional",
    "RateDependentNotional",
    "KernelInputs",
    "KernelResult",
    "NO_CAP",
    "NO_FLOOR",
    "NO_CURVE",
]
