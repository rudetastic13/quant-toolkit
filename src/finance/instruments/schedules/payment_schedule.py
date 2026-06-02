"""PaymentSchedule: numpy array container built from generate_schedule()."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from finance.dates import (
    Date,
    Term,
    Frequency,
    DayCountMethod,
    BDC,
    Roll,
    Direction,
    np_date_based_utils_module,
    period_fractions,
)
from finance.dates.schedules import generate_schedule
from finance.instruments.schedules.observation import LookbackStyle, build_observation_grid

_dt_utils = np_date_based_utils_module()


def adjust_schedule_boundaries(
    boundaries: np.ndarray,
    bdc: BDC,
    calendar: str,
    *,
    protected: set | None = None,
    adjust_endpoints: bool = True,
) -> np.ndarray:
    """Roll schedule boundary dates to good business days under ``bdc``.

    Every boundary is BDC-adjusted.  When ``adjust_endpoints`` is False, any boundary that
    matches a user-supplied explicit date in ``protected`` (effective / maturity /
    first-regular / last-regular) is restored to its raw value, so a contractual odd-day
    endpoint can flow through unadjusted.  ``BDC.NoAdjustment`` is a no-op.

    Adjustment is idempotent (re-adjusting a good business day returns it unchanged), so
    downstream payment-date adjustment is safe to apply on top.
    """
    adjusted = _dt_utils.adjust_date(boundaries.copy(), bdc, calendar)
    if not adjust_endpoints and protected:
        prot = np.array(sorted(protected), dtype="datetime64[D]")
        keep = np.isin(boundaries, prot)
        adjusted[keep] = boundaries[keep]
    return adjusted


@dataclass
class PaymentSchedule:
    """Numpy array container for a single leg's accrual/payment grid.

    Built once from ``build_payment_schedule``.  The ``notional_schedule``
    slot is left ``None`` until an ``AmortSchedule`` fills it.
    """

    accrual_starts: np.ndarray       # datetime64[D], N periods
    accrual_ends: np.ndarray         # datetime64[D], N periods
    payment_dates: np.ndarray        # datetime64[D], N periods (BDC-adjusted + delay)
    period_fracs: np.ndarray         # float64, N periods
    reset_dates: np.ndarray | None   # datetime64[D] — for floating resets
    fixing_dates: np.ndarray | None  # datetime64[D] — in-arrears fixing dates
    notional_schedule: np.ndarray | None = field(default=None)  # float64, N periods

    # Tier-2 observation grid (compounded/averaged legs). M total fixings across all periods.
    obs_read_starts: np.ndarray | None = field(default=None)   # datetime64[D], M (rate-read window start)
    obs_read_ends: np.ndarray | None = field(default=None)     # datetime64[D], M (rate-read window end)
    obs_weights: np.ndarray | None = field(default=None)       # float64, M (accrual weights)
    obs_offsets: np.ndarray | None = field(default=None)       # intp, N (reduceat boundaries)

    @property
    def n_periods(self) -> int:
        return len(self.accrual_starts)

    @property
    def has_observation_grid(self) -> bool:
        return self.obs_offsets is not None


def build_payment_schedule(
    effective: Date,
    maturity: Date,
    frequency: Frequency,
    day_count_method: DayCountMethod,
    bdc: BDC,
    calendar: str,
    roll: Roll = Roll.Empty,
    direction: Direction = Direction.Forward,
    payment_delay: Term | None = None,
    first_regular: Date | None = None,
    last_regular: Date | None = None,
    reset_frequency: Frequency | None = None,
    adjust_endpoints: bool = True,
    build_observations: bool = False,
    observation_calendar: str | None = None,
    rate_lookback: int = 0,
    rate_lockout: int = 0,
    lookback_style: LookbackStyle | None = None,
) -> PaymentSchedule:
    """Build a PaymentSchedule from schedule parameters.

    Steps
    -----
    1. ``generate_schedule()`` → boundary dates
    2. Derive ``accrual_starts``, ``accrual_ends``
    3. BDC-adjust accrual ends, then apply ``payment_delay`` → ``payment_dates``
    4. If ``reset_frequency`` differs from ``frequency``: second schedule → ``reset_dates``
    5. ``period_fractions()`` → ``period_fracs``
    6. If ``build_observations`` (compounded/averaged legs): daily Tier-2 observation grid
       via ``build_observation_grid`` → ``obs_value_dates`` / ``obs_weights`` / ``obs_offsets``.
    """
    # 1. Generate boundary dates
    boundaries = generate_schedule(
        start_date=effective,
        end_date=maturity,
        frequency=frequency,
        first_regular_date=first_regular,
        last_regular_date=last_regular,
        roll_convention=roll,
        direction=direction,
    )

    # 2. Accrual grid — BDC-adjust the generated roll dates. Explicit endpoints
    #    (effective/maturity/first/last regular) flow raw only when adjust_endpoints=False.
    protected = {effective.to_numpy(), maturity.to_numpy()}
    if first_regular is not None:
        protected.add(first_regular.to_numpy())
    if last_regular is not None:
        protected.add(last_regular.to_numpy())
    boundaries = adjust_schedule_boundaries(
        boundaries, bdc, calendar, protected=protected, adjust_endpoints=adjust_endpoints
    )
    accrual_starts = boundaries[:-1]
    accrual_ends = boundaries[1:]

    # 3. Payment dates = BDC-adjusted accrual ends + optional delay (payments always adjust)
    payment_dates = _dt_utils.adjust_date(accrual_ends.copy(), bdc, calendar)
    if payment_delay is not None:
        payment_dates = _dt_utils.add_term(payment_dates, payment_delay, bdc, calendar)

    # 4. Reset dates (if floating leg with different reset frequency)
    reset_dates = None
    if reset_frequency is not None and reset_frequency != frequency:
        reset_boundaries = generate_schedule(
            start_date=effective,
            end_date=maturity,
            frequency=reset_frequency,
            first_regular_date=first_regular,
            last_regular_date=last_regular,
            roll_convention=roll,
            direction=direction,
        )
        reset_dates = reset_boundaries[:-1]

    # 5. Period fractions
    pf = period_fractions(day_count_method, accrual_starts, accrual_ends)

    # 6. Tier-2 observation grid (compounded/averaged legs)
    obs_read_starts = obs_read_ends = obs_weights = obs_offsets = None
    if build_observations:
        grid = build_observation_grid(
            accrual_starts=accrual_starts,
            accrual_ends=accrual_ends,
            calendar=observation_calendar or calendar,
            day_count=day_count_method,
            lookback=rate_lookback,
            lockout=rate_lockout,
            lookback_style=lookback_style if lookback_style is not None else LookbackStyle.Lookback,
        )
        obs_read_starts = grid.read_starts
        obs_read_ends = grid.read_ends
        obs_weights = grid.weights
        obs_offsets = grid.offsets

    return PaymentSchedule(
        accrual_starts=accrual_starts,
        accrual_ends=accrual_ends,
        payment_dates=payment_dates,
        period_fracs=pf,
        reset_dates=reset_dates,
        fixing_dates=None,
        notional_schedule=None,
        obs_read_starts=obs_read_starts,
        obs_read_ends=obs_read_ends,
        obs_weights=obs_weights,
        obs_offsets=obs_offsets,
    )
