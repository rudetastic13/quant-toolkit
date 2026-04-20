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

_dt_utils = np_date_based_utils_module()

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

    @property
    def n_periods(self) -> int:
        return len(self.accrual_starts)


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
) -> PaymentSchedule:
    """Build a PaymentSchedule from schedule parameters.

    Steps
    -----
    1. ``generate_schedule()`` → boundary dates
    2. Derive ``accrual_starts``, ``accrual_ends``
    3. BDC-adjust accrual ends, then apply ``payment_delay`` → ``payment_dates``
    4. If ``reset_frequency`` differs from ``frequency``: second schedule → ``reset_dates``
    5. ``period_fractions()`` → ``period_fracs``
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

    # 2. Accrual grid
    accrual_starts = boundaries[:-1]
    accrual_ends = boundaries[1:]

    # 3. Payment dates = BDC-adjusted accrual ends + optional delay
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

    return PaymentSchedule(
        accrual_starts=accrual_starts,
        accrual_ends=accrual_ends,
        payment_dates=payment_dates,
        period_fracs=pf,
        reset_dates=reset_dates,
        fixing_dates=None,
        notional_schedule=None,
    )
