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
from finance.dates.enums import FixingType
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
    reset_starts: np.ndarray | None  # datetime64[D], N — single-fixing projection window start
    reset_ends: np.ndarray | None    # datetime64[D], N — single-fixing projection window end
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


def _single_fixing_windows(
    reset_boundaries: np.ndarray,
    accrual_starts: np.ndarray,
    fixing_type: FixingType,
    bdc: BDC,
    calendar: str,
    reset_frequency: Frequency,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each accrual period to its governing single-fixing reset window.

    Windows are the consecutive ``[reset_boundaries[j-1], reset_boundaries[j]]`` spans,
    prepended with one synthetic pre-effective window (one reset term back, BDC-adjusted on
    the reset calendar) so an in-advance first period has something to read.

    - ``Arrears``: the window *containing* the accrual start — for aligned frequencies this
      is the accrual window itself; for a slower reset it is the enclosing super-period.
    - ``Advance``: the window *ending* on/before the accrual start — the previous span, the
      synthetic pre-effective window for the first period(s).

    A slower ``reset_frequency`` (e.g. pay Monthly / reset Annually) naturally maps several
    consecutive accrual periods to the same repeated window; downstream projection dedups
    repeated dates.
    """
    try:
        pre = _dt_utils.add_term(
            reset_boundaries[:1].copy(), -Term.from_frequency(reset_frequency), bdc, calendar
        )
    except KeyError:  # Frequency.Once has no Term equivalent — mirror the first span
        span = reset_boundaries[1] - reset_boundaries[0] if reset_boundaries.shape[0] > 1 else np.timedelta64(1, "D")
        pre = _dt_utils.adjust_date(reset_boundaries[:1] - span, bdc, calendar)
    win_starts = np.concatenate([pre, reset_boundaries[:-1]])
    win_ends = reset_boundaries
    k = np.searchsorted(reset_boundaries, accrual_starts, side="right")
    if fixing_type == FixingType.Advance:
        k = k - 1
    k = np.clip(k, 0, len(win_ends) - 1)
    return win_starts[k], win_ends[k]


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
    fixing_type: FixingType = FixingType.Arrears,
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
    4. Single-fixing reset windows → ``reset_starts``/``reset_ends``, keyed off
       ``reset_frequency`` vs ``frequency`` and ``fixing_type``:
       aligned (reset == pay, or unset) reads the accrual window (Arrears) or the previous
       one (Advance); a slower reset builds a reset-frequency grid and ``searchsorted``-maps
       each period to its governing (repeated) window.  A faster reset has no single fixing
       — it requires the observation grid (averaged/compounded coupons).
    5. ``period_fractions()`` → ``period_fracs``
    6. If ``build_observations`` (compounded/averaged legs): Tier-2 observation grid.
       Sub-periods come from ``reset_frequency`` when it is intra-period (daily today,
       monthly-in-quarterly, ...); ``Advance`` or a slower reset collapse to one governing
       fixing per period (the step-4 windows).
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

    # 4. Single-fixing reset windows, keyed off reset vs payment frequency + fixing type.
    #    Frequency is an IntEnum ordered slow > fast, so "< frequency" means a faster reset.
    reset_cal = observation_calendar or calendar
    reset_faster = reset_frequency is not None and reset_frequency < frequency
    reset_starts = reset_ends = None
    if reset_faster:
        if not build_observations:
            raise ValueError(
                "reset_frequency faster than payment_frequency means multiple fixings per "
                "period — use an averaged/compounded coupon (observation grid), not a "
                "single-fixing float"
            )
    else:
        if reset_frequency is None or reset_frequency == frequency:
            reset_boundaries = boundaries  # aligned: reuse the adjusted payment boundaries
            window_freq = frequency
        else:  # slower reset (super-period): its own grid at reset_frequency
            reset_boundaries = generate_schedule(
                start_date=effective,
                end_date=maturity,
                frequency=reset_frequency,
                first_regular_date=first_regular,
                last_regular_date=last_regular,
                roll_convention=roll,
                direction=direction,
            )
            reset_boundaries = adjust_schedule_boundaries(
                reset_boundaries, bdc, reset_cal, protected=protected, adjust_endpoints=adjust_endpoints
            )
            window_freq = reset_frequency
        reset_starts, reset_ends = _single_fixing_windows(
            reset_boundaries, accrual_starts, fixing_type, bdc, reset_cal, window_freq
        )

    # 5. Period fractions
    pf = period_fractions(day_count_method, accrual_starts, accrual_ends)

    # 6. Tier-2 observation grid (compounded/averaged legs)
    obs_read_starts = obs_read_ends = obs_weights = obs_offsets = None
    if build_observations:
        if reset_faster and fixing_type == FixingType.Advance:
            raise NotImplementedError(
                "averaged/compounded in-advance over intra-period fixings (observing the "
                "prior period's sub-windows) is a designed seam"
            )
        if not reset_faster:
            # Advance, aligned, or super-period: one governing fixing per period — the
            # step-4 windows.  The weight cancels in a single-fixing reduction.
            obs_read_starts = reset_starts.copy()
            obs_read_ends = reset_ends.copy()
            obs_weights = pf.copy()
            obs_offsets = np.arange(accrual_starts.shape[0], dtype=np.intp)
        else:
            # Arrears intra-period: sub-periods at reset_frequency (Daily = every business
            # day, the classic RFR grid), then lookback/lockout on those sub-periods.
            sub_boundaries = None
            if reset_frequency > Frequency.Daily:
                sub = generate_schedule(
                    start_date=effective,
                    end_date=maturity,
                    frequency=reset_frequency,
                    first_regular_date=first_regular,
                    last_regular_date=last_regular,
                    roll_convention=roll,
                    direction=direction,
                )
                sub_boundaries = adjust_schedule_boundaries(
                    sub, bdc, reset_cal, protected=protected, adjust_endpoints=adjust_endpoints
                )
            grid = build_observation_grid(
                accrual_starts=accrual_starts,
                accrual_ends=accrual_ends,
                calendar=reset_cal,
                day_count=day_count_method,
                lookback=rate_lookback,
                lockout=rate_lockout,
                lookback_style=lookback_style if lookback_style is not None else LookbackStyle.Lookback,
                sub_boundaries=sub_boundaries,
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
        reset_starts=reset_starts,
        reset_ends=reset_ends,
        notional_schedule=None,
        obs_read_starts=obs_read_starts,
        obs_read_ends=obs_read_ends,
        obs_weights=obs_weights,
        obs_offsets=obs_offsets,
    )
