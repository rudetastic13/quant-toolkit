"""Amortisation: what a leg has outstanding over each accrual period.

Two ways to say it, one truth the compiler reads (the per-period notional array):

- **Generated** (``StraightLine``, ``LevelPay``): computed here from the leg's amortisation
  window — ``amort_start`` (interest-only before it), ``amort_end`` (the balloon date: whatever
  is outstanding is repaid there, and it may precede maturity), ``amortization_term`` (the
  horizon the schedule is solved on; longer than the window leaves a balloon) and
  ``principal_frequency`` (the coupon frequency or a slower multiple of it).
- **Custom** (:class:`NotionalSchedule`): dated events saying "from this accrual period the
  outstanding is X" — known curtailments, a seasoned loan's actual history.

A level payment is re-solved only when the coupon rate changes (a fixed step-up here; an ARM
reset in general), from the balance then outstanding over the remaining horizon, and held until
the next change.  On a floating leg the rate is only known at reprice time, so the balance path is
a *kernel stage* (a per-leg recursion over the projected rates, between the rate and cash stages),
not a compile-time column; that stage is not built yet, so ``LevelPay`` on a floating leg raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common.object.serializer import serializable
from finance.dates import BDC, Date, Frequency
from finance.dates.utils import add_term
from finance.instruments.enums import AmortizationType
from finance.instruments.schedules.base_schedule import BaseEvent, BaseSchedule
from finance.instruments.schedules.payment_schedule import PaymentSchedule, build_payment_schedule

FloatArray = np.ndarray
DateArray = np.ndarray


@serializable("NotionalEvent")
@dataclass
class NotionalEvent(BaseEvent):
    """From the accrual period starting on ``start_date``, ``notional`` is outstanding."""

    notional: float = 0.0


@serializable("NotionalSchedule")
@dataclass
class NotionalSchedule(BaseSchedule):
    """Dated outstanding notionals; periods before the first event carry the leg's face."""

    events: list[NotionalEvent] = field(default_factory=list)

    def profile(self, accrual_starts: DateArray, face: float) -> FloatArray:
        """Outstanding notional over each accrual period.

        Parameters
        ----------
        accrual_starts : ndarray of datetime64[D], shape (n_periods,)
        face : float
            Outstanding before the first event.

        Returns
        -------
        ndarray of float64, shape (n_periods,)
        """
        dates = np.array([event.start_date.to_numpy() for event in self.events], dtype="datetime64[D]")
        values = np.array([face] + [float(event.notional) for event in self.events])
        return values[np.searchsorted(dates, accrual_starts, side="right")]


def notional_profile(leg, schedule: PaymentSchedule) -> FloatArray:
    """The leg's outstanding notional over each period of its payment schedule.

    Parameters
    ----------
    leg : CommonInstrument
    schedule : PaymentSchedule
        The leg's own accrual grid.

    Returns
    -------
    ndarray of float64, shape (n_periods,)

    Raises
    ------
    ValueError
        If a principal date does not fall on a coupon payment date (the principal frequency
        must be the coupon frequency or a slower multiple, counted from ``amort_start``).
    NotImplementedError
        For ``LevelPay`` on a leg with a floating coupon segment.
    """
    face = float(leg.notional)
    kind = leg.amortization
    if kind is AmortizationType.Bullet:
        return np.full(schedule.n_periods, face)
    if kind is AmortizationType.Custom:
        return leg.notional_schedule.profile(schedule.accrual_starts, face)

    amort_start: Date = leg.amort_start or leg.effective
    amort_end: Date = leg.amort_end or leg.maturity
    term_end: Date = (
        add_term(amort_start, leg.amortization_term, BDC.NoAdjustment, "no_holidays")
        if leg.amortization_term is not None
        else amort_end
    )
    frequency: Frequency = leg.principal_frequency or leg.payment_frequency

    # Principal periods over the solve horizon, on the leg's own roll and calendar.
    horizon = build_payment_schedule(
        amort_start, term_end, frequency, leg.day_count_method, leg.business_day_convention, leg.pay_calendar,
        roll=leg.roll_convention, direction=leg.direction,
    )
    principal_dates = horizon.accrual_ends  # K principal dates; a paydown lands on each
    amort_end_np = np.datetime64(amort_end.to_numpy(), "D")
    inside = principal_dates <= amort_end_np
    if not np.isin(principal_dates[inside], schedule.accrual_ends).all():
        raise ValueError(
            f"principal dates at {frequency.name} from {amort_start} do not all fall on coupon payment dates; "
            "the principal frequency must be the coupon frequency or a slower multiple of it"
        )

    if kind is AmortizationType.StraightLine:
        paydown = np.full(principal_dates.size, face / principal_dates.size)
        balances = face - np.concatenate([[0.0], np.cumsum(paydown)])  # before each principal period, then after the last
    else:
        balances = _level_pay_balances(leg, face, horizon)

    # Contract periods: outstanding = balance entering the principal period that contains the
    # period's start; face before amort_start (k = 0); zero from the balloon date on.
    k = np.searchsorted(principal_dates, schedule.accrual_starts, side="right")
    out = balances[k]
    out[schedule.accrual_starts >= amort_end_np] = 0.0
    return out


def _level_pay_balances(leg, face: float, horizon: PaymentSchedule) -> FloatArray:
    """Balance entering each principal period of ``horizon`` (and after the last), annuity-style.

    The payment is solved from the balance outstanding whenever the coupon rate changes, over the
    periods that remain, at that rate; between changes it is held.  Exact for any accrual
    fractions: ``P = B / Σ_i Π_{j≤i} 1 / (1 + r τ_j)``.
    """
    events = leg.coupon_schedule.events
    if any(event.coupon_type.is_floating for event in events):
        raise NotImplementedError(
            "LevelPay on a floating leg re-solves the payment from each reset's rate, so the balance path "
            "depends on the projected rates and belongs to a reprice-time kernel stage, not a compile-time "
            "profile; that stage is not built yet"
        )
    governing = leg.coupon_schedule.schedule_from_accrual_grid(horizon.accrual_starts)
    rates = np.array([float(events[i].coupon_rate) for i in governing])
    fracs = horizon.period_fracs
    balances = np.empty(fracs.size + 1)
    balance = face
    payment = 0.0
    for k in range(fracs.size):
        balances[k] = balance
        if k == 0 or rates[k] != rates[k - 1]:
            growth = 1.0 + rates[k] * fracs[k:]
            payment = balance / np.cumprod(1.0 / growth).sum()
        balance = balance * (1.0 + rates[k] * fracs[k]) - payment
    balances[-1] = max(balance, 0.0)  # fully amortising solves to zero up to rounding
    return balances
