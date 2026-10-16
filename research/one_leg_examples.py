"""One-legged instruments: every coupon shape a leg supports, built flat or with a schedule.

A leg is ``CommonInstrument``.  The flat coupon fields (``coupon_type``, ``coupon_rate``,
``spread``, ``cap`` ...) describe one coupon for the whole life and are lowered into a
one-event ``CouponSchedule`` at construction; passing ``coupon_schedule`` directly tags the leg
``CouponType.Custom`` and is the general path.  Either way the pricer reads only the schedule.

Dating is always generated from the flat fields (``payment_frequency``, stubs, BDC, calendar,
``payment_delay``); there is no slot for an explicit payment schedule.  Piecewise-regular
dating (monthly, then quarterly) is several legs joined end to end — ``join_legs`` — since the
pricer rolls every leg of an instrument into one PV anyway.  The reset cadence is separate
from payment: ``reset_frequency`` slower than payment repeats one governing fixing across
periods, equal/unset reads the accrual window, faster builds the observation grid.

Each example prints the compiled columns the engine sees and the PV against a flat-ish curve.
The same shapes are pinned as regression tests in ``finance/tests/regression/test_leg_shapes.py``.

Run from the repo root:

    PYTHONPATH=src python research/one_leg_examples.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finance.dates import BDC, Date, DayCountMethod, Frequency, Roll
from finance.dates.enums import FixingType
from finance.instruments import CommonInstrument, join_legs
from finance.instruments.enums import CouponType
from finance.instruments.schedules import CompoundedCouponEvent, CouponSchedule, FixedCouponEvent
from finance.instruments.schedules.observation import LookbackStyle
from finance.markets import HistoricalFixings
from finance.markets.context import MarketContext
from finance.markets.curves import YieldCurve
from finance.pricing.pricers import LegPricer
from finance.pricing.types import RateKind

AS_OF = Date(2026, 6, 1)
EFFECTIVE = Date(2026, 6, 3)
MATURITY = Date(2029, 6, 3)
INDEX = "USD.SOFR.ON"       # the leg's projection key
NOTIONAL = 10_000_000.0


# ---------------------------------------------------------------------------------------------
# A product is signed legs plus the three curve labels; LegPricer takes it from there.
# ---------------------------------------------------------------------------------------------
@dataclass(kw_only=True)
class Legs:
    legs: tuple[tuple[CommonInstrument, float], ...]
    currency: str = "USD"
    index_name: str = "SOFR"
    funding_id: str = "STDCSA"


def market() -> MarketContext:
    origin = AS_OF.to_numpy()
    dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 90, 365, 730, 1095, 1825)])
    t = (dates.astype(np.int64) - origin.astype(np.int64)) / 365.0
    z = np.array([0.0, 0.041, 0.040, 0.039, 0.040, 0.042])
    # Example 3 fixes in advance: its first coupon reads a fixing set a year before as_of, so
    # the curve carries that print.  Fixings travel with the curve; the generator decides
    # "historical" by comparing the window start with the curve origin.
    fixings = HistoricalFixings(dates=np.array(["2025-06-03"], dtype="datetime64[D]"), values=np.array([0.0435]))
    sofr = YieldCurve.build(dates, np.exp(-z * t), currency="USD", index_name="SOFR", historical_fixings=fixings)
    return MarketContext(AS_OF, {sofr.name: sofr, "USD.STDCSA": sofr})


def leg(**fields) -> CommonInstrument:
    """A USD leg with the dating every example shares; coupon fields come from the caller."""
    base = dict(
        effective=EFFECTIVE, maturity=MATURITY, currency="USD", notional=NOTIONAL,
        day_count_method=DayCountMethod.Actual360, business_day_convention=BDC.ModifiedFollowing,
        roll_convention=Roll.Empty, pay_calendar="no_holidays",
    )
    base.update(fields)
    return CommonInstrument(**base)


# ---------------------------------------------------------------------------------------------
# The examples
# ---------------------------------------------------------------------------------------------
def examples() -> list[tuple[str, tuple[tuple[CommonInstrument, float], ...]]]:
    def one(single: CommonInstrument):
        return ((single, 1.0),)  # a single receive leg
    out = []

    out.append(("1. Fixed 4.25%, semi-annual", one(leg(
        payment_frequency=Frequency.SemiAnnually,
        coupon_type=CouponType.Fixed, coupon_rate=0.0425,
    ))))

    out.append(("2. Floating, quarterly fixing = quarterly pay (term-rate style)", one(leg(
        payment_frequency=Frequency.Quarterly,
        coupon_type=CouponType.Floating, rate_index=INDEX, spread=0.0015,
    ))))

    out.append(("3. Floating, annual reset in advance, pays monthly (one fixing governs 12 coupons; year 1 = the 2025-06-03 print)", one(leg(
        payment_frequency=Frequency.Monthly, reset_frequency=Frequency.Annually,
        fixing_type=FixingType.Advance,
        coupon_type=CouponType.Floating, rate_index=INDEX, spread=0.0020,
    ))))

    out.append(("4. Monthly compounder paying quarterly (ObservationShift: each fixing reads its month)", one(leg(
        payment_frequency=Frequency.Quarterly, reset_frequency=Frequency.Monthly,
        lookback_style=LookbackStyle.ObservationShift,
        coupon_type=CouponType.GeometricAveraged, rate_index=INDEX,
    ))))

    out.append(("5. Daily compounder paying monthly (SOFR OIS leg)", one(leg(
        payment_frequency=Frequency.Monthly, reset_frequency=Frequency.Daily,
        coupon_type=CouponType.GeometricAveraged, rate_index=INDEX, spread=0.0010,
    ))))

    out.append(("6. Daily arithmetic average paying monthly (SR1-style loan)", one(leg(
        payment_frequency=Frequency.Monthly, reset_frequency=Frequency.Daily,
        coupon_type=CouponType.ArithmeticAveraged, rate_index=INDEX, spread=0.0025,
    ))))

    out.append(("7. Step-up fixed: 3% / 4% / 5% by year (schedule -> CouponType.Custom)", one(leg(
        payment_frequency=Frequency.SemiAnnually,
        coupon_schedule=CouponSchedule(events=[
            FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.03),
            FixedCouponEvent(start_date=Date(2027, 6, 3), coupon_rate=0.04),
            FixedCouponEvent(start_date=Date(2028, 6, 3), coupon_rate=0.05),
        ]),
    ))))

    out.append(("8. Floating quarterly with 2% floor / 6% cap on the period coupon, 0% index floor", one(leg(
        payment_frequency=Frequency.Quarterly,
        coupon_type=CouponType.Floating, rate_index=INDEX, spread=0.0030,
        index_floor=0.0, floor=0.02, cap=0.06,
    ))))

    out.append(("9. Fixed 3.5% for year 1, then daily-compounded SOFR + 15bp (custom schedule)", one(leg(
        payment_frequency=Frequency.Quarterly, reset_frequency=Frequency.Daily,
        rate_index=INDEX,  # leg-level: applies to every floating segment
        coupon_schedule=CouponSchedule(events=[
            FixedCouponEvent(start_date=EFFECTIVE, coupon_rate=0.035),
            CompoundedCouponEvent(start_date=Date(2027, 6, 3), rate_index=INDEX, spread=0.0015),
        ]),
    ))))

    # Piecewise-regular dating: two legs joined where the first matures.  Fixed 4% paid
    # monthly for two years, then quarterly to maturity — one instrument, two generated grids.
    join = Date(2028, 6, 3)
    out.append(("10. Fixed 4%, monthly for 2Y then quarterly (two legs joined with join_legs)", join_legs(
        leg(maturity=join, payment_frequency=Frequency.Monthly, coupon_type=CouponType.Fixed, coupon_rate=0.04),
        leg(effective=join, payment_frequency=Frequency.Quarterly, coupon_type=CouponType.Fixed, coupon_rate=0.04),
    )))

    return out


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------
def describe(title: str, legs, mkt: MarketContext) -> None:
    program = LegPricer().compile([Legs(legs=legs)])
    ki = program.inputs
    res = program.price(mkt)
    cf = res.cashflows

    kinds = sorted({RateKind(int(k)).name for k in ki.rate_kind})
    print(f"\n{title}")
    for i, (one, _sign) in enumerate(legs):
        events = [c.name for c in one.coupon_schedule.coupon_types]
        print(f"   leg {i}: coupon_type={one.coupon_type.name:<18} events={events}  {one.effective} -> {one.maturity}")
    print(f"   periods={ki.n_flows:<3} legs={ki.n_legs} rate_kinds={kinds}  obs_periods={ki.n_obs_periods}  fixings={ki.obs_w.size}")
    print(f"   pay dates: {ki.pay_dates[0]} .. {ki.pay_dates[-1]}")
    first = legs[0][0]
    if first.is_floating and ki.n_obs_periods == 0:
        # single-fixing float: show the governing reset window per period (first three)
        windows = [f"[{s} -> {e}]" for s, e in zip(ki.reset_starts[:3], ki.reset_ends[:3])]
        print(f"   reset windows (first 3): {' '.join(windows)}")
    if ki.n_obs_periods:
        m = int(np.diff(np.append(ki.obs_offsets, ki.obs_w.size))[0])
        print(f"   first obs period: {m} fixings, first read window [{ki.obs_starts[0]} -> {ki.obs_ends[0]}], weight {ki.obs_w[0]:.5f}")
    rates = " ".join(f"{r:.4%}" for r in cf.rate[:4])
    print(f"   realized rates (first 4): {rates}")
    if ki.n_legs > 1:
        print(f"   leg PVs: {', '.join(f'{v:,.2f}' for v in res.leg_pv)}")
    print(f"   PV = {res.pv:,.2f}")


def main() -> None:
    mkt = market()
    print(f"as_of {AS_OF}, leg {EFFECTIVE} -> {MATURITY}, notional {NOTIONAL:,.0f}, Act/360, MF, no_holidays")
    for title, legs in examples():
        describe(title, legs, mkt)


if __name__ == "__main__":
    main()
