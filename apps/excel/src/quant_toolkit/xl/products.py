"""``QuantToolkit.IRProducts.*`` worksheet functions (xlOil): instruments, returned as handles.

One function per product type, so each shows only the arguments that product has. All of them
build through the rate index's conventions: an optional argument left out, or pointed at an
empty cell, takes the convention's value.

Dating is the same everywhere: ``Effective`` is a date, or a term from ``AsOf`` (``"2BD"``);
``Maturity`` is a date, or a term from the effective date (``"5Y"``). ``AsOf`` is only read when
``Effective`` is a term, and is today if left out.

The sign of the notional is the direction: positive receives fixed, negative pays fixed.
"""
from __future__ import annotations

from quant_toolkit.finance.conventions.keys import CurveKey
from quant_toolkit.finance.dates import Date, DayCountMethod, Frequency
from quant_toolkit.finance.dates.enums import FixingType
from quant_toolkit.finance.instruments import resolution
from quant_toolkit.finance.instruments.enums import CouponType

from quant_toolkit.xl.core import handles
from quant_toolkit.xl.core.convert import date_or_term, enum_member, given, term
from quant_toolkit.xl.core.register import xl_func


def _fixed_float_swap(
    Notional,
    RateIndex,
    FixedRate,
    Effective,
    Maturity,
    AsOf=None,
    FixedPayFrequency=None,
    FloatPayFrequency=None,
    PaymentDelay=None,
    FixedDayCount=None,
    FloatDayCount=None,
    FloatResetFrequency=None,
    FloatFixingCalendar=None,
    FloatFixingType=None,
    FloatCouponType=None,
    FloatRateLookback=None,
    FloatRateLockout=None,
    FloatSpread=None,
    FloatCap=None,
    FloatFloor=None,
    FloatIndexFloor=None,
):
    """Convert cell values, build the swap through ``Swap.fixed_float_swap`` and store it.

    The shared body of the swap worksheet functions; the arguments are theirs, documented on
    :func:`FixedFloatSwap`. Each one is passed to the library by name, and one that is ``None``
    or an empty string is passed as ``None`` so the rate index's convention applies.

    Returns
    -------
    str
        A ``Swap;;;<digest>`` handle. The digest is taken over the swap's own JSON, so the same
        contract always gives the same handle however it was specified (dates or terms), and a
        term resolved on a different day gives a different one.
    """
    index = str(RateIndex).strip()
    swap = resolution.Swap.fixed_float_swap(
        notional=float(Notional),
        rate_index=index,
        fixed_rate=float(FixedRate),
        effective=date_or_term(Effective),
        maturity=date_or_term(Maturity),
        as_of=Date.from_excel(AsOf) if given(AsOf) else None,
        currency=CurveKey.parse(index).currency,
        fixed_frequency=enum_member(Frequency, FixedPayFrequency) if given(FixedPayFrequency) else None,
        float_frequency=enum_member(Frequency, FloatPayFrequency) if given(FloatPayFrequency) else None,
        payment_delay=term(PaymentDelay) if given(PaymentDelay) else None,
        fixed_day_count=enum_member(DayCountMethod, FixedDayCount) if given(FixedDayCount) else None,
        float_day_count=enum_member(DayCountMethod, FloatDayCount) if given(FloatDayCount) else None,
        reset_frequency=enum_member(Frequency, FloatResetFrequency) if given(FloatResetFrequency) else None,
        fixing_calendar=str(FloatFixingCalendar).strip().lower() if given(FloatFixingCalendar) else None,
        fixing_type=enum_member(FixingType, FloatFixingType) if given(FloatFixingType) else None,
        float_coupon_type=enum_member(CouponType, FloatCouponType) if given(FloatCouponType) else None,
        rate_lookback=term(FloatRateLookback) if given(FloatRateLookback) else None,
        rate_lockout=term(FloatRateLockout) if given(FloatRateLockout) else None,
        spread=float(FloatSpread) if given(FloatSpread) else 0.0,
        cap=float(FloatCap) if given(FloatCap) else None,
        floor=float(FloatFloor) if given(FloatFloor) else None,
        index_floor=float(FloatIndexFloor) if given(FloatIndexFloor) else None,
    )
    # keyed on the contract itself, so a term resolved on a different day is a different handle
    return handles.store("Swap", swap, swap.to_json(indent=None))


@xl_func("IRProducts")
def FixedFloatSwap(  # noqa: N802
    Notional,
    RateIndex,
    FixedRate,
    Effective,
    Maturity,
    AsOf=None,
    FixedPayFrequency=None,
    FloatPayFrequency=None,
    PaymentDelay=None,
    FixedDayCount=None,
    FloatDayCount=None,
    FloatResetFrequency=None,
    FloatFixingCalendar=None,
    FloatFixingType=None,
    FloatCouponType=None,
    FloatRateLookback=None,
    FloatRateLockout=None,
    FloatSpread=None,
    FloatCap=None,
    FloatFloor=None,
    FloatIndexFloor=None,
):
    """Fixed-float swap built from the rate index's conventions, with every leg term overridable.

    Returns a Swap handle. Positive Notional receives fixed, negative pays fixed.

    Parameters
    ----------
    Notional : float
        Size and direction: positive receives fixed, negative pays fixed.
    RateIndex : str
        The floating index as a curve key, e.g. "USD.SOFR.1D" or "USD.CME-TERM-SOFR.3M". Supplies the conventions.
    FixedRate : float
        Fixed coupon as a decimal (0.0385 for 3.85%).
    Effective : date or str
        Start date, or a term from AsOf such as "2BD".
    Maturity : date or str
        End date, or a term from the effective date such as "5Y".
    AsOf : date, optional
        Date a term Effective is counted from. Today if omitted; not used when Effective is a date.
    FixedPayFrequency : str, optional
        Fixed leg payment frequency: "Monthly", "Quarterly", "SemiAnnually", "Annually". Convention if omitted.
    FloatPayFrequency : str, optional
        Floating leg payment frequency. Convention if omitted.
    PaymentDelay : str or int, optional
        Lag from period end to payment, e.g. "2BD"; a number is days. Convention if omitted.
    FixedDayCount : str, optional
        Fixed leg day count: "Actual360", "Actual365", "ActualActual", "Thirty360", "ThirtyE360". Convention if omitted.
    FloatDayCount : str, optional
        Floating leg day count. Convention if omitted.
    FloatResetFrequency : str, optional
        How often the floating rate resets within a payment period. Convention if omitted.
    FloatFixingCalendar : str, optional
        Calendar the fixings are observed on, e.g. "US_SIFMA". Convention if omitted.
    FloatFixingType : str, optional
        "Arrears" (fixing known at period end) or "Advance" (at period start). Convention if omitted.
    FloatCouponType : str, optional
        How fixings make a coupon: "Floating", "GeometricAveraged" (compounded), "ArithmeticAveraged". Convention
        if omitted.
    FloatRateLookback : str or int, optional
        Observation shift back from each accrual day, e.g. "2BD". Convention if omitted.
    FloatRateLockout : str or int, optional
        Days before period end over which the last fixing is repeated, e.g. "2BD". Convention if omitted.
    FloatSpread : float, optional
        Spread added to the floating rate, as a decimal. Zero if omitted.
    FloatCap : float, optional
        Cap on the floating coupon rate, as a decimal. None if omitted.
    FloatFloor : float, optional
        Floor on the floating coupon rate, as a decimal. None if omitted.
    FloatIndexFloor : float, optional
        Floor on each index fixing before compounding or averaging, as a decimal. None if omitted.

    Returns
    -------
    str
        A Swap handle ("Swap;;;74086109689f") for the Pricing and Risk functions. The same
        contract always gives the same handle.

    Raises
    ------
    ValueError
        If Maturity is not after Effective, or RateIndex is not a known projection key.

    Notes
    -----
    The swap discounts on the currency's standard CSA key (``USD.STDCSA``). Leg-level business
    day conventions, rolls, calendars and stubs are not exposed and take the convention.

    Examples
    --------
    ``=QuantToolkit.IRProducts.FixedFloatSwap(-250000000, "USD.SOFR.1D", 0.0385, "2BD", "5Y")``
    """
    return _fixed_float_swap(
        Notional, RateIndex, FixedRate, Effective, Maturity, AsOf,
        FixedPayFrequency, FloatPayFrequency, PaymentDelay, FixedDayCount, FloatDayCount,
        FloatResetFrequency, FloatFixingCalendar, FloatFixingType, FloatCouponType,
        FloatRateLookback, FloatRateLockout, FloatSpread, FloatCap, FloatFloor, FloatIndexFloor,
    )


@xl_func("IRProducts")
def OvernightIndexSwap(  # noqa: N802
    Notional,
    RateIndex,
    FixedRate,
    Effective,
    Maturity,
    AsOf=None,
    FixedPayFrequency=None,
    FloatPayFrequency=None,
    PaymentDelay=None,
    FixedDayCount=None,
    FloatDayCount=None,
    FloatRateLookback=None,
    FloatRateLockout=None,
    FloatSpread=None,
):
    """Overnight index swap: fixed against the compounded overnight rate.

    Returns a Swap handle. Positive Notional receives fixed, negative pays fixed.

    Parameters
    ----------
    Notional : float
        Size and direction: positive receives fixed, negative pays fixed.
    RateIndex : str
        An overnight index as a curve key, e.g. "USD.SOFR.1D". Supplies the conventions.
    FixedRate : float
        Fixed coupon as a decimal (0.0385 for 3.85%).
    Effective : date or str
        Start date, or a term from AsOf such as "2BD".
    Maturity : date or str
        End date, or a term from the effective date such as "5Y".
    AsOf : date, optional
        Date a term Effective is counted from. Today if omitted; not used when Effective is a date.
    FixedPayFrequency : str, optional
        Fixed leg payment frequency: "Monthly", "Quarterly", "SemiAnnually", "Annually". Convention if omitted.
    FloatPayFrequency : str, optional
        Floating leg payment frequency. Convention if omitted.
    PaymentDelay : str or int, optional
        Lag from period end to payment, e.g. "2BD"; a number is days. Convention if omitted.
    FixedDayCount : str, optional
        Fixed leg day count: "Actual360", "Actual365", "ActualActual", "Thirty360", "ThirtyE360". Convention if omitted.
    FloatDayCount : str, optional
        Floating leg day count. Convention if omitted.
    FloatRateLookback : str or int, optional
        Observation shift back from each accrual day, e.g. "2BD". Convention if omitted.
    FloatRateLockout : str or int, optional
        Days before period end over which the last fixing is repeated, e.g. "2BD". Convention if omitted.
    FloatSpread : float, optional
        Spread added to the compounded rate, as a decimal. Zero if omitted.

    Returns
    -------
    str
        A Swap handle ("Swap;;;74086109689f") for the Pricing and Risk functions. The same
        contract always gives the same handle.

    Raises
    ------
    ValueError
        If RateIndex is not an overnight index (use FixedFloatSwap for a term index), or
        Maturity is not after Effective.

    Notes
    -----
    The reset frequency, fixing calendar, fixing type, coupon type, caps and floors are fixed
    by the index's conventions here; FixedFloatSwap exposes them.

    Examples
    --------
    ``=QuantToolkit.IRProducts.OvernightIndexSwap(100000000, "USD.SOFR.1D", 0.038, "2BD", "5Y", $B$1)``
    """
    if not CurveKey.parse(str(RateIndex)).is_overnight:
        raise ValueError(f"{RateIndex!r} is not an overnight index; use FixedFloatSwap for a term index")
    return _fixed_float_swap(
        Notional, RateIndex, FixedRate, Effective, Maturity, AsOf,
        FixedPayFrequency, FloatPayFrequency, PaymentDelay, FixedDayCount, FloatDayCount,
        FloatRateLookback=FloatRateLookback, FloatRateLockout=FloatRateLockout, FloatSpread=FloatSpread,
    )
