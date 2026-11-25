"""Compiler: lower a portfolio of legs into columnar KernelInputs.

``compile_portfolio`` walks each leg's ``PaymentSchedule`` + ``CouponSchedule`` and emits
the per-flow columns of ``KernelInputs`` (the compile-once step).  Repricing lives in the
engines (``quant_toolkit.finance.pricing.engines``), which all consume the same ``KernelInputs``.

Piecewise coupons (fixed -> float -> fixed step-ups, N switches) are native:
``CouponSchedule.schedule_from_accrual_grid`` maps each accrual period to its governing
event, and each period's columns are filled from that event.  A step-up is just a varying
``fixed_rate`` column; a switch is just a varying ``rate_kind`` column.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_toolkit.finance.instruments.enums import CouponType, MarginTreatment, PrincipalExchange
from quant_toolkit.finance.instruments.schedules.coupon_schedule import CouponSchedule
from quant_toolkit.finance.instruments.schedules.payment_schedule import PaymentSchedule
from quant_toolkit.finance.pricing.kernels.inputs import (
    KernelInputs,
    NO_CAP,
    NO_FLOOR,
    NO_CURVE,
)
from quant_toolkit.finance.pricing.types import RateKind

_KIND = {
    CouponType.Zero: RateKind.Fixed,
    CouponType.Fixed: RateKind.Fixed,
    CouponType.Floating: RateKind.Float,
    CouponType.GeometricAveraged: RateKind.Compounded,
    CouponType.ArithmeticAveraged: RateKind.Averaged,
}


@dataclass
class LegSpec:
    """Compiler input for one leg.

    coupon : CouponSchedule
        Authored source of truth (one or many events). Lowered to per-period columns.
    sign : float
        +1 receive / -1 pay.
    discount_curve, projection_curve : str
        Curve names resolved against the MarketContext at reprice time.
    notional : ndarray of float64, shape (n_periods,)
        Notional outstanding over each accrual period.  A bullet repeats one amount; scheduled
        amortisation (straight-line, custom, a constant prepayment speed) is any vector known
        at compile time.  A notional that depends on the realised rate (level-pay on a
        floater, rate-driven prepayment) is not a column and is not supported here.
    instrument : int
        Instrument index, so leg PVs roll up to net instrument PV.
    principal_exchange : PrincipalExchange
        Principal flows on top of the coupons: ``Initial`` pays ``-notional[0]`` on the first
        accrual start; ``Final`` pays each paydown ``notional[i] - notional[i + 1]`` on payment
        date ``i`` and the last notional on the last payment date, so a bullet produces one
        final flow and an amortiser one per step.  Lowered as fixed flows with
        ``rate = period_frac = 1``, so ``cash = amount`` and no kernel knows about them.
    """

    schedule: PaymentSchedule
    coupon: CouponSchedule
    sign: float
    discount_curve: str
    projection_curve: str | None
    notional: np.ndarray
    instrument: int = 0
    principal_exchange: PrincipalExchange = PrincipalExchange.Unused


def _bound(value, sentinel: float) -> float:
    """Map an event bound to a column value: ``None`` -> 'no bound' sentinel; else the value.

    A genuine ``0.0`` is preserved (e.g. a SOFR coupon floored at 0%) — only ``None`` means
    "no bound".
    """
    if value is None:
        return sentinel
    return float(value)


def _lower_event(event) -> dict:
    """Lower one CouponEvent to its column values."""
    kind = _KIND.get(event.coupon_type, RateKind.Fixed)
    return {
        "rate_kind": int(kind),
        "fixed_rate": float(getattr(event, "coupon_rate", 0.0) or 0.0),
        "spread": float(getattr(event, "spread", 0.0) or 0.0),
        "index_floor": _bound(getattr(event, "index_floor", None), NO_FLOOR),
        "cap": _bound(getattr(event, "cap", None), NO_CAP),
        "floor": _bound(getattr(event, "floor", None), NO_FLOOR),
        "margin": int(getattr(event, "margin_treatment", MarginTreatment.Inclusive)),
    }


def compile_portfolio(legs: list[LegSpec]) -> KernelInputs:
    """Lower a list of legs into one columnar ``KernelInputs`` (compile once)."""
    curve_names: list[str] = []

    def curve_id(name: str | None) -> int:
        if name is None:
            return NO_CURVE
        if name not in curve_names:
            curve_names.append(name)
        return curve_names.index(name)

    # per-flow column accumulators
    pay, frac, sign, notl = [], [], [], []
    rkind, frate, spr, ifloor, cap, floor, marg = [], [], [], [], [], [], []
    disc, proj, rstart, rend = [], [], [], []
    principal_flag = []
    leg_offsets, leg_instrument = [], []

    # global obs accumulators
    obs_rs, obs_re, obs_w = [], [], []
    obs_offsets, obs_flow, obs_proj = [], [], []

    f = 0          # running flow count
    m = 0          # running obs count

    for leg in legs:
        ps = leg.schedule
        n = ps.n_periods
        leg_offsets.append(f)
        leg_instrument.append(leg.instrument)

        # lower the (possibly piecewise) coupon to per-period columns
        event_idx = leg.coupon.schedule_from_accrual_grid(ps.accrual_starts)
        cols = [_lower_event(leg.coupon.events[i]) for i in event_idx]

        rk = np.array([c["rate_kind"] for c in cols], dtype=np.int8)
        disc_id = curve_id(leg.discount_curve)
        proj_id = curve_id(leg.projection_curve)

        amounts = np.asarray(leg.notional, dtype=np.float64)
        if amounts.shape != (n,):
            raise ValueError(f"leg notional must have one value per period: expected shape ({n},), got {amounts.shape}")
        pay.append(ps.payment_dates.astype("datetime64[D]"))
        frac.append(np.asarray(ps.period_fracs, dtype=np.float64))
        sign.append(np.full(n, float(leg.sign)))
        notl.append(amounts)
        rkind.append(rk)
        frate.append(np.array([c["fixed_rate"] for c in cols], dtype=np.float64))
        spr.append(np.array([c["spread"] for c in cols], dtype=np.float64))
        ifloor.append(np.array([c["index_floor"] for c in cols], dtype=np.float64))
        cap.append(np.array([c["cap"] for c in cols], dtype=np.float64))
        floor.append(np.array([c["floor"] for c in cols], dtype=np.float64))
        marg.append(np.array([c["margin"] for c in cols], dtype=np.int8))
        disc.append(np.full(n, disc_id, dtype=np.int64))
        proj.append(np.where(rk == RateKind.Fixed, NO_CURVE, proj_id).astype(np.int64))
        principal_flag.append(np.zeros(n, dtype=np.bool_))
        # Float projection windows: the schedule's governing reset windows (fixing type and
        # reset-frequency aware); accrual windows for schedules that didn't build them.
        has_reset = ps.reset_starts is not None
        rstart.append((ps.reset_starts if has_reset else ps.accrual_starts).astype("datetime64[D]"))
        rend.append((ps.reset_ends if has_reset else ps.accrual_ends).astype("datetime64[D]"))

        # Principal flows after the coupons: fixed flows with rate = period_frac = 1.  A paydown
        # is the drop in outstanding notional at each payment date (to zero after the last), so
        # a bullet lowers to one final flow and an amortiser to one flow per step.
        principal = []
        if leg.principal_exchange.initial:
            principal.append((ps.accrual_starts[0], -amounts[0]))
        if leg.principal_exchange.final:
            paydown = amounts - np.append(amounts[1:], 0.0)
            for i in np.flatnonzero(paydown):
                principal.append((ps.payment_dates[i], paydown[i]))
        for date, amount in principal:
            pay.append(np.array([date], dtype="datetime64[D]"))
            frac.append(np.array([1.0]))
            sign.append(np.array([float(leg.sign)]))
            notl.append(np.array([amount], dtype=np.float64))
            rkind.append(np.array([RateKind.Fixed], dtype=np.int8))
            frate.append(np.array([1.0]))
            spr.append(np.array([0.0]))
            ifloor.append(np.array([NO_FLOOR]))
            cap.append(np.array([NO_CAP]))
            floor.append(np.array([NO_FLOOR]))
            marg.append(np.array([int(MarginTreatment.Inclusive)], dtype=np.int8))
            disc.append(np.array([disc_id], dtype=np.int64))
            proj.append(np.array([NO_CURVE], dtype=np.int64))
            principal_flag.append(np.ones(1, dtype=np.bool_))
            rstart.append(np.array([date], dtype="datetime64[D]"))
            rend.append(np.array([date], dtype="datetime64[D]"))

        # global observation grid: only compounded/averaged periods contribute fixings
        needs_obs = (rk == RateKind.Compounded) | (rk == RateKind.Averaged)
        if needs_obs.any():
            if not ps.has_observation_grid:
                raise ValueError(
                    "leg has compounded/averaged periods but its PaymentSchedule has no "
                    "observation grid (build_payment_schedule(..., build_observations=True))."
                )
            leg_off = np.append(ps.obs_offsets, len(ps.obs_weights))
            for p in np.nonzero(needs_obs)[0]:
                s, e = int(leg_off[p]), int(leg_off[p + 1])
                obs_rs.append(ps.obs_fixing_starts[s:e])
                obs_re.append(ps.obs_fixing_ends[s:e])
                obs_w.append(ps.obs_weights[s:e])
                obs_offsets.append(m)
                obs_flow.append(f + int(p))
                obs_proj.append(proj_id)
                m += e - s

        f += n + len(principal)

    def cat_dt(xs):
        return np.concatenate(xs).astype("datetime64[D]") if xs else np.array([], dtype="datetime64[D]")

    def cat_f(xs):
        return np.concatenate(xs).astype(np.float64) if xs else np.array([], dtype=np.float64)

    return KernelInputs(
        pay_dates=cat_dt(pay),
        period_frac=cat_f(frac),
        sign=cat_f(sign),
        notional=cat_f(notl),
        rate_kind=np.concatenate(rkind).astype(np.int8) if rkind else np.array([], np.int8),
        fixed_rate=cat_f(frate),
        spread=cat_f(spr),
        index_floor=cat_f(ifloor),
        cap=cat_f(cap),
        floor=cat_f(floor),
        margin=np.concatenate(marg).astype(np.int8) if marg else np.array([], np.int8),
        discount_curve=np.concatenate(disc).astype(np.int64) if disc else np.array([], np.int64),
        proj_curve=np.concatenate(proj).astype(np.int64) if proj else np.array([], np.int64),
        reset_starts=cat_dt(rstart),
        reset_ends=cat_dt(rend),
        is_principal=np.concatenate(principal_flag) if principal_flag else np.array([], np.bool_),
        leg_offsets=np.array(leg_offsets, dtype=np.intp),
        leg_instrument=np.array(leg_instrument, dtype=np.intp),
        n_instruments=(max(leg_instrument) + 1) if leg_instrument else 0,
        obs_starts=cat_dt(obs_rs),
        obs_ends=cat_dt(obs_re),
        obs_w=cat_f(obs_w),
        obs_offsets=np.array(obs_offsets, dtype=np.intp),
        obs_flow=np.array(obs_flow, dtype=np.intp),
        obs_proj_curve=np.array(obs_proj, dtype=np.int64),
        curve_names=curve_names,
    )


__all__ = ["LegSpec", "compile_portfolio"]
