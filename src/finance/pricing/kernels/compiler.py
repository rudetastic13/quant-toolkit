"""Compiler: lower a portfolio of legs into columnar KernelInputs, and reprice them.

``compile_portfolio`` walks each leg's ``PaymentSchedule`` + ``CouponSchedule`` and emits
the per-flow columns of ``KernelInputs`` (the compile-once step).  ``reprice`` consumes a
``KernelInputs`` + a ``MarketContext`` and produces PVs, grouping flows by ``rate_kind`` so
the per-scenario work is a handful of vectorized kernel calls regardless of population size.

Piecewise coupons (fixed -> float -> fixed step-ups, N switches) are native:
``CouponSchedule.schedule_from_accrual_grid`` maps each accrual period to its governing
event, and each period's columns are filled from that event.  A step-up is just a varying
``fixed_rate`` column; a switch is just a varying ``rate_kind`` column.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finance.instruments.enums import CouponType, MarginTreatment
from finance.instruments.schedules.coupon_schedule import CouponSchedule
from finance.instruments.schedules.payment_schedule import PaymentSchedule
from finance.markets.rate_generator import RateGenerator
from finance.pricing.engines.numpy.rates import compounded, averaged
from finance.pricing.engines.numpy.dcf import dcf
from finance.pricing.kernels.inputs import (
    KernelInputs,
    KernelResult,
    NotionalProvider,
    NO_CAP,
    NO_FLOOR,
    NO_CURVE,
)
from finance.pricing.kernels.shaping import prep_obs, shape_float, shape_period
from finance.pricing.types import RateKind

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
    notional : NotionalProvider
        Static/scheduled (materialized at compile); rate-dependent is a designed seam.
    instrument : int
        Instrument index, so leg PVs roll up to net instrument PV.
    """

    schedule: PaymentSchedule
    coupon: CouponSchedule
    sign: float
    discount_curve: str
    projection_curve: str | None
    notional: NotionalProvider
    instrument: int = 0


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

        pay.append(ps.payment_dates.astype("datetime64[D]"))
        frac.append(np.asarray(ps.period_fracs, dtype=np.float64))
        sign.append(np.full(n, float(leg.sign)))
        notl.append(leg.notional.materialize(n))
        rkind.append(rk)
        frate.append(np.array([c["fixed_rate"] for c in cols], dtype=np.float64))
        spr.append(np.array([c["spread"] for c in cols], dtype=np.float64))
        ifloor.append(np.array([c["index_floor"] for c in cols], dtype=np.float64))
        cap.append(np.array([c["cap"] for c in cols], dtype=np.float64))
        floor.append(np.array([c["floor"] for c in cols], dtype=np.float64))
        marg.append(np.array([c["margin"] for c in cols], dtype=np.int8))
        disc.append(np.full(n, disc_id, dtype=np.int64))
        proj.append(np.where(rk == RateKind.Fixed, NO_CURVE, proj_id).astype(np.int64))
        # Float projection windows: the schedule's governing reset windows (fixing type and
        # reset-frequency aware); accrual windows for schedules that didn't build them.
        has_reset = ps.reset_starts is not None
        rstart.append((ps.reset_starts if has_reset else ps.accrual_starts).astype("datetime64[D]"))
        rend.append((ps.reset_ends if has_reset else ps.accrual_ends).astype("datetime64[D]"))

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
                obs_rs.append(ps.obs_read_starts[s:e])
                obs_re.append(ps.obs_read_ends[s:e])
                obs_w.append(ps.obs_weights[s:e])
                obs_offsets.append(m)
                obs_flow.append(f + int(p))
                obs_proj.append(proj_id)
                m += e - s

        f += n

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


def reprice(ki: KernelInputs, market) -> KernelResult:
    """Reprice a compiled portfolio against a market (the per-scenario hot path)."""
    F = ki.n_flows
    rate = np.zeros(F, dtype=np.float64)
    rate_generator = RateGenerator(market)

    # -- discount factors, one curve.discount_factor call per discount curve --
    df = np.ones(F, dtype=np.float64)
    for ci, name in enumerate(ki.curve_names):
        mask = ki.discount_curve == ci
        if mask.any():
            df[mask] = market.zero_curve(name).discount_factor(ki.pay_dates[mask])

    # -- fixed (incl. step-ups: just a varying fixed_rate column) --
    fx = ki.rate_kind == RateKind.Fixed
    rate[fx] = ki.fixed_rate[fx]

    # -- simple float: project over the accrual window, then shape --
    fl = ki.rate_kind == RateKind.Float
    if fl.any():
        idx = np.zeros(int(fl.sum()), dtype=np.float64)
        proj_fl = ki.proj_curve[fl]
        rs_fl, re_fl = ki.reset_starts[fl], ki.reset_ends[fl]
        for ci, name in enumerate(ki.curve_names):
            mm = proj_fl == ci
            if mm.any():
                idx[mm] = rate_generator.simple_rate(name, rs_fl[mm], re_fl[mm])
        rate[fl] = shape_float(idx, ki.index_floor[fl], ki.spread[fl], ki.floor[fl], ki.cap[fl])

    # -- compounded / averaged via the global observation grid --
    P = ki.n_obs_periods
    if P:
        M = ki.obs_w.shape[0]
        # per-observation period index, then expand period-level params to observations
        per_obs_period = np.repeat(np.arange(P), np.diff(np.append(ki.obs_offsets, M)))
        flow_of_period = ki.obs_flow                       # (P,)
        flow_of_obs = flow_of_period[per_obs_period]        # (M,)

        obs_rate = np.zeros(M, dtype=np.float64)
        obs_proj_of_obs = ki.obs_proj_curve[per_obs_period]
        for ci, name in enumerate(ki.curve_names):
            mm = obs_proj_of_obs == ci
            if mm.any():
                # dedup the rate lookup across the whole portfolio's fixing grid
                obs_rate[mm] = rate_generator.simple_rate(
                    name,
                    ki.obs_starts[mm],
                    ki.obs_ends[mm],
                )

        # input prep: per-fixing index_floor, then inclusive spread (broadcast from the period)
        incl = ki.margin[flow_of_obs] == int(MarginTreatment.Inclusive)
        obs_rate = prep_obs(obs_rate, ki.index_floor[flow_of_obs], ki.spread[flow_of_obs], incl)

        # single-kind portfolios skip the reduction they'd discard anyway
        period_kind = ki.rate_kind[flow_of_period]
        is_comp = period_kind == RateKind.Compounded
        if is_comp.all():
            period_rate = compounded(obs_rate, ki.obs_w, ki.obs_offsets)   # (P,)
        elif not is_comp.any():
            period_rate = averaged(obs_rate, ki.obs_w, ki.obs_offsets)     # (P,)
        else:
            comp = compounded(obs_rate, ki.obs_w, ki.obs_offsets)
            avg = averaged(obs_rate, ki.obs_w, ki.obs_offsets)
            period_rate = np.where(is_comp, comp, avg)

        # output prep: exclusive spread, then floor/cap on the final period coupon
        excl = ki.margin[flow_of_period] == int(MarginTreatment.Exclusive)
        period_rate = shape_period(
            period_rate, ki.spread[flow_of_period], excl, ki.floor[flow_of_period], ki.cap[flow_of_period]
        )

        rate[flow_of_period] = period_rate

    # -- assemble cash, discount, reduce to leg then instrument PV --
    cash = ki.notional * rate * ki.period_frac
    leg_pv = dcf(cash, df, ki.sign, ki.leg_offsets)
    instrument_pv = np.bincount(ki.leg_instrument, weights=leg_pv, minlength=ki.n_instruments)
    flow_pv = cash * df * ki.sign

    return KernelResult(instrument_pv=instrument_pv, leg_pv=leg_pv, flow_pv=flow_pv, rate=rate)


__all__ = ["LegSpec", "compile_portfolio", "reprice"]
