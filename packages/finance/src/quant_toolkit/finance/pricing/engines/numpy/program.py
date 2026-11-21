"""NumpyProgram — the reference engine: vectorised numpy over the columnar ``KernelInputs``.

Reads curves through ``RateGenerator`` on every reprice (no precomputed geometry), groups flows
by ``rate_kind``, and reduces with ``reduceat``.  It is the correctness reference the faster
engines are held to, and the one engine that needs no optional dependency.  It has no adjoint:
risk on this backend is bump-and-reprice (``Sensitivities``).
"""
from __future__ import annotations

import numpy as np

from quant_toolkit.finance.instruments.enums import MarginTreatment
from quant_toolkit.finance.markets.rate_generator import RateGenerator
from quant_toolkit.finance.pricing.engines.numpy.dcf import dcf
from quant_toolkit.finance.pricing.engines.numpy.rates import averaged, compounded
from quant_toolkit.finance.pricing.kernels.inputs import KernelInputs, KernelResult
from quant_toolkit.finance.pricing.kernels.shaping import prep_obs, shape_float, shape_period
from quant_toolkit.finance.pricing.types import RateKind


class NumpyProgram:
    """Compiled inputs repriced by the numpy engine; stateless beyond the inputs."""

    def __init__(self, inputs: KernelInputs, market=None):
        self.inputs = inputs

    def compatible(self, market) -> bool:
        """Always: nothing is prepared against the market's geometry."""
        return True

    def reprice(self, market) -> KernelResult:
        """Reprice the compiled portfolio against a market (the per-scenario hot path)."""
        ki = self.inputs
        F = ki.n_flows
        rate = np.zeros(F, dtype=np.float64)
        rate_generator = RateGenerator(market)

        # -- discount factors, one curve.discount_factor call per discount curve --
        df = np.zeros(F, dtype=np.float64)
        for ci, name in enumerate(ki.curve_names):
            curve = market[name]
            mask = (ki.discount_curve == ci) & (ki.pay_dates >= curve.origin)
            if mask.any():
                df[mask] = curve.discount_factor(ki.pay_dates[mask])
        active_flow = df > 0.0

        # -- fixed (incl. step-ups: just a varying fixed_rate column) --
        fx = ki.rate_kind == RateKind.Fixed
        rate[fx] = ki.fixed_rate[fx]

        # -- simple float: project over the accrual window, then shape --
        fl = (ki.rate_kind == RateKind.Float) & active_flow
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
            active_obs = active_flow[flow_of_obs]

            obs_rate = np.zeros(M, dtype=np.float64)
            obs_proj_of_obs = ki.obs_proj_curve[per_obs_period]
            for ci, name in enumerate(ki.curve_names):
                mm = (obs_proj_of_obs == ci) & active_obs
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

        return KernelResult(
            instrument_pv=instrument_pv,
            leg_pv=leg_pv,
            flow_pv=flow_pv,
            rate=rate,
            df=df,
        )
