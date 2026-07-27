"""JaxProgram — a differentiable reprice over the columnar KernelInputs.

This is the JAX-backend twin of ``compiler.reprice``: the *same* ``KernelInputs`` the numpy
engine consumes, lifted into a pure ``jax.numpy`` valuation so risk falls out of autodiff.
It mirrors the numpy kernel period-for-period — fixed / single-fixing float / daily
compounded / arithmetic averaged, plus spread, per-fixing ``index_floor``, and the
final-period ``cap``/``floor`` with inclusive vs exclusive margin treatment.

Two design choices make it useful for risk:

1. **Static geometry, differentiable parameters.**  Everything date-shaped (day offsets,
   day-count fractions, segment maps) is frozen at build time as gradient-free constants.
   The *only* leaves are the per-curve zero-rate vectors ``z`` — exactly what the calibrator
   solves for and ``Sensitivities`` bumps.

2. **Role-split parameters: ``proj`` vs ``disc``.**  A flow's cash reads the *projection*
   curve; its discount factor reads the *funding* curve.  We keep these as **two separate
   pytrees** (``proj_params`` and ``disc_params``), even when they are the same curve name.
   So ``∂PV/∂proj_params`` is **index (projection) risk** and ``∂PV/∂disc_params`` is
   **funding (discount) risk** — cleanly separated by differentiation, in single-curve OIS as
   well as in a genuine basis where the two are different curves entirely.
"""
from __future__ import annotations

import numpy as np

import jax.numpy as jnp
from jax.ops import segment_sum

from finance.pricing.engines.jax.curves import CurveGeometry, curve_geometry, make_df
from finance.pricing.kernels.inputs import KernelInputs
from finance.pricing.kernels.shaping import prep_obs, shape_float, shape_period
from finance.pricing.types import RateKind
from finance.instruments.enums import MarginTreatment

_INCLUSIVE = int(MarginTreatment.Inclusive)
_EXCLUSIVE = int(MarginTreatment.Exclusive)


def _act360(start_off: np.ndarray, end_off: np.ndarray) -> np.ndarray:
    """Act/360 day-count fraction from day offsets — the convention the rate kernels use."""
    return (end_off - start_off) / 360.0


class JaxProgram:
    """Compiled, differentiable repricer bound to a market's curve *geometry*.

    Geometry (pillar dates / interpolation) is fixed at construction from ``market``; the
    free variables are the per-curve zero-rate vectors.  ``params_from_market`` returns the
    initial ``(disc_params, proj_params)`` pytrees (dicts keyed by curve name); pass shifted
    copies to reprice a scenario, or hand the closures to ``jax.grad`` for risk.
    """

    def __init__(self, inputs: KernelInputs, market):
        self.inputs = inputs
        ki = inputs
        self.curve_names: list[str] = list(ki.curve_names)

        # -- static curve geometry (gradient-free), one per curve name ------------------
        self.geoms: dict[str, CurveGeometry] = {
            name: curve_geometry(market.zero_curve(name)) for name in self.curve_names
        }
        self._df = {name: make_df(g) for name, g in self.geoms.items()}

        # -- per-flow static columns ----------------------------------------------------
        o = market.zero_curve(self.curve_names[0]).origin.astype(np.int64) if self.curve_names else 0

        def offs(dates: np.ndarray) -> np.ndarray:
            return (dates.astype(np.int64) - o).astype(np.float64)

        self.F = ki.n_flows
        self.pay_off = jnp.asarray(offs(ki.pay_dates))
        self.period_frac = jnp.asarray(ki.period_frac)
        self.notional = jnp.asarray(ki.notional)
        self.sign = jnp.asarray(ki.sign)
        self.fixed_rate = jnp.asarray(ki.fixed_rate)
        self.spread = jnp.asarray(ki.spread)
        self.index_floor = jnp.asarray(ki.index_floor)
        self.cap = jnp.asarray(ki.cap)
        self.floor = jnp.asarray(ki.floor)
        self.margin = np.asarray(ki.margin)

        self.discount_curve = np.asarray(ki.discount_curve)
        self.proj_curve = np.asarray(ki.proj_curve)

        self.is_fixed = jnp.asarray(ki.rate_kind == int(RateKind.Fixed))
        self.is_float = ki.rate_kind == int(RateKind.Float)

        # single-fixing float accrual window (Act/360 simple rate)
        rs_off = offs(ki.reset_starts) if ki.reset_starts.size else np.zeros(self.F)
        re_off = offs(ki.reset_ends) if ki.reset_ends.size else np.zeros(self.F)
        self.reset_start_off = jnp.asarray(rs_off)
        self.reset_end_off = jnp.asarray(re_off)
        self.float_tau = jnp.asarray(np.where(self.is_float, _act360(rs_off, re_off), 1.0))

        # -- leg / instrument reduction maps --------------------------------------------
        leg_sizes = np.diff(np.append(ki.leg_offsets, ki.n_flows))
        self.flow_leg = jnp.asarray(np.repeat(np.arange(ki.n_legs), leg_sizes))
        self.n_legs = ki.n_legs
        self.leg_instrument = jnp.asarray(ki.leg_instrument)
        self.n_instruments = ki.n_instruments
        self.flow_instrument = jnp.asarray(np.repeat(ki.leg_instrument, leg_sizes))

        # -- observation grid (compounded / averaged) ----------------------------------
        self.P = ki.n_obs_periods
        if self.P:
            M = ki.obs_w.shape[0]
            self.M = M
            obs_s = offs(ki.obs_starts)
            obs_e = offs(ki.obs_ends)
            self.obs_tau = jnp.asarray(_act360(obs_s, obs_e))
            self.obs_w = jnp.asarray(ki.obs_w)
            # Dedup the daily-fixing curve lookups (the dominant cost on a book): every SOFR
            # instrument shares the same fixing calendar, so we evaluate ``df`` once per UNIQUE
            # observation date and gather, instead of once per fixing.  The unique grid and the
            # scatter-back inverse indices are static (dates are fixed at compile), so the gather
            # is a plain index op — fully differentiable.  This mirrors numpy's ``_project_dedup``.
            all_obs = np.concatenate([obs_s, obs_e])
            uniq, inverse = np.unique(all_obs, return_inverse=True)
            inverse = inverse.reshape(-1)
            self.obs_unique_off = jnp.asarray(uniq)              # (U,)  U << M on a real book
            self.obs_inv_start = jnp.asarray(inverse[:M])        # (M,)  unique-idx of each start
            self.obs_inv_end = jnp.asarray(inverse[M:])          # (M,)  unique-idx of each end
            per_obs_period = np.repeat(np.arange(self.P), np.diff(np.append(ki.obs_offsets, M)))
            self.per_obs_period = jnp.asarray(per_obs_period)
            self.obs_flow = np.asarray(ki.obs_flow)               # (P,) period -> flow
            self.obs_proj_curve = np.asarray(ki.obs_proj_curve)   # (P,) projection curve per period
            flow_of_obs = self.obs_flow[per_obs_period]           # (M,)
            self.obs_proj_of_obs = self.obs_proj_curve[per_obs_period]
            # per-observation shaping params (broadcast from the governing flow)
            self.obs_index_floor = jnp.asarray(ki.index_floor[flow_of_obs])
            self.obs_spread = jnp.asarray(ki.spread[flow_of_obs])
            self.obs_incl = jnp.asarray(ki.margin[flow_of_obs] == _INCLUSIVE)
            # per-period shaping params
            self.period_kind = np.asarray(ki.rate_kind[self.obs_flow])
            self.period_spread = jnp.asarray(ki.spread[self.obs_flow])
            self.period_excl = jnp.asarray(ki.margin[self.obs_flow] == _EXCLUSIVE)
            self.period_floor = jnp.asarray(ki.floor[self.obs_flow])
            self.period_cap = jnp.asarray(ki.cap[self.obs_flow])
            self.obs_flow_j = jnp.asarray(self.obs_flow)
            self.period_is_comp = jnp.asarray(self.period_kind == int(RateKind.Compounded))

    # -- parameter plumbing -------------------------------------------------------------
    def params_from_market(self, market) -> tuple[dict, dict]:
        """Initial ``(disc_params, proj_params)`` — zero-rate vectors keyed by curve name.

        Both dicts start from the same calibrated curves, but they are *separate* leaves:
        differentiating in ``disc_params`` gives funding risk, in ``proj_params`` index risk.
        """
        z = {name: jnp.asarray(curve_geometry(market.zero_curve(name)).z0) for name in self.curve_names}
        return {k: v for k, v in z.items()}, {k: jnp.asarray(np.asarray(v)) for k, v in z.items()}

    # -- the differentiable valuation ---------------------------------------------------
    def _df_per_flow(self, disc_params: dict) -> jnp.ndarray:
        """(F,) discount factor at each flow's pay date off its funding curve."""
        df = jnp.ones(self.F)
        for ci, name in enumerate(self.curve_names):
            mask = jnp.asarray(self.discount_curve == ci)
            df = jnp.where(mask, self._df[name](disc_params[name], self.pay_off), df)
        return df

    def _rate(self, proj_params: dict) -> jnp.ndarray:
        """(F,) realized per-period rate — fixed / float / compounded / averaged, shaped."""
        rate = jnp.where(self.is_fixed, self.fixed_rate, 0.0)

        # -- single-fixing float: simple rate over the accrual window, then shape -------
        if bool(self.is_float.any()):
            is_float = jnp.asarray(self.is_float)
            idx = jnp.zeros(self.F)
            for ci, name in enumerate(self.curve_names):
                mm = jnp.asarray((self.proj_curve == ci) & self.is_float)
                df_s = self._df[name](proj_params[name], self.reset_start_off)
                df_e = self._df[name](proj_params[name], self.reset_end_off)
                simple = (df_s / df_e - 1.0) / self.float_tau
                idx = jnp.where(mm, simple, idx)
            idx = shape_float(idx, self.index_floor, self.spread, self.floor, self.cap, xp=jnp)
            rate = jnp.where(is_float, idx, rate)

        # -- compounded / averaged over the daily observation grid ----------------------
        if self.P:
            obs_rate = jnp.zeros(self.M)
            for ci, name in enumerate(self.curve_names):
                mm = jnp.asarray(self.obs_proj_of_obs == ci)
                df_u = self._df[name](proj_params[name], self.obs_unique_off)  # once per unique date
                df_s = df_u[self.obs_inv_start]
                df_e = df_u[self.obs_inv_end]
                simple = (df_s / df_e - 1.0) / self.obs_tau
                obs_rate = jnp.where(mm, simple, obs_rate)
            # per-fixing index floor, then inclusive spread
            obs_rate = prep_obs(obs_rate, self.obs_index_floor, self.obs_spread, self.obs_incl, xp=jnp)

            log_g = jnp.log1p(obs_rate * self.obs_w)
            growth = jnp.expm1(segment_sum(log_g, self.per_obs_period, num_segments=self.P))
            weight = segment_sum(self.obs_w, self.per_obs_period, num_segments=self.P)
            comp = growth / weight
            avg = segment_sum(obs_rate * self.obs_w, self.per_obs_period, num_segments=self.P) / weight
            period_rate = jnp.where(self.period_is_comp, comp, avg)
            # exclusive spread, then floor / cap on the final-period coupon
            period_rate = shape_period(
                period_rate, self.period_spread, self.period_excl, self.period_floor, self.period_cap, xp=jnp
            )
            rate = rate.at[self.obs_flow_j].set(period_rate)

        return rate

    def flow_pv(self, disc_params: dict, proj_params: dict) -> jnp.ndarray:
        """(F,) per-flow PV = notional · rate · frac · DF(pay) · sign."""
        cash = self.notional * self._rate(proj_params) * self.period_frac
        return cash * self._df_per_flow(disc_params) * self.sign

    def leg_pv(self, disc_params: dict, proj_params: dict) -> jnp.ndarray:
        """(L,) per-leg PV."""
        return segment_sum(self.flow_pv(disc_params, proj_params), self.flow_leg, num_segments=self.n_legs)

    def instrument_pv(self, disc_params: dict, proj_params: dict) -> jnp.ndarray:
        """(n_instruments,) per-instrument PV."""
        return segment_sum(
            self.flow_pv(disc_params, proj_params), self.flow_instrument, num_segments=self.n_instruments
        )

    def pv(self, disc_params: dict, proj_params: dict) -> jnp.ndarray:
        """Scalar total PV across the whole program (handy for grad/hessian)."""
        return jnp.sum(self.flow_pv(disc_params, proj_params))


__all__ = ["JaxProgram"]
