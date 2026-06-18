"""Numba forward-pricing kernel — a hand-rolled twin of the numpy ``reprice``.

Same columnar ``KernelInputs`` layout the numpy and JAX engines consume, lowered into one
``@njit`` pass: fixed / single-fixing float / daily-compounded / arithmetic-averaged, with
spread, per-fixing ``index_floor``, and final-period ``cap``/``floor`` + margin treatment.

The point of the prototype is the compile model.  Numba specialises on **dtype, not shape**:
the ``@njit(cache=True)`` kernel compiles ONCE for ``(float64[:], int64[:], ...)`` and then
runs for any book size — a 1-swap test and a 2,000-swap book hit the *same* compiled code, no
recompile.  That is the property the JAX backend can't offer (it specialises per array shape).

Curves are flattened to ``(curve_x, curve_logdf, curve_node_off)`` so the LogLinearDF
interpolation (+ flat-forward extrapolation) runs inside nopython mode with no Python objects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from finance.pricing.kernels.inputs import KernelInputs  # noqa: E402
from finance.pricing.types import RateKind
from finance.instruments.enums import MarginTreatment

FIXED = int(RateKind.Fixed)
FLOAT = int(RateKind.Float)
COMPOUNDED = int(RateKind.Compounded)
AVERAGED = int(RateKind.Averaged)

# Explicit signatures => Numba compiles EAGERLY at import (and writes the disk cache), instead
# of lazily on first call.  No first-reprice latency in the hot path, and the signature is
# self-documenting.  ``[::1]`` pins C-contiguous 1-D arrays (what ``prepare`` always produces);
# ``F`` = #flows, ``M`` = #fixings, ``U`` = #unique fixing dates, ``P`` = #compounded periods.
_DF_SIG = "float64(int64, float64, float64[::1], float64[::1], int64[::1])"
_REPRICE_SIG = (
    "float64[::1]("
    "float64[::1], float64[::1], float64[::1], float64[::1], int64[::1], float64[::1], "
    "float64[::1], float64[::1], float64[::1], float64[::1], boolean[::1], boolean[::1], "
    "int64[::1], int64[::1], float64[::1], float64[::1], float64[::1], int64[::1], int64, "
    "float64[::1], float64[::1], int64[::1], int64[::1], int64[::1], int64[::1], int64[::1], "
    "float64[::1], int64, float64[::1], float64[::1], int64[::1], int64)"
)


# ---------------------------------------------------------------------------
# Curve discounting inside nopython mode: LogLinearDF + flat-forward extrap.
# ---------------------------------------------------------------------------
@njit(_DF_SIG, cache=True, inline="always")
def _df(ci, xq, cx, cl, coff):
    """DF(xq) off curve ``ci`` — log-linear in ln(DF), flat-forward beyond the last pillar."""
    lo = coff[ci]
    hi = coff[ci + 1]
    if xq <= cx[lo]:
        return np.exp(cl[lo])
    if xq >= cx[hi - 1]:                       # flat-forward: continue last segment slope
        slope = (cl[hi - 1] - cl[hi - 2]) / (cx[hi - 1] - cx[hi - 2])
        return np.exp(cl[hi - 1] + slope * (xq - cx[hi - 1]))
    a = lo
    b = hi - 1
    while b - a > 1:                            # binary search for the bracketing nodes
        mid = (a + b) // 2
        if cx[mid] <= xq:
            a = mid
        else:
            b = mid
    lg = cl[a] + (cl[b] - cl[a]) * (xq - cx[a]) / (cx[b] - cx[a])
    return np.exp(lg)


@njit(_REPRICE_SIG, cache=True, fastmath=True)
def numba_reprice(
    # per-flow columns
    pay_off, period_frac, sign, notional, rate_kind, fixed_rate, spread,
    index_floor, cap, floor, margin_incl, margin_excl, discount_curve, proj_curve,
    reset_start_off, reset_end_off, float_tau, flow_instrument, n_instruments,
    # observation grid (compounded / averaged)
    obs_w, obs_tau, per_obs_period, obs_flow, obs_proj_period, obs_inv_start, obs_inv_end,
    obs_unique_off, P,
    # flattened curves
    curve_x, curve_logdf, curve_node_off, n_curves,
):
    F = pay_off.shape[0]
    M = obs_w.shape[0]
    rate = np.zeros(F)

    # -- discount factor per flow (its funding curve, at the pay date) ------------------
    df = np.empty(F)
    for f in range(F):
        df[f] = _df(discount_curve[f], pay_off[f], curve_x, curve_logdf, curve_node_off)

    # -- fixed + single-fixing float ----------------------------------------------------
    for f in range(F):
        k = rate_kind[f]
        if k == FIXED:
            rate[f] = fixed_rate[f]
        elif k == FLOAT:
            ci = proj_curve[f]
            dfs = _df(ci, reset_start_off[f], curve_x, curve_logdf, curve_node_off)
            dfe = _df(ci, reset_end_off[f], curve_x, curve_logdf, curve_node_off)
            r = (dfs / dfe - 1.0) / float_tau[f]
            if r < index_floor[f]:
                r = index_floor[f]
            r += spread[f]
            if r < floor[f]:
                r = floor[f]
            if r > cap[f]:
                r = cap[f]
            rate[f] = r

    # -- compounded / averaged over the daily obs grid (deduped curve lookups) ----------
    if P > 0:
        U = obs_unique_off.shape[0]
        df_uniq = np.empty((n_curves, U))       # DF once per unique date per curve, then gather
        for ci in range(n_curves):
            for u in range(U):
                df_uniq[ci, u] = _df(ci, obs_unique_off[u], curve_x, curve_logdf, curve_node_off)

        g_log = np.zeros(P)                      # Σ log1p(r·w)   (compounded)
        sw = np.zeros(P)                         # Σ w
        sa = np.zeros(P)                         # Σ r·w          (averaged)
        for m in range(M):
            p = per_obs_period[m]
            flow = obs_flow[p]
            ci = obs_proj_period[p]
            dfs = df_uniq[ci, obs_inv_start[m]]
            dfe = df_uniq[ci, obs_inv_end[m]]
            r = (dfs / dfe - 1.0) / obs_tau[m]
            if r < index_floor[flow]:            # per-fixing index floor
                r = index_floor[flow]
            if margin_incl[flow]:                # inclusive spread
                r += spread[flow]
            w = obs_w[m]
            g_log[p] += np.log1p(r * w)
            sw[p] += w
            sa[p] += r * w

        for p in range(P):
            flow = obs_flow[p]
            if rate_kind[flow] == COMPOUNDED:
                pr = np.expm1(g_log[p]) / sw[p]
            else:
                pr = sa[p] / sw[p]
            if margin_excl[flow]:                # exclusive spread
                pr += spread[flow]
            if pr < floor[flow]:
                pr = floor[flow]
            if pr > cap[flow]:
                pr = cap[flow]
            rate[flow] = pr

    # -- cash, discount, reduce to instrument PV ----------------------------------------
    inst_pv = np.zeros(n_instruments)
    for f in range(F):
        flow_pv = notional[f] * rate[f] * period_frac[f] * df[f] * sign[f]
        inst_pv[flow_instrument[f]] += flow_pv
    return inst_pv


# ---------------------------------------------------------------------------
# Static extraction: KernelInputs + market -> the plain numpy args above.
# ---------------------------------------------------------------------------
@dataclass
class NumbaInputs:
    args: tuple

    def reprice(self):
        return numba_reprice(*self.args)


def _act360(s, e):
    return (e - s) / 360.0


def prepare(program, market) -> NumbaInputs:
    """Pull a compiled ``PricingProgram`` + market into the kernel's numpy arguments (once)."""
    ki: KernelInputs = program.inputs
    names = list(ki.curve_names)
    origin = market.discount(names[0]).origin.astype(np.int64)

    def offs(dates):
        return (dates.astype(np.int64) - origin).astype(np.float64)

    F = ki.n_flows
    is_float = ki.rate_kind == FLOAT
    rs = offs(ki.reset_starts) if ki.reset_starts.size else np.zeros(F)
    re = offs(ki.reset_ends) if ki.reset_ends.size else np.zeros(F)
    float_tau = np.where(is_float, _act360(rs, re), 1.0)

    leg_sizes = np.diff(np.append(ki.leg_offsets, F))
    flow_instrument = np.repeat(ki.leg_instrument, leg_sizes).astype(np.int64)

    margin_incl = (ki.margin == int(MarginTreatment.Inclusive))
    margin_excl = (ki.margin == int(MarginTreatment.Exclusive))

    # observation grid + static dedup of the fixing-date curve lookups
    P = ki.n_obs_periods
    if P:
        M = ki.obs_w.shape[0]
        obs_s = offs(ki.obs_starts)
        obs_e = offs(ki.obs_ends)
        all_obs = np.concatenate([obs_s, obs_e])
        uniq, inv = np.unique(all_obs, return_inverse=True)
        inv = inv.reshape(-1)
        per_obs_period = np.repeat(np.arange(P), np.diff(np.append(ki.obs_offsets, M))).astype(np.int64)
        obs_tau = _act360(obs_s, obs_e)
        obs_inv_start = inv[:M].astype(np.int64)
        obs_inv_end = inv[M:].astype(np.int64)
        obs_unique_off = uniq.astype(np.float64)
        obs_flow = ki.obs_flow.astype(np.int64)
        obs_proj_period = ki.obs_proj_curve.astype(np.int64)
        obs_w = ki.obs_w.astype(np.float64)
    else:
        per_obs_period = np.zeros(0, np.int64)
        obs_tau = obs_inv_start = obs_inv_end = obs_unique_off = obs_w = np.zeros(0, np.float64)
        obs_flow = obs_proj_period = np.zeros(0, np.int64)
        obs_inv_start = obs_inv_start.astype(np.int64)
        obs_inv_end = obs_inv_end.astype(np.int64)

    # flatten curves -> (x, logdf, node_off)
    cx, cl, coff = [], [], [0]
    for name in names:
        c = market.discount(name)
        cx.append(offs(c.node_dates))
        cl.append(np.log(c.node_dfs))
        coff.append(coff[-1] + c.node_dates.shape[0])
    curve_x = np.concatenate(cx).astype(np.float64)
    curve_logdf = np.concatenate(cl).astype(np.float64)
    curve_node_off = np.array(coff, dtype=np.int64)

    args = (
        offs(ki.pay_dates), ki.period_frac.astype(np.float64), ki.sign.astype(np.float64),
        ki.notional.astype(np.float64), ki.rate_kind.astype(np.int64), ki.fixed_rate.astype(np.float64),
        ki.spread.astype(np.float64), ki.index_floor.astype(np.float64), ki.cap.astype(np.float64),
        ki.floor.astype(np.float64), margin_incl, margin_excl,
        ki.discount_curve.astype(np.int64), ki.proj_curve.astype(np.int64),
        rs, re, float_tau, flow_instrument, int(ki.n_instruments),
        obs_w, obs_tau, per_obs_period, obs_flow, obs_proj_period, obs_inv_start, obs_inv_end,
        obs_unique_off, int(P),
        curve_x, curve_logdf, curve_node_off, len(names),
    )
    return NumbaInputs(args=args)


__all__ = ["numba_reprice", "prepare", "NumbaInputs"]
