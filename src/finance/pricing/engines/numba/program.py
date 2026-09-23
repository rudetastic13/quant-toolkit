"""Shape-polymorphic Numba pricing program with a hand-written reverse adjoint.

The kernel consumes the same columnar :class:`KernelInputs` as the NumPy and JAX engines,
but curve interpolation, coupon projection, shaping, discounting, and reduction all stay in
one nopython call.  Its only differentiable inputs are flattened continuously-compounded
pillar zero rates.  The adjoint scatters through log-linear interpolation's two local node
weights, producing exact projection and funding gradients per cashflow in the primal pass.

Numba specializes the function on dtype and dimensionality, not array length.  Consequently
one compiled signature prices a singleton, a changing futures strip, or a large swap book
without the shape-triggered recompilations of a traced JAX program.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from finance.instruments.enums import MarginTreatment
from finance.markets.curves import CurveInterpolator
from finance.pricing.kernels.inputs import KernelInputs, KernelResult
from finance.pricing.types import RateKind

_FIXED = int(RateKind.Fixed)
_FLOAT = int(RateKind.Float)
_COMPOUNDED = int(RateKind.Compounded)
_INCLUSIVE = int(MarginTreatment.Inclusive)
_EXCLUSIVE = int(MarginTreatment.Exclusive)


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _i64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int64)


def _b1(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.bool_)


@njit(cache=True, inline="always")  # pragma: no cover - executed as Numba machine code
def _node_param(node, curve_node_start, curve_param_start):
    """Flattened parameter index for a curve node; the pinned origin has no parameter."""
    if node == curve_node_start:
        return -1
    return curve_param_start + node - curve_node_start - 1


@njit(cache=True, inline="always")  # pragma: no cover - executed as Numba machine code
def _node_logdf(node, curve_node_start, curve_param_start, node_t, z):
    if node == curve_node_start:
        return 0.0
    p = curve_param_start + node - curve_node_start - 1
    return -z[p] * node_t[node]


@njit(cache=True, inline="always")  # pragma: no cover - executed as Numba machine code
def _df_details(ci, xq, curve_x, node_t, curve_node_off, curve_param_off, z):
    """Return DF plus the at-most-two nonzero derivatives of ln(DF) with respect to z."""
    lo = curve_node_off[ci]
    hi = curve_node_off[ci + 1]
    po = curve_param_off[ci]

    if xq < curve_x[lo]:
        return 0.0, -1, 0.0, -1, 0.0
    if xq == curve_x[lo]:
        return 1.0, -1, 0.0, -1, 0.0

    if xq >= curve_x[hi - 1]:
        left = hi - 2
        right = hi - 1
        w_right = 1.0 + (xq - curve_x[right]) / (curve_x[right] - curve_x[left])
        w_left = 1.0 - w_right
    else:
        left = lo
        right = hi - 1
        while right - left > 1:
            mid = (left + right) // 2
            if curve_x[mid] <= xq:
                left = mid
            else:
                right = mid
        w_right = (xq - curve_x[left]) / (curve_x[right] - curve_x[left])
        w_left = 1.0 - w_right

    logdf_left = _node_logdf(left, lo, po, node_t, z)
    logdf_right = _node_logdf(right, lo, po, node_t, z)
    df = np.exp(w_left * logdf_left + w_right * logdf_right)
    p_left = _node_param(left, lo, po)
    p_right = _node_param(right, lo, po)
    d_left = -node_t[left] * w_left if p_left >= 0 else 0.0
    d_right = -node_t[right] * w_right if p_right >= 0 else 0.0
    return df, p_left, d_left, p_right, d_right


@njit(cache=True, inline="always")  # pragma: no cover - executed as Numba machine code
def _scatter(row, p0, d0, p1, d1, scale):
    if p0 >= 0:
        row[p0] += scale * d0
    if p1 >= 0:
        row[p1] += scale * d1


@njit(cache=True)  # pragma: no cover - executed as Numba machine code
def _execute(
    # flow columns
    pay_off,
    period_frac,
    sign,
    notional,
    rate_kind,
    fixed_rate,
    spread,
    index_floor,
    cap,
    floor,
    margin,
    discount_curve,
    proj_curve,
    reset_start_off,
    reset_end_off,
    float_tau,
    reset_is_fixed,
    reset_fixing,
    flow_instrument,
    leg_offsets,
    leg_instrument,
    n_instruments,
    # observation grid
    obs_w,
    obs_tau,
    per_obs_period,
    obs_flow,
    obs_proj_period,
    obs_inv_start,
    obs_inv_end,
    obs_unique_off,
    obs_is_fixed,
    obs_fixing,
    # curve geometry and parameters
    curve_x,
    node_t,
    curve_node_off,
    curve_param_off,
    z,
    calculate_adjoint,
):
    f_count = pay_off.size
    m_count = obs_w.size
    period_count = obs_flow.size
    curve_count = curve_node_off.size - 1
    q_count = z.size

    rate = np.zeros(f_count)
    df = np.empty(f_count)
    disc_p0 = np.empty(f_count, dtype=np.int64)
    disc_p1 = np.empty(f_count, dtype=np.int64)
    disc_d0 = np.empty(f_count)
    disc_d1 = np.empty(f_count)

    # Discounting is independent of projection, so evaluate it once up front and retain
    # the local interpolation stencil for the funding adjoint.
    for f in range(f_count):
        details = _df_details(
            discount_curve[f], pay_off[f], curve_x, node_t, curve_node_off, curve_param_off, z
        )
        df[f] = details[0]
        disc_p0[f], disc_d0[f], disc_p1[f], disc_d1[f] = details[1], details[2], details[3], details[4]

    index_grad = np.zeros((f_count, q_count)) if calculate_adjoint else np.zeros((f_count, 0))
    funding_grad = np.zeros((f_count, q_count)) if calculate_adjoint else np.zeros((f_count, 0))

    # Fixed and single-fixing floating coupons.
    for f in range(f_count):
        kind = rate_kind[f]
        if kind == _FIXED:
            rate[f] = fixed_rate[f]
        elif kind == _FLOAT:
            if df[f] == 0.0:
                rate[f] = 0.0
                continue
            active = 1.0
            q_ratio = 0.0
            s_details = (1.0, -1, 0.0, -1, 0.0)
            e_details = (1.0, -1, 0.0, -1, 0.0)
            if reset_is_fixed[f]:
                projected = reset_fixing[f]
                active = 0.0
            else:
                ci = proj_curve[f]
                s_details = _df_details(
                    ci, reset_start_off[f], curve_x, node_t, curve_node_off, curve_param_off, z
                )
                e_details = _df_details(
                    ci, reset_end_off[f], curve_x, node_t, curve_node_off, curve_param_off, z
                )
                q_ratio = s_details[0] / e_details[0]
                projected = (q_ratio - 1.0) / float_tau[f]
            if projected < index_floor[f]:
                projected = index_floor[f]
                active = 0.0
            shaped = projected + spread[f]
            if shaped < floor[f]:
                shaped = floor[f]
                active = 0.0
            if shaped > cap[f]:
                shaped = cap[f]
                active = 0.0
            rate[f] = shaped

            if calculate_adjoint and active != 0.0:
                pv_per_rate = notional[f] * period_frac[f] * df[f] * sign[f]
                scale = pv_per_rate * q_ratio / float_tau[f]
                _scatter(index_grad[f], s_details[1], s_details[2], s_details[3], s_details[4], scale)
                _scatter(index_grad[f], e_details[1], e_details[2], e_details[3], e_details[4], -scale)

    # Compounded/averaged observation grids.  Endpoint discount factors are deduplicated
    # across the whole book and retain their local curve stencils for the adjoint gather.
    if period_count > 0:
        unique_count = obs_unique_off.size
        df_unique = np.empty((curve_count, unique_count))
        unique_p0 = np.empty((curve_count, unique_count), dtype=np.int64)
        unique_p1 = np.empty((curve_count, unique_count), dtype=np.int64)
        unique_d0 = np.empty((curve_count, unique_count))
        unique_d1 = np.empty((curve_count, unique_count))
        for ci in range(curve_count):
            for u in range(unique_count):
                details = _df_details(
                    ci, obs_unique_off[u], curve_x, node_t, curve_node_off, curve_param_off, z
                )
                df_unique[ci, u] = details[0]
                unique_p0[ci, u], unique_d0[ci, u] = details[1], details[2]
                unique_p1[ci, u], unique_d1[ci, u] = details[3], details[4]

        obs_rate = np.empty(m_count)
        obs_active = np.ones(m_count)
        growth_log = np.zeros(period_count)
        sum_weight = np.zeros(period_count)
        sum_arithmetic = np.zeros(period_count)
        for m in range(m_count):
            p = per_obs_period[m]
            flow = obs_flow[p]
            if df[flow] == 0.0:
                projected = 0.0
                obs_active[m] = 0.0
            elif obs_is_fixed[m]:
                projected = obs_fixing[m]
                obs_active[m] = 0.0
            else:
                ci = obs_proj_period[p]
                dfs = df_unique[ci, obs_inv_start[m]]
                dfe = df_unique[ci, obs_inv_end[m]]
                projected = (dfs / dfe - 1.0) / obs_tau[m]
            if projected < index_floor[flow]:
                projected = index_floor[flow]
                obs_active[m] = 0.0
            if margin[flow] == _INCLUSIVE:
                projected += spread[flow]
            obs_rate[m] = projected
            w = obs_w[m]
            growth_log[p] += np.log1p(projected * w)
            sum_weight[p] += w
            sum_arithmetic[p] += projected * w

        growth = np.empty(period_count)
        period_active = np.ones(period_count)
        for p in range(period_count):
            flow = obs_flow[p]
            growth[p] = np.exp(growth_log[p])
            if rate_kind[flow] == _COMPOUNDED:
                shaped = (growth[p] - 1.0) / sum_weight[p]
            else:
                shaped = sum_arithmetic[p] / sum_weight[p]
            if margin[flow] == _EXCLUSIVE:
                shaped += spread[flow]
            if shaped < floor[flow]:
                shaped = floor[flow]
                period_active[p] = 0.0
            if shaped > cap[flow]:
                shaped = cap[flow]
                period_active[p] = 0.0
            rate[flow] = shaped

        if calculate_adjoint:
            for m in range(m_count):
                if obs_active[m] == 0.0:
                    continue
                p = per_obs_period[m]
                if period_active[p] == 0.0:
                    continue
                flow = obs_flow[p]
                ci = obs_proj_period[p]
                us = obs_inv_start[m]
                ue = obs_inv_end[m]
                q_ratio = df_unique[ci, us] / df_unique[ci, ue]
                if rate_kind[flow] == _COMPOUNDED:
                    drate_dobs = growth[p] * obs_w[m] / (
                        sum_weight[p] * (1.0 + obs_rate[m] * obs_w[m])
                    )
                else:
                    drate_dobs = obs_w[m] / sum_weight[p]
                pv_per_rate = notional[flow] * period_frac[flow] * df[flow] * sign[flow]
                scale = pv_per_rate * drate_dobs * q_ratio / obs_tau[m]
                _scatter(
                    index_grad[flow], unique_p0[ci, us], unique_d0[ci, us],
                    unique_p1[ci, us], unique_d1[ci, us], scale,
                )
                _scatter(
                    index_grad[flow], unique_p0[ci, ue], unique_d0[ci, ue],
                    unique_p1[ci, ue], unique_d1[ci, ue], -scale,
                )

    # Cashflow PV and the discount/funding half of the adjoint.
    flow_pv = np.empty(f_count)
    for f in range(f_count):
        flow_pv[f] = notional[f] * rate[f] * period_frac[f] * df[f] * sign[f]
        if calculate_adjoint:
            _scatter(
                funding_grad[f], disc_p0[f], disc_d0[f], disc_p1[f], disc_d1[f], flow_pv[f]
            )

    leg_pv = np.zeros(leg_offsets.size)
    for leg in range(leg_offsets.size):
        end = leg_offsets[leg + 1] if leg + 1 < leg_offsets.size else f_count
        total = 0.0
        for f in range(leg_offsets[leg], end):
            total += flow_pv[f]
        leg_pv[leg] = total

    instrument_pv = np.zeros(n_instruments)
    for leg in range(leg_offsets.size):
        instrument_pv[leg_instrument[leg]] += leg_pv[leg]

    instrument_index_grad = np.zeros((n_instruments, q_count)) if calculate_adjoint else np.zeros((0, 0))
    instrument_funding_grad = np.zeros((n_instruments, q_count)) if calculate_adjoint else np.zeros((0, 0))
    if calculate_adjoint:
        for f in range(f_count):
            inst = flow_instrument[f]
            for q in range(q_count):
                instrument_index_grad[inst, q] += index_grad[f, q]
                instrument_funding_grad[inst, q] += funding_grad[f, q]

    return (
        instrument_pv,
        leg_pv,
        flow_pv,
        rate,
        index_grad,
        funding_grad,
        instrument_index_grad,
        instrument_funding_grad,
        df,
    )


@dataclass(frozen=True)
class NumbaAdjointResult:
    """Primal values and exact zero-rate gradients from one fused Numba execution."""

    primal: KernelResult
    cashflow_index_gradient: np.ndarray
    cashflow_funding_gradient: np.ndarray
    instrument_index_gradient: np.ndarray
    instrument_funding_gradient: np.ndarray
    curve_names: tuple[str, ...]
    curve_param_offsets: np.ndarray

    def curve_slice(self, curve_name: str) -> slice:
        try:
            ci = self.curve_names.index(curve_name)
        except ValueError:
            raise KeyError(f"program has no curve named '{curve_name}'") from None
        return slice(int(self.curve_param_offsets[ci]), int(self.curve_param_offsets[ci + 1]))

    def cashflow_gradient(self, curve_name: str, *, role: str = "total") -> np.ndarray:
        sl = self.curve_slice(curve_name)
        if role == "index":
            return self.cashflow_index_gradient[:, sl]
        if role == "funding":
            return self.cashflow_funding_gradient[:, sl]
        if role == "total":
            return self.cashflow_index_gradient[:, sl] + self.cashflow_funding_gradient[:, sl]
        raise ValueError("role must be 'index', 'funding', or 'total'")

    def instrument_gradient(self, curve_name: str, *, role: str = "total") -> np.ndarray:
        sl = self.curve_slice(curve_name)
        if role == "index":
            return self.instrument_index_gradient[:, sl]
        if role == "funding":
            return self.instrument_funding_gradient[:, sl]
        if role == "total":
            return self.instrument_index_gradient[:, sl] + self.instrument_funding_gradient[:, sl]
        raise ValueError("role must be 'index', 'funding', or 'total'")


class NumbaProgram:
    """Prepared pricing geometry reusable across arbitrary zero-rate values of one shape.

    The cashflow arrays may have any lengths when a new ``NumbaProgram`` is created; they all
    call the same Numba dispatcher signature.  Within a prepared program, ``reprice`` accepts
    any market with identical curve pillar geometry, which is the normal scenario/risk path.
    """

    def __init__(self, inputs: KernelInputs, market):
        self.inputs = inputs
        self.curve_names = tuple(inputs.curve_names)
        if not self.curve_names:
            raise ValueError("NumbaProgram requires at least one curve")

        origins = [np.datetime64(market.zero_curve(name).origin, "D") for name in self.curve_names]
        if any(origin != origins[0] for origin in origins[1:]):
            raise ValueError("all curves in a NumbaProgram must share the same origin")
        self.origin = origins[0]
        self._origin_ord = self.origin.astype(np.int64)

        def offs(dates):
            return _f64(dates.astype(np.int64) - self._origin_ord)

        ki = inputs
        f_count = ki.n_flows
        self.pay_off = offs(ki.pay_dates)
        self.period_frac = _f64(ki.period_frac)
        self.sign = _f64(ki.sign)
        self.notional = _f64(ki.notional)
        self.rate_kind = _i64(ki.rate_kind)
        self.fixed_rate = _f64(ki.fixed_rate)
        self.spread = _f64(ki.spread)
        self.index_floor = _f64(ki.index_floor)
        self.cap = _f64(ki.cap)
        self.floor = _f64(ki.floor)
        self.margin = _i64(ki.margin)
        self.discount_curve = _i64(ki.discount_curve)
        self.proj_curve = _i64(ki.proj_curve)
        self.reset_start_off = offs(ki.reset_starts) if ki.reset_starts.size else np.zeros(f_count)
        self.reset_end_off = offs(ki.reset_ends) if ki.reset_ends.size else np.zeros(f_count)
        is_float = self.rate_kind == _FLOAT
        self.float_tau = _f64(np.where(is_float, (self.reset_end_off - self.reset_start_off) / 360.0, 1.0))

        self.leg_offsets = _i64(ki.leg_offsets)
        self.leg_instrument = _i64(ki.leg_instrument)
        leg_sizes = np.diff(np.append(ki.leg_offsets, f_count))
        self.flow_instrument = _i64(np.repeat(ki.leg_instrument, leg_sizes))
        self.n_instruments = int(ki.n_instruments)

        period_count = ki.n_obs_periods
        if period_count:
            m_count = ki.obs_w.size
            obs_start_off = offs(ki.obs_starts)
            obs_end_off = offs(ki.obs_ends)
            all_obs = np.concatenate([obs_start_off, obs_end_off])
            unique, inverse = np.unique(all_obs, return_inverse=True)
            self.obs_unique_off = _f64(unique)
            self.obs_inv_start = _i64(inverse[:m_count])
            self.obs_inv_end = _i64(inverse[m_count:])
            self.per_obs_period = _i64(
                np.repeat(np.arange(period_count), np.diff(np.append(ki.obs_offsets, m_count)))
            )
            self.obs_w = _f64(ki.obs_w)
            self.obs_tau = _f64((obs_end_off - obs_start_off) / 360.0)
            self.obs_flow = _i64(ki.obs_flow)
            self.obs_proj_period = _i64(ki.obs_proj_curve)
        else:
            self.obs_unique_off = np.zeros(0, dtype=np.float64)
            self.obs_inv_start = np.zeros(0, dtype=np.int64)
            self.obs_inv_end = np.zeros(0, dtype=np.int64)
            self.per_obs_period = np.zeros(0, dtype=np.int64)
            self.obs_w = np.zeros(0, dtype=np.float64)
            self.obs_tau = np.zeros(0, dtype=np.float64)
            self.obs_flow = np.zeros(0, dtype=np.int64)
            self.obs_proj_period = np.zeros(0, dtype=np.int64)

        curve_x: list[np.ndarray] = []
        node_t: list[np.ndarray] = []
        node_off = [0]
        param_off = [0]
        self._node_dates: list[np.ndarray] = []
        for name in self.curve_names:
            curve = market.zero_curve(name)
            if curve.interpolation is not CurveInterpolator.LogLinearDF:
                raise NotImplementedError(
                    f"NumbaProgram supports LogLinearDF; curve '{name}' uses {curve.interpolation.name}"
                )
            dates = curve.node_dates.astype("datetime64[D]")
            if dates.size < 2:
                raise ValueError(f"curve '{name}' needs at least an origin and one pillar")
            x = offs(dates)
            curve_x.append(x)
            node_t.append(x / 365.0)
            node_off.append(node_off[-1] + dates.size)
            param_off.append(param_off[-1] + dates.size - 1)
            self._node_dates.append(dates)
        self.curve_x = _f64(np.concatenate(curve_x))
        self.node_t = _f64(np.concatenate(node_t))
        self.curve_node_offsets = _i64(node_off)
        self.curve_param_offsets = _i64(param_off)

        self._zero_reset_fixing = np.zeros(f_count, dtype=np.float64)
        self._false_reset_fixing = np.zeros(f_count, dtype=np.bool_)
        self._zero_obs_fixing = np.zeros(self.obs_w.size, dtype=np.float64)
        self._false_obs_fixing = np.zeros(self.obs_w.size, dtype=np.bool_)
        active_flow = ki.pay_dates >= self.origin
        historical_reset = (
            (ki.rate_kind == _FLOAT)
            & active_flow
            & (ki.reset_starts < self.origin)
        )
        if ki.n_obs_periods:
            flow_of_obs = ki.obs_flow[self.per_obs_period]
            historical_observation = (
                active_flow[flow_of_obs]
                & (ki.obs_starts < self.origin)
            )
        else:
            historical_observation = np.zeros(0, dtype=np.bool_)
        self._requires_historical_fixings = bool(
            historical_reset.any() or historical_observation.any()
        )

    def params_from_market(self, market) -> np.ndarray:
        """Flatten non-origin zero rates after validating the prepared curve geometry."""
        if np.datetime64(market.as_of_date.to_str(), "D") != self.origin:
            raise ValueError("market as-of date differs from the prepared NumbaProgram origin")
        params: list[np.ndarray] = []
        for name, expected_dates in zip(self.curve_names, self._node_dates):
            curve = market.zero_curve(name)
            if curve.interpolation is not CurveInterpolator.LogLinearDF:
                raise ValueError(f"curve '{name}' interpolation changed after preparation")
            dates = curve.node_dates.astype("datetime64[D]")
            if not np.array_equal(dates, expected_dates):
                raise ValueError(f"curve '{name}' pillar geometry changed after preparation")
            x = (dates[1:].astype(np.int64) - self._origin_ord) / 365.0
            params.append(-np.log(curve.node_dfs[1:]) / x)
        return _f64(np.concatenate(params))

    def _fixings_from_market(self, market):
        ki = self.inputs
        reset_mask = np.zeros(ki.n_flows, dtype=np.bool_)
        reset_value = np.zeros(ki.n_flows, dtype=np.float64)
        obs_mask = np.zeros(ki.obs_w.size, dtype=np.bool_)
        obs_value = np.zeros(ki.obs_w.size, dtype=np.float64)
        for ci, name in enumerate(self.curve_names):
            yield_curve = market.yield_curve(name)
            origin = yield_curve.zero_curve.origin
            path = yield_curve.historical_fixings
            rm = (
                (ki.proj_curve == ci)
                & (ki.rate_kind == _FLOAT)
                & (ki.pay_dates >= origin)
                & (ki.reset_starts < origin)
            )
            if rm.any():
                if path is None:
                    raise ValueError(
                        f"curve '{name}' requires historical fixings before {origin}."
                    )
                reset_mask[rm] = True
                reset_value[rm] = path.get_value(ki.reset_starts[rm])
            if ki.n_obs_periods:
                curve_of_obs = ki.obs_proj_curve[self.per_obs_period]
                flow_of_obs = ki.obs_flow[self.per_obs_period]
                om = (
                    (curve_of_obs == ci)
                    & (ki.pay_dates[flow_of_obs] >= origin)
                    & (ki.obs_starts < origin)
                )
                if om.any():
                    if path is None:
                        raise ValueError(
                            f"curve '{name}' requires historical fixings before {origin}."
                        )
                    obs_mask[om] = True
                    obs_value[om] = path.get_value(ki.obs_starts[om])
        return _b1(reset_mask), _f64(reset_value), _b1(obs_mask), _f64(obs_value)

    def _args(self, z, fixings):
        reset_mask, reset_value, obs_mask, obs_value = fixings
        return (
            self.pay_off,
            self.period_frac,
            self.sign,
            self.notional,
            self.rate_kind,
            self.fixed_rate,
            self.spread,
            self.index_floor,
            self.cap,
            self.floor,
            self.margin,
            self.discount_curve,
            self.proj_curve,
            self.reset_start_off,
            self.reset_end_off,
            self.float_tau,
            reset_mask,
            reset_value,
            self.flow_instrument,
            self.leg_offsets,
            self.leg_instrument,
            self.n_instruments,
            self.obs_w,
            self.obs_tau,
            self.per_obs_period,
            self.obs_flow,
            self.obs_proj_period,
            self.obs_inv_start,
            self.obs_inv_end,
            self.obs_unique_off,
            obs_mask,
            obs_value,
            self.curve_x,
            self.node_t,
            self.curve_node_offsets,
            self.curve_param_offsets,
            _f64(z),
        )

    @staticmethod
    def _primal(raw) -> KernelResult:
        return KernelResult(
            instrument_pv=raw[0],
            leg_pv=raw[1],
            flow_pv=raw[2],
            rate=raw[3],
            df=raw[8],
        )

    def reprice(self, market) -> KernelResult:
        z = self.params_from_market(market)
        raw = _execute(*self._args(z, self._fixings_from_market(market)), False)
        return self._primal(raw)

    def reprice_params(self, z: np.ndarray) -> KernelResult:
        """Reprice raw zero parameters (no historical-fixing overlay)."""
        if self._requires_historical_fixings:
            raise ValueError("market is required to reprice parameters for a seasoned program")
        raw = _execute(
            *self._args(
                z,
                (
                    self._false_reset_fixing,
                    self._zero_reset_fixing,
                    self._false_obs_fixing,
                    self._zero_obs_fixing,
                ),
            ),
            False,
        )
        return self._primal(raw)

    def value_and_grad(self, market) -> NumbaAdjointResult:
        z = self.params_from_market(market)
        raw = _execute(*self._args(z, self._fixings_from_market(market)), True)
        return NumbaAdjointResult(
            primal=self._primal(raw),
            cashflow_index_gradient=raw[4],
            cashflow_funding_gradient=raw[5],
            instrument_index_gradient=raw[6],
            instrument_funding_gradient=raw[7],
            curve_names=self.curve_names,
            curve_param_offsets=self.curve_param_offsets,
        )

    def value_and_grad_params(self, z: np.ndarray, market=None) -> NumbaAdjointResult:
        """Price and differentiate raw parameters, optionally retaining market fixings."""
        if market is None and self._requires_historical_fixings:
            raise ValueError("market is required to differentiate a seasoned program")
        fixings = self._fixings_from_market(market) if market is not None else (
            self._false_reset_fixing,
            self._zero_reset_fixing,
            self._false_obs_fixing,
            self._zero_obs_fixing,
        )
        raw = _execute(
            *self._args(z, fixings),
            True,
        )
        return NumbaAdjointResult(
            primal=self._primal(raw),
            cashflow_index_gradient=raw[4],
            cashflow_funding_gradient=raw[5],
            instrument_index_gradient=raw[6],
            instrument_funding_gradient=raw[7],
            curve_names=self.curve_names,
            curve_param_offsets=self.curve_param_offsets,
        )


__all__ = ["NumbaProgram", "NumbaAdjointResult"]
