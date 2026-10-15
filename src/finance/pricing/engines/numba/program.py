"""Shape-polymorphic Numba pricing program with a hand-written reverse adjoint.

The kernel consumes the same columnar :class:`KernelInputs` as the NumPy and JAX engines,
but coupon projection, shaping, discounting, and reduction all stay in one nopython call.
Its only differentiable inputs are flattened continuously-compounded pillar zero rates.

Curve interpolation never runs inside the kernel.  Every discount-factor query (payment
dates, reset windows, observation endpoints) is a static day offset once the program is
compiled, so at prepare time each curve hands over ``ZeroCurve.log_df_affine`` on its unique
queries: ``ln DF(q) = const[q] + Σ coef[q, k] · z[k]``, stored as CSR rows.  The primal is one
dot product per query and the adjoint scatters through the same rows, whatever the scheme —
log-linear gives two nonzeros per row, a cubic spline a dense one.

Numba specializes the function on dtype and dimensionality, not array length.  Consequently
one compiled signature prices a singleton, a changing futures strip, or a large swap book
without the shape-triggered recompilations of a traced JAX program.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from finance.instruments.enums import MarginTreatment
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
def _logdf(q, q_const, q_ptr, q_param, q_coef, z):
    """``ln DF`` of query ``q``: its affine row over the flattened zero rates."""
    acc = q_const[q]
    for j in range(q_ptr[q], q_ptr[q + 1]):
        acc += q_coef[j] * z[q_param[j]]
    return acc


@njit(cache=True, inline="always")  # pragma: no cover - executed as Numba machine code
def _scatter(row, q, scale, q_ptr, q_param, q_coef):
    """Accumulate ``scale · d ln DF(q) / dz`` into a gradient row."""
    for j in range(q_ptr[q], q_ptr[q + 1]):
        row[q_param[j]] += scale * q_coef[j]


@njit(cache=True)  # pragma: no cover - executed as Numba machine code
def _execute(
    # flow columns
    pay_q,
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
    reset_s_q,
    reset_e_q,
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
    obs_s_q,
    obs_e_q,
    obs_is_fixed,
    obs_fixing,
    # discount-factor queries as affine rows over the parameters
    q_const,
    q_ptr,
    q_param,
    q_coef,
    z,
    calculate_adjoint,
):
    f_count = pay_q.size
    m_count = obs_w.size
    period_count = obs_flow.size
    q_count = z.size

    # Every discount factor the book needs, once per call.  A query before the origin
    # carries const = -inf, so its DF is exactly zero: the expired-flow marker.
    df_q = np.empty(q_const.size)
    for q in range(q_const.size):
        df_q[q] = np.exp(_logdf(q, q_const, q_ptr, q_param, q_coef, z))

    rate = np.zeros(f_count)
    df = np.empty(f_count)
    for f in range(f_count):
        df[f] = df_q[pay_q[f]]

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
            if reset_is_fixed[f]:
                projected = reset_fixing[f]
                active = 0.0
            else:
                q_ratio = df_q[reset_s_q[f]] / df_q[reset_e_q[f]]
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
                _scatter(index_grad[f], reset_s_q[f], scale, q_ptr, q_param, q_coef)
                _scatter(index_grad[f], reset_e_q[f], -scale, q_ptr, q_param, q_coef)

    # Compounded/averaged observation grids.
    if period_count > 0:
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
                projected = (df_q[obs_s_q[m]] / df_q[obs_e_q[m]] - 1.0) / obs_tau[m]
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
                q_ratio = df_q[obs_s_q[m]] / df_q[obs_e_q[m]]
                if rate_kind[flow] == _COMPOUNDED:
                    drate_dobs = growth[p] * obs_w[m] / (
                        sum_weight[p] * (1.0 + obs_rate[m] * obs_w[m])
                    )
                else:
                    drate_dobs = obs_w[m] / sum_weight[p]
                pv_per_rate = notional[flow] * period_frac[flow] * df[flow] * sign[flow]
                scale = pv_per_rate * drate_dobs * q_ratio / obs_tau[m]
                _scatter(index_grad[flow], obs_s_q[m], scale, q_ptr, q_param, q_coef)
                _scatter(index_grad[flow], obs_e_q[m], -scale, q_ptr, q_param, q_coef)

    # Cashflow PV and the discount/funding half of the adjoint.
    flow_pv = np.empty(f_count)
    for f in range(f_count):
        flow_pv[f] = notional[f] * rate[f] * period_frac[f] * df[f] * sign[f]
        if calculate_adjoint:
            _scatter(funding_grad[f], pay_q[f], flow_pv[f], q_ptr, q_param, q_coef)

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
    curve_param_slices: tuple[slice, ...]

    def curve_slice(self, curve_name: str) -> slice:
        """Columns of ``curve_name``'s parameter block; keys sharing one curve object share the block."""
        try:
            ci = self.curve_names.index(curve_name)
        except ValueError:
            raise KeyError(f"program has no curve named '{curve_name}'") from None
        return self.curve_param_slices[ci]

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


@dataclass(frozen=True)
class _QueryTable:
    """Every discount-factor query of the book as CSR affine rows over the flattened zero rates.

    ``const[q] = -inf`` with an empty row marks a query before the origin (an expired flow).
    """

    const: np.ndarray   # (NQ,)
    ptr: np.ndarray     # (NQ + 1,)
    param: np.ndarray   # (NNZ,) flattened parameter index
    coef: np.ndarray    # (NNZ,)


class NumbaProgram:
    """Prepared pricing geometry reusable across arbitrary zero-rate values of one shape.

    The cashflow arrays may have any lengths when a new ``NumbaProgram`` is created; they all
    call the same Numba dispatcher signature.  Within a prepared program, ``reprice`` accepts
    any market with identical curve pillar geometry and interpolation scheme, which is the
    normal scenario/risk path.
    """

    def __init__(self, inputs: KernelInputs, market):
        self.inputs = inputs
        self.curve_names = tuple(inputs.curve_names)
        if not self.curve_names:
            raise ValueError("NumbaProgram requires at least one curve")

        origins = [np.datetime64(market.yield_curve(name).origin, "D") for name in self.curve_names]
        if any(origin != origins[0] for origin in origins[1:]):
            raise ValueError("all curves in a NumbaProgram must share the same origin")
        self.origin = origins[0]
        self._origin_ord = self.origin.astype(np.int64)

        # -- parameter blocks: one per distinct discount-curve object -----------------------
        # Two keys bound to one curve (``USD.SOFR.ON`` and ``USD.STDCSA`` for SOFR discounting)
        # share a block: a move of the curve is one move of its parameters and both roles'
        # adjoints land in the same columns.  The block is named after the owning key.
        self.curve_owners = tuple(market.curve_owner(name) for name in self.curve_names)
        self.block_names = tuple(dict.fromkeys(self.curve_owners))
        block_sizes = [market.zero_curve(name).n_params for name in self.block_names]
        block_start = np.concatenate([[0], np.cumsum(block_sizes)]).astype(np.int64)
        block_slices = {
            name: slice(int(block_start[b]), int(block_start[b + 1])) for b, name in enumerate(self.block_names)
        }
        self.curve_param_slices = tuple(block_slices[owner] for owner in self.curve_owners)
        self.n_params = int(block_start[-1])

        def offs(dates):
            return _f64(dates.astype(np.int64) - self._origin_ord)

        ki = inputs
        f_count = ki.n_flows
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
        pay_off = offs(ki.pay_dates)
        reset_start_off = offs(ki.reset_starts) if ki.reset_starts.size else np.zeros(f_count)
        reset_end_off = offs(ki.reset_ends) if ki.reset_ends.size else np.zeros(f_count)
        is_float = self.rate_kind == _FLOAT
        self.float_tau = _f64(np.where(is_float, (reset_end_off - reset_start_off) / 360.0, 1.0))

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
            self.per_obs_period = _i64(
                np.repeat(np.arange(period_count), np.diff(np.append(ki.obs_offsets, m_count)))
            )
            self.obs_w = _f64(ki.obs_w)
            self.obs_tau = _f64((obs_end_off - obs_start_off) / 360.0)
            self.obs_flow = _i64(ki.obs_flow)
            obs_curve = ki.obs_proj_curve[self.per_obs_period]
        else:
            obs_start_off = obs_end_off = np.zeros(0, dtype=np.float64)
            self.per_obs_period = np.zeros(0, dtype=np.int64)
            self.obs_w = np.zeros(0, dtype=np.float64)
            self.obs_tau = np.zeros(0, dtype=np.float64)
            self.obs_flow = np.zeros(0, dtype=np.int64)
            obs_curve = np.zeros(0, dtype=np.int64)

        # -- discount-factor queries: dedup per curve, then one affine row each ------------
        self.pay_q = np.full(f_count, -1, dtype=np.int64)
        self.reset_s_q = np.full(f_count, -1, dtype=np.int64)
        self.reset_e_q = np.full(f_count, -1, dtype=np.int64)
        self.obs_s_q = np.full(obs_start_off.size, -1, dtype=np.int64)
        self.obs_e_q = np.full(obs_end_off.size, -1, dtype=np.int64)
        consts, ptrs, params, coefs = [], [], [], []
        q_off = 0
        self._node_dates: list[np.ndarray] = []
        self._schemes: list[tuple] = []
        for ci, name in enumerate(self.curve_names):
            yield_curve = market.yield_curve(name)
            zero_curve = yield_curve.zero_curve
            dates = yield_curve.node_dates
            if dates.size < 2:
                raise ValueError(f"curve '{name}' needs at least an origin and one pillar")
            self._node_dates.append(dates)
            self._schemes.append(zero_curve.scheme)

            pay_m = ki.discount_curve == ci
            reset_m = is_float & (ki.proj_curve == ci)
            obs_m = obs_curve == ci
            targets = (
                (self.pay_q, pay_m, pay_off),
                (self.reset_s_q, reset_m, reset_start_off),
                (self.reset_e_q, reset_m, reset_end_off),
                (self.obs_s_q, obs_m, obs_start_off),
                (self.obs_e_q, obs_m, obs_end_off),
            )
            unique, inverse = np.unique(np.concatenate([x[m] for _, m, x in targets]), return_inverse=True)
            cursor = 0
            for out, m, _ in targets:
                n = int(m.sum())
                out[m] = q_off + inverse[cursor : cursor + n]
                cursor += n

            live = unique >= 0.0
            const = np.full(unique.size, -np.inf)
            jac = np.zeros((unique.size, zero_curve.n_params))
            if live.any():
                const[live], jac[live] = zero_curve.log_df_affine(unique[live])
            nonzero = jac != 0.0
            consts.append(const)
            ptrs.append(np.count_nonzero(nonzero, axis=1))
            params.append(np.nonzero(nonzero)[1] + self.curve_param_slices[ci].start)
            coefs.append(jac[nonzero])
            q_off += unique.size

        counts = np.concatenate(ptrs) if ptrs else np.zeros(0, dtype=np.int64)
        self.queries = _QueryTable(
            const=_f64(np.concatenate(consts)),
            ptr=_i64(np.concatenate([[0], np.cumsum(counts)])),
            param=_i64(np.concatenate(params)),
            coef=_f64(np.concatenate(coefs)),
        )

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

    def compatible(self, market) -> bool:
        """Whether ``market`` has the origin, pillar geometry and schemes this program was prepared on."""
        if np.datetime64(market.as_of_date.to_str(), "D") != self.origin:
            return False
        for name, owner, expected_dates, scheme in zip(
            self.curve_names, self.curve_owners, self._node_dates, self._schemes
        ):
            try:
                yield_curve = market.yield_curve(name)
                shared = market.zero_curve(owner) is yield_curve.zero_curve
            except KeyError:
                return False
            if not shared or yield_curve.zero_curve.scheme != scheme:
                return False
            if not np.array_equal(yield_curve.node_dates, expected_dates):
                return False
        return True

    def params_from_market(self, market) -> np.ndarray:
        """Concatenate every parameter block's ``DiscountCurve.params`` after validating the prepared geometry.

        Blocks are laid out per distinct curve object (see ``block_names``), not per key."""
        if not self.compatible(market):
            raise ValueError(
                "market origin, pillar geometry or interpolation differs from the prepared NumbaProgram"
            )
        return _f64(np.concatenate([market.zero_curve(name).params for name in self.block_names]))

    def _fixings_from_market(self, market):
        ki = self.inputs
        reset_mask = np.zeros(ki.n_flows, dtype=np.bool_)
        reset_value = np.zeros(ki.n_flows, dtype=np.float64)
        obs_mask = np.zeros(ki.obs_w.size, dtype=np.bool_)
        obs_value = np.zeros(ki.obs_w.size, dtype=np.float64)
        for ci, name in enumerate(self.curve_names):
            yield_curve = market.yield_curve(name)
            origin = yield_curve.origin
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
            self.pay_q,
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
            self.reset_s_q,
            self.reset_e_q,
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
            self.obs_s_q,
            self.obs_e_q,
            obs_mask,
            obs_value,
            self.queries.const,
            self.queries.ptr,
            self.queries.param,
            self.queries.coef,
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
            curve_param_slices=self.curve_param_slices,
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
            curve_param_slices=self.curve_param_slices,
        )

