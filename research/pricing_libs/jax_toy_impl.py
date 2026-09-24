"""JAX toy implementation — autodiff risk vs. the standard bump-and-reprice engine.

What this is
------------
A side-by-side comparison, on a real calibrated USD SOFR curve, of:

  1. **The standard path** — exactly the §9 worked example in
     ``docs/pricing_architecture.md``: calibrate a curve from a quote strip, bind it into a
     ``MarketContext``, compile a swap once with ``SwapPricer``, read PV + per-leg PV, and pull
     risk (parallel DV01 + a key-rate ladder) from the ``Sensitivities`` bump-and-reprice layer.

  2. **A JAX path** — the *same* compiled cashflows and the *same* curve parameterisation
     (continuously-compounded zero rates at the pillars, ``DF = exp(-z·t)``, log-linear in
     ``ln DF`` — identical to ``CurveInterpolator.LogLinearDF``), but priced through a pure,
     differentiable ``jax.numpy`` valuation. Then:

       * ``value``           -> PV                       (cross-checked vs the engine)
       * ``jax.grad``        -> the WHOLE key-rate ladder (∂PV/∂zₚ) in ONE reverse-mode pass,
                                vs. the engine's 2·P repricings  (cross-checked vs KRD)
       * ``jax.hessian``     -> the full pillar gamma matrix ∂²PV/∂zₚ∂z_q — second-order
                                cross-convexity that the bump layer does not produce at all
                                (cross-checked vs a parallel second difference)
       * ``jax.jacobian``    -> a per-instrument KRD matrix for a book, shape (n_inst, P),
                                matching ``KeyRateLadder.krd``

Why it matters
--------------
The engine's risk is *bump-and-reprice*: a P-pillar ladder costs 2·P full repricings, and
second-order risk needs an O(P²) bump grid. Autodiff gives the entire first-order ladder from
ONE backward pass and the entire second-order gamma matrix from one Hessian — exact (no bump
noise), and the cost is independent of P. This is the toy that shows the engine's columnar
``KernelInputs`` is already in the right shape to hand to an AD backend.

Scope (kept deliberately small)
-------------------------------
* Vanilla USD SOFR OIS (fixed vs. geometric-compounded), no spread / cap / floor — so the
  daily-compounded float coupon telescopes *exactly* to ``DF(start)/DF(end) - 1`` and the JAX
  leg reproduces the engine's compounded kernel without re-implementing the daily grid.
* All flows sit interior to the curve's last pillar, so flat-forward extrapolation never
  triggers (``jnp.interp`` clamps; the engine extrapolates — equal here, noted not equal in
  general).

Run
---
    PYTHONPATH=src/ python research/pricing_libs/jax_toy_impl.py
"""
from __future__ import annotations

import time

import numpy as np

import jax

# Curve math and discounting are in float64; rates risk is unforgiving of float32 noise.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402  (must follow the x64 config flip)

from finance.dates import Date  # noqa: E402
from finance.instruments.resolution import Swap, curve_name  # noqa: E402
from finance.markets.context import MarketContext  # noqa: E402
from finance.markets.curves import (  # noqa: E402
    CurveNamespace,
    CurveInterpolator,
    YieldCurve,
    ZeroCurve,
)
from finance.pricing.calibration import (  # noqa: E402
    CurveCalibrator,
    CurveDefinition,
    GlobalSolver,
    deposit_helper,
    fra_helper,
    swap_helper,
)
from finance.pricing.pricers import SwapPricer  # noqa: E402
from finance.pricing.pricers.base import PricingProgram  # noqa: E402
from finance.pricing.risk import Sensitivities  # noqa: E402
from finance.pricing.types import RateKind  # noqa: E402

BP = 1e-4  # one basis point, in continuously-compounded-rate units


# ---------------------------------------------------------------------------
# 1. Market — the §9 calibration, verbatim.
# ---------------------------------------------------------------------------
def build_market(as_of: Date) -> tuple[MarketContext, ZeroCurve, str]:
    """Calibrate a USD SOFR curve from a deposit/FRA/swap strip and bind it into a market."""
    cn = curve_name("USD", "SOFR")
    helpers = [
        deposit_helper(rate=0.0430, tenor="1M", as_of=as_of),
        deposit_helper(rate=0.0432, tenor="3M", as_of=as_of),
        deposit_helper(rate=0.0435, tenor="6M", as_of=as_of),
        fra_helper(rate=0.0440, start="6M", end="12M", as_of=as_of),
        swap_helper(rate=0.0420, tenor="2Y", as_of=as_of),
        swap_helper(rate=0.0410, tenor="3Y", as_of=as_of),
        swap_helper(rate=0.0405, tenor="5Y", as_of=as_of),
        swap_helper(rate=0.0415, tenor="10Y", as_of=as_of),
    ]
    base = MarketContext(as_of_date=as_of, curves=CurveNamespace())
    target = CurveDefinition("USD", "SOFR", CurveInterpolator.LogLinearDF)
    result = CurveCalibrator(helpers, GlobalSolver(), target).calibrate(base)
    assert result.solver_result.converged, "calibration failed"
    market = base.with_curve(
        YieldCurve.from_registry(result.origin, result.zero_curve, currency="USD", index_name="SOFR")
    )
    return market, result.zero_curve, cn


# ---------------------------------------------------------------------------
# 2. The compiled cashflows, lifted out of KernelInputs into plain arrays.
# ---------------------------------------------------------------------------
class CompiledFlows:
    """The columnar facts the JAX repricer needs, pulled once from a PricingProgram.

    This is the whole point of the columnar ``KernelInputs``: the same struct-of-arrays the
    numpy engine reduces is exactly what an AD backend consumes. We read the day-offsets
    (relative to the curve origin) and per-flow coupon data and never touch dates again.
    """

    def __init__(self, program: PricingProgram, origin: np.datetime64):
        ki = program.inputs
        o = origin.astype(np.int64)

        def offs(dates: np.ndarray) -> np.ndarray:
            return (dates.astype(np.int64) - o).astype(np.float64)

        # flow -> instrument (mirrors PricingProgram.price's leg_of_flow construction)
        leg_of_flow = np.repeat(
            np.arange(ki.n_legs), np.diff(np.append(ki.leg_offsets, ki.n_flows))
        )
        self.flow_instrument = jnp.asarray(ki.leg_instrument[leg_of_flow])
        self.n_instruments = ki.n_instruments

        # per-flow columns (day offsets + coupon data) as JAX arrays
        self.pay = jnp.asarray(offs(ki.pay_dates))
        self.acc_start = jnp.asarray(offs(ki.reset_starts))
        self.acc_end = jnp.asarray(offs(ki.reset_ends))
        self.period_frac = jnp.asarray(ki.period_frac)
        self.notional = jnp.asarray(ki.notional)
        self.sign = jnp.asarray(ki.sign)
        self.fixed_rate = jnp.asarray(ki.fixed_rate)
        self.is_compounded = jnp.asarray(ki.rate_kind == int(RateKind.Compounded))

        # vanilla-only guard: this toy models Fixed + Compounded with no spread/cap/floor
        kinds = set(np.unique(ki.rate_kind).tolist())
        assert kinds <= {int(RateKind.Fixed), int(RateKind.Compounded)}, (
            f"toy supports Fixed/Compounded only; got rate kinds {kinds}"
        )
        assert not np.any(ki.spread) and np.all(np.isinf(ki.cap)) and np.all(np.isinf(ki.floor)), (
            "toy assumes no spread/cap/floor (so the float coupon telescopes exactly)"
        )


# ---------------------------------------------------------------------------
# 3. The differentiable curve + valuation (pure jax.numpy).
# ---------------------------------------------------------------------------
def zero_rates_from_curve(curve: ZeroCurve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a ZeroCurve into the (origin-inclusive day grid, pillar years, pillar zeros).

    The parameter vector ``z`` is the continuously-compounded zero rate at each non-origin
    pillar — *exactly* what ``Sensitivities.bumped_curve`` shifts, so JAX gradients in ``z``
    are directly comparable to the engine's key-rate ladder.
    """
    x_nodes = curve.x  # incl. origin (0.0)
    t = x_nodes[1:] / 365.0  # Act/365 year fraction at each non-origin pillar
    return x_nodes, t, curve.node_zero_rates


def make_df(x_nodes: np.ndarray, t_nodes: np.ndarray):
    """Build DF(x; z): log-linear in ln(DF), matching CurveInterpolator.LogLinearDF.

    Returns a closure ``df(z, x_query)`` differentiable in the pillar zero-rates ``z``.
    """
    x_nodes = jnp.asarray(x_nodes)
    t_nodes = jnp.asarray(t_nodes)

    def df(z, x_query):
        # ln DF at the nodes: origin pinned at 0, pillars at -zₚ·tₚ
        logdf_nodes = jnp.concatenate([jnp.zeros(1), -z * t_nodes])
        return jnp.exp(jnp.interp(x_query, x_nodes, logdf_nodes))

    return df


def make_pricer(flows: CompiledFlows, df):
    """Returns ``flow_pv(z) -> (F,)``: per-flow PVs as a differentiable function of ``z``.

    Fixed coupon:        cash = N · c · τ
    Compounded (OIS):    the daily-compounded SOFR over [start, end] telescopes to
                         ∏(1 + rᵢτᵢ) - 1 = DF(start)/DF(end) - 1, so cash = N · (DF_s/DF_e - 1).
    flow_pv = cash · DF(pay) · sign.
    """

    def flow_pv(z):
        df_pay = df(z, flows.pay)
        df_s = df(z, flows.acc_start)
        df_e = df(z, flows.acc_end)
        cash_fixed = flows.notional * flows.fixed_rate * flows.period_frac
        cash_float = flows.notional * (df_s / df_e - 1.0)
        cash = jnp.where(flows.is_compounded, cash_float, cash_fixed)
        return cash * df_pay * flows.sign

    return flow_pv


def make_instrument_pv(flows: CompiledFlows, flow_pv):
    """Roll per-flow PVs up to per-instrument PVs (segment sum), differentiable in ``z``."""

    def instrument_pv(z):
        contrib = flow_pv(z)
        return jnp.zeros(flows.n_instruments).at[flows.flow_instrument].add(contrib)

    return instrument_pv


# ---------------------------------------------------------------------------
# 4. The demo.
# ---------------------------------------------------------------------------
def _fmt(a: np.ndarray, w: int = 11, p: int = 2) -> str:
    return "[" + " ".join(f"{float(v):>{w},.{p}f}" for v in np.atleast_1d(a)) + "]"


def main() -> None:
    as_of = Date(2026, 6, 1)
    market, curve, cn = build_market(as_of)

    swap = Swap.fixed_float_swap(
        notional=100e6,
        rate_index="SOFR",
        fixed_rate=0.041,
        tenor="5Y",
        as_of=as_of,
    )
    program = SwapPricer().compile([swap])  # compile once

    # ----- STANDARD PATH ---------------------------------------------------
    print("=" * 78)
    print("STANDARD ENGINE  (compile-once / reprice-many + bump-and-reprice risk)")
    print("=" * 78)
    priced = program.price(market)
    sens = Sensitivities(program, market)
    eng_dv01 = sens.dv01(cn)
    eng_krd = sens.key_rate_durations(cn)
    print(f"PV                : {priced.pv:>15,.2f}")
    print(f"leg PV            : {_fmt(priced.leg_pv, 16)}   (fixed, float)")
    print(f"pillars (yrs)     : {_fmt(eng_krd.pillar_years, 11, 2)}")
    print(f"parallel DV01     : {_fmt(eng_dv01, 16)}   (PV change per +1bp)")
    print(f"key-rate ladder   : {_fmt(eng_krd.krd[0], 11, 2)}")
    print(f"  ladder sum      : {float(eng_krd.total[0]):>15,.2f}   (== DV01 by additivity)")

    # ----- JAX PATH --------------------------------------------------------
    x_nodes, t_nodes, z0 = zero_rates_from_curve(curve)
    flows = CompiledFlows(program, market.yield_curve(cn).origin)
    df = make_df(x_nodes, t_nodes)
    flow_pv = make_pricer(flows, df)

    pv_fn = jax.jit(lambda z: jnp.sum(flow_pv(z)))      # scalar PV(z)
    grad_fn = jax.jit(jax.grad(pv_fn))                  # ∂PV/∂z  (the whole ladder, 1 pass)
    hess_fn = jax.jit(jax.hessian(pv_fn))               # ∂²PV/∂z∂z (full gamma matrix)

    z = jnp.asarray(z0)
    pv_jax = float(pv_fn(z))
    grad = np.asarray(grad_fn(z))      # ∂PV per unit zero-rate move at each pillar
    hess = np.asarray(hess_fn(z))      # (P, P)

    # grad is ∂PV per unit rate; ·1bp -> PV change per +1bp == the key-rate ladder.
    jax_krd = grad * BP
    jax_dv01 = float(grad.sum() * BP)  # parallel = sum of pillar sensitivities

    print()
    print("=" * 78)
    print("JAX AUTODIFF  (same curve, same cashflows — risk by differentiation)")
    print("=" * 78)
    print(f"PV                : {pv_jax:>15,.2f}   (Δ vs engine: {pv_jax - priced.pv:+.2e})")
    print(f"parallel DV01     : {jax_dv01:>15,.2f}   (grad·1bp; Δ vs engine: "
          f"{jax_dv01 - float(eng_dv01[0]):+.3e})")
    print(f"key-rate ladder   : {_fmt(jax_krd, 11, 2)}   (∂PV/∂zₚ·1bp, ONE backward pass)")
    print(f"  max |Δ| vs engine ladder : {np.max(np.abs(jax_krd - eng_krd.krd[0])):.3e}")

    # ----- SECOND ORDER (gamma / convexity) — autodiff-only --------------
    # diagonal gammaₚ scaled to a 1bp move: ½·Hₚₚ·(1bp)²  (per-pillar convexity P&L)
    gamma_1bp = 0.5 * np.diag(hess) * BP**2
    # parallel convexity: ½·(1bp)²·Σ Hₚ_q ; validate against a full parallel 2nd difference.
    parallel_gamma_jax = 0.5 * hess.sum() * BP**2
    yield_curve = market.yield_curve(cn)
    up = program.reprice(market.with_curve(yield_curve.with_zero_curve(_bump(curve, +BP)))).instrument_pv[0]
    dn = program.reprice(market.with_curve(yield_curve.with_zero_curve(_bump(curve, -BP)))).instrument_pv[0]
    parallel_gamma_engine = 0.5 * (up + dn - 2.0 * priced.pv)

    print()
    print("-- second order (gamma / convexity) — NOT produced by the bump layer ----------")
    print(f"per-pillar gamma  : {_fmt(gamma_1bp, 11, 4)}   (½·Hₚₚ·1bp², PV curvature/pillar)")
    print(f"parallel convexity: {parallel_gamma_jax:>13,.4f}   (½·ΣH·1bp²)")
    print(f"  engine 2nd diff : {parallel_gamma_engine:>13,.4f}   (½·[PV₊+PV₋−2·PV₀]; "
          f"Δ {parallel_gamma_jax - parallel_gamma_engine:+.2e})")
    print(f"  largest off-diagonal cross-gamma |Hₚ_q|·1bp² : "
          f"{np.max(np.abs(hess - np.diag(np.diag(hess)))) * BP**2:.4e}")

    # ----- A BOOK: per-instrument KRD matrix via one Jacobian ------------
    book = [
        Swap.fixed_float_swap(
            notional=100e6,
            rate_index="SOFR",
            fixed_rate=0.041,
            tenor="5Y",
            as_of=as_of,
        ),
        Swap.fixed_float_swap(
            notional=-25e6,
            rate_index="SOFR",
            fixed_rate=0.040,
            tenor="10Y",
            as_of=as_of,
        ),
    ]
    book_prog = SwapPricer().compile(book)
    book_flows = CompiledFlows(book_prog, market.yield_curve(cn).origin)
    book_inst_pv = make_instrument_pv(book_flows, make_pricer(book_flows, df))
    jac = np.asarray(jax.jit(jax.jacobian(book_inst_pv))(z)) * BP  # (n_inst, P) KRD matrix

    book_priced = book_prog.price(market)
    book_krd_engine = Sensitivities(book_prog, market).key_rate_durations(cn).krd

    print()
    print("=" * 78)
    print("BOOK  (2 swaps) — per-instrument KRD matrix from ONE jax.jacobian")
    print("=" * 78)
    print(f"instrument PV     : {_fmt(book_priced.instrument_pv, 16)}")
    for i in range(jac.shape[0]):
        print(f"  inst {i} KRD (jax): {_fmt(jac[i], 11, 2)}")
        print(f"  inst {i} KRD (eng): {_fmt(book_krd_engine[i], 11, 2)}")
    print(f"  max |Δ| jax vs engine    : {np.max(np.abs(jac - book_krd_engine)):.3e}")

    # ----- A note on cost --------------------------------------------------
    P = z0.size
    print()
    print("-- cost note ------------------------------------------------------------------")
    print(f"engine ladder      : 2·P = {2 * P} full repricings for the {P}-pillar ladder")
    print("jax grad           : 1 reverse-mode pass for the whole ladder, ∀P")
    _timings(pv_fn, grad_fn, hess_fn, z, sens, cn)


def _bump(curve: ZeroCurve, bp: float) -> ZeroCurve:
    """Parallel zero-rate shift — same parameterisation Sensitivities.bumped_curve uses."""
    from finance.pricing.risk import bumped_curve

    return bumped_curve(curve, bp)


def _timings(pv_fn, grad_fn, hess_fn, z, sens, cn) -> None:
    """Rough wall-clock — JAX is JIT-compiled, so we warm up first, then time steady state."""
    pv_fn(z).block_until_ready()
    grad_fn(z).block_until_ready()
    hess_fn(z).block_until_ready()

    def clock(fn, n=200):
        t0 = time.perf_counter()
        for _ in range(n):
            r = fn()
            if hasattr(r, "block_until_ready"):
                r.block_until_ready()
        return (time.perf_counter() - t0) / n * 1e6  # microseconds

    t_grad = clock(lambda: grad_fn(z))
    t_hess = clock(lambda: hess_fn(z))
    t_eng = clock(lambda: sens.key_rate_durations(cn), n=50)
    print(f"jax grad  (full ladder)  : {t_grad:8.1f} µs / call")
    print(f"jax hessian (gamma)      : {t_hess:8.1f} µs / call")
    print(f"engine bump ladder       : {t_eng:8.1f} µs / call")


if __name__ == "__main__":
    main()
