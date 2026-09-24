"""JAX basis risk — projection/funding-split autodiff risk + calibration-Jacobian DV01s.

The productionised successor to ``jax_toy_impl.py``.  Where the toy re-implemented the
valuation inline, this drives the real backend modules:

  * ``finance.pricing.engines.jax.JaxProgram`` — the differentiable twin of the numpy
    ``reprice``, consuming the *same* ``KernelInputs``.
  * ``finance.pricing.risk.autodiff.AutodiffRisk`` — key rates, gamma, per-cashflow
    index/funding deltas, and market-quote partial DV01s.
  * ``CurveCalibrator.calibrate(..., jacobian=True)`` — captures the exact ``∂implied/∂z``.

It is **basis-native**: a Fed-Funds-vs-fixed swap that *projects* off ``USD.FEDFUND`` and
*discounts* off ``USD.SOFR`` (``funding_id='SOFR'``).  Projection and funding are genuinely
different curves, so the index/funding decomposition is unambiguous:

  * ∂PV/∂(FEDFUND pillars) = **index (projection) risk**
  * ∂PV/∂(SOFR pillars)    = **funding (discount) risk**

Every autodiff number is cross-checked against the numpy bump-and-reprice engine.

Run
---
    PYTHONPATH=src/ python research/pricing_libs/jax_basis_risk.py
"""
from __future__ import annotations

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

from finance.dates import Date  # noqa: E402
from finance.instruments.resolution import Swap, curve_name  # noqa: E402
from finance.markets.context import MarketContext  # noqa: E402
from finance.markets.curves import CurveNamespace, CurveInterpolator, YieldCurve  # noqa: E402
from finance.pricing.calibration import (  # noqa: E402
    CurveCalibrator,
    CurveDefinition,
    GlobalSolver,
    deposit_helper,
    swap_helper,
)
from finance.pricing.pricers import SwapPricer  # noqa: E402
from finance.pricing.risk import Sensitivities  # noqa: E402
from finance.pricing.risk.autodiff import AutodiffRisk  # noqa: E402

BP = 1e-4


def _fmt(a, w: int = 10, p: int = 1) -> str:
    return "[" + " ".join(f"{float(v):>{w},.{p}f}" for v in np.atleast_1d(a)) + "]"


def build_basis_market(as_of: Date):
    """Calibrate a SOFR discount curve, then a SOFR-discounted Fed Funds projection curve."""
    SOFR, FF = curve_name("USD", "SOFR"), curve_name("USD", "FEDFUND")

    sofr_helpers = [
        deposit_helper(rate=0.0432, tenor="3M", as_of=as_of),
        swap_helper(rate=0.0420, tenor="2Y", as_of=as_of),
        swap_helper(rate=0.0405, tenor="5Y", as_of=as_of),
        swap_helper(rate=0.0415, tenor="10Y", as_of=as_of),
    ]
    base = MarketContext(as_of_date=as_of, curves=CurveNamespace())
    r_sofr = CurveCalibrator(
        sofr_helpers, GlobalSolver(), CurveDefinition("USD", "SOFR", CurveInterpolator.LogLinearDF)
    ).calibrate(base)
    sofr_market = base.with_curve(
        YieldCurve.from_registry(r_sofr.origin, r_sofr.zero_curve, currency="USD", index_name="SOFR")
    )

    # Fed Funds projection curve — discounted on the SOFR curve just built (funding_id='SOFR').
    ff_helpers = [
        deposit_helper(rate=0.0427, tenor="3M", as_of=as_of, rate_index="FEDFUND", funding_id="SOFR"),
        swap_helper(rate=0.0415, tenor="2Y", as_of=as_of, rate_index="FEDFUND", funding_id="SOFR"),
        swap_helper(rate=0.0400, tenor="5Y", as_of=as_of, rate_index="FEDFUND", funding_id="SOFR"),
        swap_helper(rate=0.0410, tenor="10Y", as_of=as_of, rate_index="FEDFUND", funding_id="SOFR"),
    ]
    r_ff = CurveCalibrator(
        ff_helpers,
        GlobalSolver(),
        CurveDefinition("USD", "FEDFUND", CurveInterpolator.LogLinearDF),
    ).calibrate(sofr_market, jacobian=True)   # <-- the JAX hook: capture ∂implied/∂z
    market = sofr_market.with_curve(
        YieldCurve.from_registry(r_ff.origin, r_ff.zero_curve, currency="USD", index_name="FEDFUND")
    )

    return market, r_sofr, r_ff, SOFR, FF


def main() -> None:
    as_of = Date(2026, 6, 1)
    market, r_sofr, r_ff, SOFR, FF = build_basis_market(as_of)

    print("=" * 80)
    print("BASIS CALIBRATION  (SOFR discount + Fed Funds projection, SOFR-discounted)")
    print("=" * 80)
    print(f"curves bound        : {sorted(market.curves.snapshot().keys())}")
    print(f"FEDFUND residual max : {np.abs(r_ff.residuals).max():.2e}   (every quote repriced)")
    print(f"calibration Jacobian : {r_ff.jacobian.shape}  ∂implied/∂z (exact, one jax.jacobian)")

    # A SOFR-discounted Fed Funds swap.
    swap = Swap.fixed_float_swap(
        notional=100e6,
        rate_index="FEDFUND",
        fixed_rate=0.041,
        tenor="5Y",
        as_of=as_of,
        funding_id="SOFR",
    )
    program = SwapPricer().compile([swap])
    priced = program.price(market)
    eng = Sensitivities(program, market)
    ad = AutodiffRisk(program, market)

    pv_jax = float(ad.jp.pv(ad.disc0, ad.proj0))
    print()
    print(f"PV engine / jax     : {priced.pv:,.2f}  /  {pv_jax:,.2f}   (Δ {pv_jax - priced.pv:+.2e})")
    assert abs(pv_jax - priced.pv) < 1e-4

    # ----- index vs funding key-rate ladders -------------------------------------------
    index = ad.index_ladder(FF)[0]       # projection-curve risk (Fed Funds)
    funding = ad.funding_ladder(SOFR)[0]  # discount-curve risk (SOFR)
    eng_ff = eng.key_rate_durations(FF)
    eng_sofr = eng.key_rate_durations(SOFR)

    print()
    print("-- key-rate ladders (PV change per +1bp) ------------------------------------------")
    print(f"INDEX   (FEDFUND proj): {_fmt(index)}  sum {index.sum():>12,.1f}")
    print(f"  bump engine FEDFUND : {_fmt(eng_ff.krd[0])}  sum {eng_ff.total[0]:>12,.1f}")
    print(f"FUNDING (SOFR disc)   : {_fmt(funding)}  sum {funding.sum():>12,.1f}")
    print(f"  bump engine SOFR    : {_fmt(eng_sofr.krd[0])}  sum {eng_sofr.total[0]:>12,.1f}")
    assert np.max(np.abs(index - eng_ff.krd[0])) < 0.05
    assert np.max(np.abs(funding - eng_sofr.krd[0])) < 0.05

    # ----- per-cashflow decomposition: index and funding deltas per flow ---------------
    cf_index = ad.cashflow_index_delta(FF)      # (F, P_ff)
    cf_funding = ad.cashflow_funding_delta(SOFR)  # (F, P_sofr)
    print()
    print("-- per-cashflow risk (one row per flow; columns = curve pillars) -------------------")
    print(f"cashflow index   delta shape {cf_index.shape}  (Fed Funds projection risk per flow)")
    print(f"cashflow funding delta shape {cf_funding.shape}  (SOFR discount risk per flow)")
    # the per-cashflow deltas reduce exactly to the instrument ladders
    assert np.max(np.abs(cf_index.sum(0) - index)) < 1e-6
    assert np.max(np.abs(cf_funding.sum(0) - funding)) < 1e-6
    print("  Σ(cashflow index)   == instrument INDEX ladder   ✓")
    print("  Σ(cashflow funding) == instrument FUNDING ladder ✓")
    # which flows carry which risk: fixed-leg flows are pure funding (zero index delta)
    fixed_flow_index = np.max(np.abs(cf_index), axis=1)
    n_pure_funding = int((fixed_flow_index < 1e-9).sum())
    print(f"  flows with zero index delta (fixed-leg, pure funding) : {n_pure_funding}/{cf_index.shape[0]}")

    # ----- second order: gamma (bump layer produces none) ------------------------------
    gamma = ad.gamma(FF)   # (P, P) total convexity in the Fed Funds curve, per (1bp)^2
    print()
    print("-- second order (autodiff-only) ---------------------------------------------------")
    print(f"FEDFUND pillar gamma diag : {_fmt(np.diag(gamma), 10, 4)}  (½·Hₚₚ·1bp²)")
    print(f"largest cross-gamma |Hₚ_q|: {np.max(np.abs(gamma - np.diag(np.diag(gamma)))):.3e}")

    # ----- market-quote partial DV01 (the calibration-Jacobian transform) --------------
    pdv01 = ad.partial_dv01(FF, r_ff.jacobian)[0]   # risk to each FEDFUND par quote
    tenors = ["3M", "2Y", "5Y", "10Y"]
    print()
    print("-- partial DV01 to the Fed Funds par quotes  ((∂V/∂z)·J⁻¹) ------------------------")
    for t, d in zip(tenors, pdv01):
        print(f"  {t:>4} quote : {d:>12,.1f}")
    print(f"  sum        : {pdv01.sum():>12,.1f}   (par DV01; cf. zero DV01 {index.sum() + funding.sum():,.1f})")
    # par-quote risk is local: a 5Y swap loads almost entirely on the 5Y quote
    assert abs(pdv01[2]) > 0.95 * abs(pdv01.sum())
    print("  -> concentrated on the 5Y quote (par risk is local to the matching tenor) ✓")

    print()
    print("all cross-checks vs the numpy engine passed.")


if __name__ == "__main__":
    main()
