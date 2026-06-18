"""Performance: JAX autodiff risk vs. the numpy bump-and-reprice kernel.

Times the two engines on the *same* compiled ``PricingProgram`` over a book of swaps, so the
comparison is apples-to-apples (identical cashflows, identical curve parameterisation):

  * **PV**           — numpy ``reprice`` vs one jitted ``JaxProgram.pv``.
  * **Key-rate ladder** — numpy bump (``2·P`` repricings) vs one jitted reverse-mode jacobian
    (the whole ladder in a single pass; cost ~independent of ``P``).
  * **Gamma**        — numpy needs an ``O(P²)`` bump grid; JAX gets the full Hessian in one
    jitted pass (the bump layer ships no second-order at all).

JAX is JIT-compiled, so we report the one-off **compile** cost and the **steady-state**
per-call cost separately — the steady state is what a risk run amortises to.

Run
---
    PYTHONPATH=src/ python research/pricing_libs/jax_perf_vs_numpy.py
"""
from __future__ import annotations

import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

from finance.dates import Date  # noqa: E402
from finance.instruments.resolution import Swap, curve_name  # noqa: E402
from finance.markets.context import MarketContext  # noqa: E402
from finance.markets.curves import CurveNamespace, CurveInterpolator  # noqa: E402
from finance.pricing.calibration import (  # noqa: E402
    CurveCalibrator, CurveDefinition, GlobalSolver, deposit_helper, fra_helper, swap_helper,
)
from finance.pricing.pricers import SwapPricer  # noqa: E402
from finance.pricing.risk import Sensitivities  # noqa: E402
from finance.pricing.risk.autodiff import AutodiffRisk  # noqa: E402
from finance.pricing.risk.sensitivities import bumped_curve  # noqa: E402


def _clock(fn, n: int, warmup: int = 1) -> float:
    """Mean wall-clock per call (µs); blocks on JAX arrays; ``warmup`` calls excluded."""
    for _ in range(warmup):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
        if hasattr(r, "block_until_ready"):
            r.block_until_ready()
    return (time.perf_counter() - t0) / n * 1e6


def build_market(as_of: Date):
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
    return CurveCalibrator(
        helpers, GlobalSolver(), CurveDefinition(cn, CurveInterpolator.LogLinearDF)
    ).calibrate(base), cn


def build_book(as_of: Date, n: int) -> list:
    tenors = ["2Y", "3Y", "5Y", "7Y", "10Y"]
    rng = np.random.default_rng(0)
    book = []
    for i in range(n):
        notl = float(rng.choice([1, -1]) * rng.integers(10, 200) * 1e6)
        book.append(Swap(notional=notl, rate_index="SOFR", fixed_rate=0.040 + 0.001 * (i % 5),
                         tenor=tenors[i % len(tenors)], as_of=as_of))
    return book


def bench(n_swaps: int, as_of: Date, market, cn: str) -> None:
    program = SwapPricer().compile(build_book(as_of, n_swaps))
    P = program.inputs.n_flows
    n_pillars = int(np.count_nonzero(
        (market.discount(cn).node_dates.astype("int64") - market.discount(cn).origin.astype("int64")) > 0
    ))
    print(f"\n{'='*78}\nBOOK: {n_swaps} swaps  ({P} flows, {n_pillars} curve pillars)\n{'='*78}")

    sens = Sensitivities(program, market)
    ad = AutodiffRisk(program, market)
    base = market.discount(cn)

    # cross-check first (perf is meaningless if the numbers disagree)
    eng_krd = sens.key_rate_durations(cn).krd
    jax_krd = ad.zero_ladder(cn)
    print(f"max |Δ| jax ladder vs bump ladder : {np.max(np.abs(jax_krd - eng_krd)):.2e}")

    # -- PV ---------------------------------------------------------------------------
    t_np_pv = _clock(lambda: program.reprice(market).instrument_pv, n=200)
    pv_fn = jax.jit(lambda d, p: ad.jp.instrument_pv(d, p))
    t_jax_pv = _clock(lambda: pv_fn(ad.disc0, ad.proj0), n=500)
    print(f"\nPV            numpy reprice   : {t_np_pv:9.1f} µs")
    print(f"              jax (jitted)    : {t_jax_pv:9.1f} µs")

    # -- key-rate ladder --------------------------------------------------------------
    t_np_krd = _clock(lambda: sens.key_rate_durations(cn), n=20)
    # compile cost: time the very first call (cold), then steady state (warm)
    ad_cold = AutodiffRisk(program, market)
    t0 = time.perf_counter(); ad_cold.zero_ladder(cn); t_compile = (time.perf_counter() - t0) * 1e6
    t_jax_krd = _clock(lambda: ad.zero_ladder(cn), n=100)
    print(f"\nKRD ladder    numpy bump (2·P) : {t_np_krd:9.1f} µs   ({2 * n_pillars} repricings)")
    print(f"              jax compile     : {t_compile:9.1f} µs   (one-off)")
    print(f"              jax steady      : {t_jax_krd:9.1f} µs   (jitted jacobian, 1 pass)")
    print(f"              speedup (steady): {t_np_krd / t_jax_krd:9.1f}x")

    # -- gamma ------------------------------------------------------------------------
    def np_gamma_grid():
        # O(P^2) cross-gamma by second differences — what bump-and-reprice would need
        t = (base.node_dates.astype("int64") - base.origin.astype("int64")) / 365.0
        pil = np.nonzero(t > 0)[0]
        bp = 1e-4
        pv0 = program.reprice(market).instrument_pv
        g = np.zeros((pil.size, pil.size))
        for a in range(pil.size):
            for b in range(a, pil.size):
                cu = bumped_curve(base, bp, pillar=int(pil[a]))
                cu = bumped_curve(cu, bp, pillar=int(pil[b]))
                pvpp = program.reprice(market.with_curve(cn, cu)).instrument_pv.sum()
                g[a, b] = g[b, a] = pvpp - pv0.sum()
        return g

    t_np_gamma = _clock(np_gamma_grid, n=3)
    ad.gamma(cn)  # warm
    t_jax_gamma = _clock(lambda: ad.gamma(cn), n=50)
    print(f"\nGamma (P×P)   numpy bump grid  : {t_np_gamma:9.1f} µs   (O(P²) repricings, approx)")
    print(f"              jax hessian     : {t_jax_gamma:9.1f} µs   (1 jitted pass)")
    print(f"              speedup         : {t_np_gamma / t_jax_gamma:9.1f}x")


def main() -> None:
    as_of = Date(2026, 6, 1)
    result, cn = build_market(as_of)
    market = result.market
    for n in (1, 20, 200):
        bench(n, as_of, market, cn)
    print("\nnote: numpy wins tiny books (no XLA dispatch overhead, no compile); JAX pulls ahead")
    print("on the full ladder and on gamma as the book / pillar count grows, and is the only")
    print("engine that produces exact second-order risk at all.")


if __name__ == "__main__":
    main()
