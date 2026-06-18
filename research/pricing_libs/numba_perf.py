"""Forward-pricing shootout: numpy vs Numba vs JAX, on the same compiled books.

Times one forward reprice (PV per instrument) through three engines built on the *same*
``KernelInputs``, and cross-checks they agree.  The headline is the compile model:

  * **Numba** compiles ONCE (first call, any size) and the *same* machine code reprices every
    later book regardless of shape — it specialises on dtype, not array length.
  * **JAX** must compile per array *shape*, so each new book size pays a fresh XLA compile.
  * **numpy** never compiles but carries Python/vectorization overhead per call.

Run
---
    PYTHONPATH=src/ python research/pricing_libs/numba_perf.py
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
from finance.pricing.engines.jax import JaxProgram  # noqa: E402
from numba_kernel import prepare  # noqa: E402


def _clock(fn, n, warmup=1):
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


def build_market(as_of):
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
    ).calibrate(base).market, cn


def build_book(as_of, n):
    tenors = ["2Y", "3Y", "5Y", "7Y", "10Y"]
    rng = np.random.default_rng(0)
    return [Swap(notional=float(rng.choice([1, -1]) * rng.integers(10, 200) * 1e6),
                 rate_index="SOFR", fixed_rate=0.040 + 0.001 * (i % 5),
                 tenor=tenors[i % 5], as_of=as_of) for i in range(n)]


def main():
    as_of = Date(2026, 6, 1)
    market, cn = build_market(as_of)

    # one-off Numba compile, timed on the smallest book; reused for every later size.
    warm = prepare(SwapPricer().compile(build_book(as_of, 1)), market)
    t0 = time.perf_counter()
    warm.reprice()
    numba_compile = (time.perf_counter() - t0) * 1e3
    print(f"Numba one-off compile (first call, size 1): {numba_compile:8.1f} ms  "
          f"-> reused for ALL sizes below (dtype-specialised, not shape)\n")

    for n in (1, 20, 200):
        program = SwapPricer().compile(build_book(as_of, n))
        F = program.inputs.n_flows
        ni = prepare(program, market)
        jp = JaxProgram(program.inputs, market)
        disc0, proj0 = jp.params_from_market(market)
        pv_jax_fn = jax.jit(lambda d, p: jp.instrument_pv(d, p))

        # JAX: the genuine first call IS the compile — measure it BEFORE anything warms it.
        t0 = time.perf_counter(); first = pv_jax_fn(disc0, proj0); first.block_until_ready()
        jax_compile = (time.perf_counter() - t0) * 1e3
        pv_jx = np.asarray(first)

        # cross-check the three agree
        pv_np = program.reprice(market).instrument_pv
        pv_nb = ni.reprice()
        dmax = max(np.max(np.abs(pv_nb - pv_np)), np.max(np.abs(pv_jx - pv_np)))

        print(f"{'='*70}\nBOOK {n:>3} swaps  ({F} flows)   max|Δ| numba/jax vs numpy: {dmax:.2e}\n{'='*70}")
        t_np = _clock(lambda: program.reprice(market).instrument_pv, n=200)
        t_nb = _clock(lambda: ni.reprice(), n=500)
        t_jx = _clock(lambda: pv_jax_fn(disc0, proj0), n=500)
        print(f"  numpy reprice   : {t_np:9.1f} µs")
        print(f"  numba reprice   : {t_nb:9.1f} µs   (no compile here — already warm)")
        print(f"  jax   reprice   : {t_jx:9.1f} µs   (+ {jax_compile:,.0f} ms one-off compile THIS shape)")
        print(f"  numba vs numpy  : {t_np / t_nb:9.1f}x   |   numba vs jax(steady): {t_jx / t_nb:6.1f}x")

    print("\nnote: Numba paid ONE compile (size 1) and repriced 20 and 200 with zero further")
    print("compilation; JAX recompiled for each shape.  This is the dtype-vs-shape difference.")


if __name__ == "__main__":
    main()
