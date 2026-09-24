"""Anatomy of the JAX one-off compile cost.

The "~1.3 s compile" on a big book is not one thing — it is three stages JAX runs the first
time a jitted function meets a given input *signature* (shapes + dtypes + static closure):

  1. **trace**   — run the Python function once on abstract values, recording a jaxpr.  This
                   executes our ``for curve in curve_names`` loops, ``jnp.where`` routing,
                   ``jnp.interp`` (searchsorted), ``segment_sum``, scatters — once, symbolically.
  2. **lower**   — jaxpr -> StableHLO (the XLA input IR).
  3. **compile** — XLA compiles HLO to a CPU executable: fusion, layout/buffer assignment,
                   LLVM codegen.  This is the expensive stage and it grows with graph size.

For ``jacfwd`` the traced graph is ~P forward-mode copies of the whole valuation; for the
Hessian it is bigger still — so compile dominates.  The executable is then **cached** keyed on
that signature, so every later call (and any same-shaped book) is a pure execute.  A
*different-shaped* book is a new signature -> recompile (the tax to plan around).

This script times trace+lower vs compile separately, at two book sizes, and shows the cache
hit on a second same-shape call.

Run
---
    PYTHONPATH=src/ python research/pricing_libs/jax_compile_anatomy.py
"""
from __future__ import annotations

import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

from finance.dates import Date  # noqa: E402
from finance.instruments.resolution import Swap, curve_name  # noqa: E402
from finance.markets.context import MarketContext  # noqa: E402
from finance.markets.curves import CurveNamespace, CurveInterpolator, YieldCurve  # noqa: E402
from finance.pricing.calibration import (  # noqa: E402
    CurveCalibrator, CurveDefinition, GlobalSolver, deposit_helper, fra_helper, swap_helper,
)
from finance.pricing.pricers import SwapPricer  # noqa: E402
from finance.pricing.engines.jax import JaxProgram  # noqa: E402


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
    result = CurveCalibrator(
        helpers, GlobalSolver(), CurveDefinition("USD", "SOFR", CurveInterpolator.LogLinearDF)
    ).calibrate(base)
    market = base.with_curve(
        YieldCurve.from_registry(result.origin, result.zero_curve, currency="USD", index_name="SOFR")
    )
    return market, cn


def book(as_of: Date, n: int) -> list:
    tenors = ["2Y", "3Y", "5Y", "7Y", "10Y"]
    return [
        Swap.fixed_float_swap(
            notional=(1.0 + i) * 1e6,
            rate_index="SOFR",
            fixed_rate=0.04,
            tenor=tenors[i % 5],
            as_of=as_of,
        )
        for i in range(n)
    ]


def anatomy(n: int, as_of: Date, market, cn: str) -> None:
    prog = SwapPricer().compile(book(as_of, n))
    jp = JaxProgram(prog.inputs, market)
    disc0, proj0 = jp.params_from_market(market)
    z = disc0[cn]
    F, M = jp.F, getattr(jp, "M", 0)
    U = int(jp.obs_unique_off.shape[0]) if M else 0
    print(f"\n{'='*72}\nBOOK {n:>3} swaps   F={F} flows, M={M} fixings, U={U} unique dates\n{'='*72}")

    specs = {
        "PV (scalar)": lambda z: jp.instrument_pv({**disc0, cn: z}, {**proj0, cn: z}),
        "KRD ladder (jacfwd)": jax.jacfwd(
            lambda z: jp.instrument_pv({**disc0, cn: z}, {**proj0, cn: z})
        ),
        "gamma (hessian)": jax.hessian(
            lambda z: jp.pv({**disc0, cn: z}, {**proj0, cn: z})
        ),
    }

    for label, fn in specs.items():
        jf = jax.jit(fn)
        t0 = time.perf_counter()
        lowered = jf.lower(z)               # trace + lower to StableHLO
        t_lower = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        compiled = lowered.compile()        # XLA -> executable (the expensive stage)
        t_compile = (time.perf_counter() - t0) * 1e3
        # steady-state execute
        compiled(z).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(100):
            compiled(z).block_until_ready()
        t_exec = (time.perf_counter() - t0) / 100 * 1e3
        try:
            hlo_lines = len(lowered.as_text().splitlines())
        except Exception:
            hlo_lines = -1
        print(f"  {label:<22} trace+lower {t_lower:8.1f} ms | XLA compile {t_compile:8.1f} ms "
              f"| execute {t_exec:7.3f} ms | HLO {hlo_lines} lines")


def main() -> None:
    as_of = Date(2026, 6, 1)
    market, cn = build_market(as_of)
    for n in (1, 200):
        anatomy(n, as_of, market, cn)

    # cache hit: a SECOND jit of the same fn at the same signature is a pure cache lookup.
    prog = SwapPricer().compile(book(as_of, 200))
    jp = JaxProgram(prog.inputs, market)
    disc0, proj0 = jp.params_from_market(market)
    z = disc0[cn]
    f = lambda z: jp.instrument_pv({**disc0, cn: z}, {**proj0, cn: z})  # noqa: E731
    g = jax.jit(jax.jacfwd(f))
    g(z).block_until_ready()                       # first call: compiles
    t0 = time.perf_counter(); g(z).block_until_ready(); warm = (time.perf_counter() - t0) * 1e3
    print(f"\nsame-signature re-call (cache hit) : {warm:.3f} ms  (no recompile)")
    print("a DIFFERENT-shaped book is a new signature -> full recompile (the tax to plan around:")
    print("pad/bucket book shapes, or enable jax_compilation_cache_dir to persist across runs).")


if __name__ == "__main__":
    main()
