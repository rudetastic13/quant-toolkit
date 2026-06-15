"""Performance study: compile-once / reprice-many over a 1,000-swap population.

Builds a calibrated SOFR market, generates ~1k randomized vanilla swaps, then times the
three hot operations the architecture is designed around:

    1. compile   — ResolvedSwaps -> one columnar PricingProgram (paid once)
    2. price      — a single reprice(market) over the whole population
    3. KRD        — bump-and-reprice across every curve pillar (2*P repricings)

The point it demonstrates: pricing 1k swaps is *one* vectorized array pass, not 1k Python
loops, and a full key-rate ladder is just 2*P of those passes over the same compiled arrays
(no rebuilds).  Run:

    PYTHONPATH=src python research/swap_performance/benchmark.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from finance.dates import Date
from finance.instruments.resolution import Swap, curve_name
from finance.markets.context import MarketContext
from finance.markets.curves import CurveNamespace, CurveInterpolator
from finance.pricing.calibration import (
    CurveCalibrator,
    CurveDefinition,
    GlobalSolver,
    deposit_helper,
    fra_helper,
    swap_helper,
)
from finance.pricing.pricers import SwapPricer
from finance.pricing.risk import Sensitivities

AS_OF = Date(2026, 6, 1)
N_SWAPS = 1_000
SEED = 20260601


def build_market() -> MarketContext:
    """Calibrate a SOFR curve off a money-market + swap quote strip (mirrors the doc example)."""
    helpers = [
        deposit_helper(rate=0.0430, tenor="1M", as_of=AS_OF),
        deposit_helper(rate=0.0432, tenor="3M", as_of=AS_OF),
        deposit_helper(rate=0.0435, tenor="6M", as_of=AS_OF),
        fra_helper(rate=0.0440, start="6M", end="12M", as_of=AS_OF),
        swap_helper(rate=0.0420, tenor="2Y", as_of=AS_OF),
        swap_helper(rate=0.0410, tenor="3Y", as_of=AS_OF),
        swap_helper(rate=0.0405, tenor="5Y", as_of=AS_OF),
        swap_helper(rate=0.0408, tenor="7Y", as_of=AS_OF),
        swap_helper(rate=0.0415, tenor="10Y", as_of=AS_OF),
        swap_helper(rate=0.0425, tenor="20Y", as_of=AS_OF),
        swap_helper(rate=0.0430, tenor="30Y", as_of=AS_OF),
    ]
    base = MarketContext(as_of_date=AS_OF, curves=CurveNamespace())
    target = CurveDefinition(curve_name("USD", "SOFR"), CurveInterpolator.LogLinearDF)
    result = CurveCalibrator(helpers, GlobalSolver(), target).calibrate(base)
    assert result.solver_result.converged, "calibration did not converge"
    return result.market


def build_population(n: int) -> list[Swap]:
    """n randomized receive/pay-fixed vanilla SOFR swaps across the curve's tenor span."""
    rng = np.random.default_rng(SEED)
    tenors = rng.integers(1, 31, size=n)               # 1Y..30Y
    notionals = rng.uniform(5e6, 250e6, size=n)
    signs = rng.choice([1.0, -1.0], size=n)            # receive (+) / pay (-) fixed
    fixed = rng.uniform(0.035, 0.045, size=n)          # struck around par
    return [
        Swap(
            notional=float(sign * notion),
            rate_index="SOFR",
            fixed_rate=float(rate),
            tenor=f"{int(yrs)}Y",
            as_of=AS_OF,
        )
        for yrs, notion, sign, rate in zip(tenors, notionals, signs, fixed)
    ]


@dataclass
class Timing:
    label: str
    seconds: float
    items: int

    @property
    def per_item_us(self) -> float:
        return self.seconds / self.items * 1e6

    def line(self) -> str:
        return f"{self.label:<28} {self.seconds * 1e3:9.2f} ms   {self.per_item_us:8.1f} us/swap"


def timed(label: str, items: int, fn):
    t0 = time.perf_counter()
    out = fn()
    return out, Timing(label, time.perf_counter() - t0, items)


def main() -> None:
    print(f"rates-lib swap performance study  —  as_of={AS_OF.to_str()}, n={N_SWAPS}, seed={SEED}\n")

    market = build_market()
    CN = curve_name("USD", "SOFR")

    swaps = build_population(N_SWAPS)
    pricer = SwapPricer()

    # 1. compile once
    program, t_compile = timed("compile (once)", N_SWAPS, lambda: pricer.compile(swaps))
    n_flows = program.inputs.pay_dates.size

    # 2. price — single reprice over the whole population
    priced, t_price = timed("price (1 reprice)", N_SWAPS, lambda: program.price(market))

    # warm reprice (steady-state, no first-call overhead)
    _, t_reprice = timed("reprice (warm)", N_SWAPS, lambda: program.price(market))

    # 3. key-rate durations — bump-and-reprice across every pillar
    sens = Sensitivities(program, market)
    krd, t_krd = timed("KRD ladder (2*P reprices)", N_SWAPS, lambda: sens.key_rate_durations(CN))
    n_reprices = 2 * krd.krd.shape[1]

    print(f"population: {N_SWAPS} swaps  ->  {n_flows:,} flows (accrual periods) compiled\n")
    for t in (t_compile, t_price, t_reprice):
        print(t.line())
    print(t_krd.line())
    print(
        f"\nKRD did {n_reprices} repricings ({krd.krd.shape[1]} pillars x 2) over the same program "
        f"=> {t_krd.seconds * 1e3 / n_reprices:.2f} ms/reprice"
    )

    # sanity: portfolio totals
    pv = priced.instrument_pv
    print(
        f"\nportfolio PV             = {pv.sum():>16,.2f}"
        f"\nportfolio DV01 (sum KRD) = {krd.total.sum():>16,.2f}"
    )


if __name__ == "__main__":
    main()