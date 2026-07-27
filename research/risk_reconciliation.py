"""Risk reconciliation — do partial-01s tie out to net DV01, across engines and risk maps?

Domain check before building the risk-map layer.  For a spot-starting (all-forward) swap
book we compute, on BOTH engines:

  * net DV01           — parallel +1bp move of the base curve
  * zero-space KRD  g  — ∂PV/∂z at the base-curve pillars (dim P, fixed)
  * partial-01s        — g re-expressed onto a *risk map* (a set of instruments), s = g·J_B⁺

and report three distinct "gaps":

  1. Σ partials  vs  net DV01     — the par-vs-zero parallel correspondence ("close if not
     exact"); present even for the square swap basis.
  2. reconstruction residual  ‖g − s·J_B‖  — zero when the map spans (K≥P, full rank),
     nonzero (= unhedgeable risk) when the map is under-specified (K<P).
  3. numpy bump  vs  JAX AD       — should collapse to ~1e-6 now that x64 is on.

Risk maps exercised: the official swap basis (K=P) and a forward-FRA strip (K≠P) — the same
curve, a different instrument representation, per the domain requirement.

The seasoned/historic-reset case is intentionally absent: neither engine supports realized
fixings for RFR coupons yet (JAX has no fixings leaf; the numpy compounded path never
consults ``fixings``).  That is engine work, tracked separately.

Run:  PYTHONPATH=src python research/risk_reconciliation.py
"""
from __future__ import annotations

import os

# Must precede any JAX import (autodiff -> jax) or the AD engine runs in float32.
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np

import sofr_curve_calibration as S
from finance.pricing.calibration import fra_helper
from finance.pricing.engines.jax.calibration import calibration_jacobian
from finance.pricing.pricers import SwapPricer
from finance.pricing.risk import Sensitivities
from finance.pricing.risk.autodiff import AutodiffRisk

import jax

AS_OF, CURVE = S.AS_OF, S.CURVE


# ---------------------------------------------------------------------------
# Book — spot-starting (all-forward) swaps at par
# ---------------------------------------------------------------------------

BOOK_TENORS = ("5Y", "10Y")


def build_book(market) -> tuple[list, list[str]]:
    swaps, labels = [], []
    for tenor in BOOK_TENORS:
        par = S.swap_helper(rate=0.0, tenor=tenor, as_of=AS_OF).implied(market)
        swaps.append(
            S.Swap.fixed_float_swap(
                notional=100e6, rate_index="SOFR", fixed_rate=par, as_of=AS_OF, tenor=tenor,
            )
        )
        labels.append(f"{tenor} recv-fixed @par")
    return swaps, labels


# ---------------------------------------------------------------------------
# Risk maps — Jacobians J_B = ∂(basis quote)/∂z against the SAME base curve
# ---------------------------------------------------------------------------

def fra_strip_jacobian(market, curve) -> tuple[np.ndarray, list[str]]:
    """Forward-FRA strip between consecutive curve-pillar tenors — a different representation."""
    tenors = [t for (t, _, _) in S.SOFR_QUOTES]
    helpers, labels = [], []
    for a, b in zip(tenors, tenors[1:]):
        try:
            helpers.append(fra_helper(rate=0.0, start=a, end=b, as_of=AS_OF))
            labels.append(f"{a}x{b}")
        except Exception:
            continue  # skip any tenor pair the FRA builder rejects
    J = calibration_jacobian(helpers, CURVE, market, curve)
    return J, labels


# ---------------------------------------------------------------------------
# Zero-space gradient g (per +1bp) and net DV01, per engine
# ---------------------------------------------------------------------------

def engine_bump(program, market) -> tuple[np.ndarray, np.ndarray]:
    sens = Sensitivities(program, market)
    return sens.key_rate_durations(CURVE).krd, sens.dv01(CURVE)  # g (n_inst,P), net (n_inst,)


def engine_ad(program, market) -> tuple[np.ndarray, np.ndarray]:
    g = AutodiffRisk(program, market).zero_ladder(CURVE)  # (n_inst, P), per +1bp
    return g, g.sum(axis=1)  # parallel proxy = key-rate additivity


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

def reconcile(g: np.ndarray, J: np.ndarray) -> dict:
    """g (n_inst,P) per +1bp; J (K,P). Partials + the basis's own parallel + residual.

    ``Σ partials`` ties EXACTLY (K≥P, full rank) to this basis's *par-parallel* DV01 —
    g·(J⁺·1_K) — NOT to the zero-parallel DV01.  A parallel par-quote shift and a parallel
    zero shift are different scenarios (they differ by curve convexity, ~2.5% here), so the
    meaningful "how far off" is the reconstruction residual ‖g − s·J‖ (unhedgeable risk),
    which is 0 iff the map spans the base curve.
    """
    Jpinv = np.linalg.pinv(J)                     # (P, K)
    s = g @ Jpinv                                 # (n_inst, K) partial-01s in the map basis
    par_parallel = s.sum(axis=1)                  # == g @ (J⁺ @ 1_K), the basis's net DV01
    recon_resid = np.abs(g - s @ J).max(axis=1)   # unhedgeable residual (0 if map spans)
    return dict(partials=s, par_parallel=par_parallel, recon_resid=recon_resid)


def report(labels, engines, maps) -> None:
    print(f"\njax x64 enabled: {jax.config.read('jax_enable_x64')}\n")
    for eng_name, (g, net_zero) in engines.items():
        print(f"=== engine: {eng_name} ===")
        for map_name, (J, K) in maps.items():
            r = reconcile(g, J)
            print(f"  risk map: {map_name}  (K={K}, P={g.shape[1]})")
            print(f"    {'instrument':20} {'net DV01 (zero∥)':>16} {'Σpartials (par∥)':>17} "
                  f"{'unhedged resid':>15}")
            for i, lbl in enumerate(labels):
                print(
                    f"    {lbl:20} {net_zero[i]:>16,.2f} {r['par_parallel'][i]:>17,.2f} "
                    f"{r['recon_resid'][i]:>15.3e}"
                )
        print()


if __name__ == "__main__":
    res = S.calibrate(S.GlobalSolver(), jacobian=True)
    market, curve, J_swap = S.register_sofr_curve(res), res.zero_curve, res.jacobian

    swaps, labels = build_book(market)
    program = SwapPricer().compile(swaps)

    pv = program.price(market).instrument_pv
    print("\nSpot-start par swaps (PV should be ~0):")
    for lbl, v in zip(labels, pv):
        print(f"  {lbl:22} PV = {v:>14,.2f}")

    J_fra, _ = fra_strip_jacobian(market, curve)
    maps = {
        "swap basis (official)": (J_swap, J_swap.shape[0]),
        "forward-FRA strip": (J_fra, J_fra.shape[0]),
    }

    engines = {
        "numpy bump (float64)": engine_bump(program, market),
        "jax AD (x64)": engine_ad(program, market),
    }

    # cross-engine agreement on the zero-space gradient g
    g_bump, g_ad = engines["numpy bump (float64)"][0], engines["jax AD (x64)"][0]
    print(f"\nmax |g_bump - g_ad| (zero KRD, per bp): {np.abs(g_bump - g_ad).max():.3e} USD/bp")

    report(labels, engines, maps)
