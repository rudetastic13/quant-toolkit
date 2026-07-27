# Risk Maps and the Dual-Number AD Direction

Two design conclusions from scoping the risk engine (July 2026). Both are validated by the
reconciliation harness `research/risk_reconciliation.py`; this doc is the short rationale.

---

## 1. Risk maps — a PV-preserving change of basis on the risk vector

**Domain.** A single **base (official) curve** owns valuation: PV is a function of its zero
rates `z` (dimension `P`, fixed) and nothing else. A trader wants net DV01 and bucketed
exposures, and wants to view the *same* position's risk against different instrument sets — a
**risk map** — without changing the PV.

**The decomposition that makes this clean.** Everything factors into two pieces with
different shape behavior:

- **Piece 1 — `g = ∂PV/∂z`**, the zero-space gradient at the base-curve pillars. Depends on
  the **cashflow structure (varies)** and the **base curve (fixed `P`)**. This is the
  expensive part (the pricer / AD).
- **Piece 2 — the risk map**, `reported = g · M`. Pure linear algebra on the fixed-`P` space.
  PV-invariant by construction: it only re-coordinatizes the *reporting* of `∂PV/∂z`.
  Changing the map never reprices and never recompiles.

**It is a Jacobian remap, not interpolation.** The report basis is a set of *real
instruments* (swaps, FRAs). Each has a true sensitivity to the base pillars,
`J_B = ∂(basis quote)/∂z`, computed by the same pricer that values everything else. Partials
are

```
s = g · J_B⁺           (pseudo-inverse: K risk instruments need not equal P pillars)
```

Interpolation bucketing weights risk by *time position* on a grid; the Jacobian weights it by
the *actual risk of tradeable instruments*, so each bucket answers a hedge question. This is
what makes "represent the SOFR curve with FRAs instead of swaps" meaningful — a FRA and a par
swap keyed to overlapping pillars have different gradient shapes, which only `J_B` captures.

**Net DV01 is basis-dependent — pick the basis.** The partials tie **exactly** (for `K ≥ P`,
full rank) to their basis's *par-parallel* DV01, **not** the *zero-parallel* DV01. A +1bp
parallel zero move shifts par quotes by 0.99–1.10bp across tenors (`J·1`), so the two
"parallels" are different scenarios differing ~2.5%:

| 5Y par swap ($100mm) | value |
|---|--:|
| net DV01, zero-parallel (bump all zeros 1bp) | −46,317.94 |
| net DV01, par-parallel (bump all par quotes 1bp) = Σ swap-basis partials | −45,167.90 |

**Decision:** define net DV01 as the **par-parallel in the report basis** (= Σ reported
partials). Then tie-out is exact by construction, and the real "how far off" metric is the

```
unhedged residual = ‖ g − s · J_B ‖      (0 iff the map spans the trade's risk)
```

Measured: ~1e-11 for the square swap basis, ~1e-6 for a 21-instrument FRA strip against a
23-pillar curve — i.e. `K ≠ P` reconciles fully as long as the map covers the trade's actual
risk; the residual only grows when it doesn't.

**Prior art (OpenGamma Strata).** Strata keeps the two representations first-class and
separate: `PV01_CALIBRATED_*` (zero/node) vs `PV01_MARKET_QUOTE_*` (par), joined by the
stored `JacobianCalibrationMatrix` + `MarketQuoteSensitivityCalculator`. It never collapses
them — validating the "define the basis" conclusion. Our two extensions beyond Strata:
**arbitrary non-calibration risk maps**, and **`K ≠ P` with a residual metric** (Strata
assumes square calibration).

---

## 2. Dual-number AD — runtime AD over a fixed-`P` base curve

**Why not JAX.** JAX is ahead-of-time, graph-based AD: XLA specializes the executable on
input shape, so it recompiles whenever the shape moves. For a risk engine over ragged,
ever-changing portfolios that is a structural mismatch — measured ~300 ms compile per fresh
instance vs ~0.5 ms warm execution (the numpy bump path is ~17 ms for a full ladder). The
compile tax is paid again on every new shape.

**The real axis is graph/AoT AD (shape-specialized) vs runtime AD (shape-agnostic).** The
risk engine wants runtime AD. Crucially, the domain (§1) says the AD engine has exactly one
job: produce `g = ∂PV/∂z` over **varying cashflow shapes** onto a **fixed-`P`** output basis.
Everything trader-facing is linear post-processing. So the only axis that varies for the AD
core is cashflows — precisely where runtime dual-number AD pays nothing, while risk-map
changes never reach the pricer at all.

**Options, ranked for this use case:**

| approach | mode | shape-compile? | notes |
|---|---|---|---|
| **Dual numbers** (operator overload) | forward | none | exact δ + γ (`Dual`/`Dual2`), any shape. Cost ∝ #inputs. rateslib's choice (Rust-backed). **Start here.** |
| **Reverse-mode AAD** (tape) | reverse | none | full gradient in ~3–5× primal regardless of #inputs — for large factor sets (vol surfaces, big books). γ via forward-over-reverse. |
| PyTorch eager (`torch.func`) | reverse | none in eager | fastest to adopt; tensor overhead on tiny arrays. |
| Enzyme / CoDiPack in the C++ core | either | none | reverse AD at C++ speed with dynamic shapes; composes with the existing `finance/_core`. |

**Numba is the compiler, not the differentiation system.** The implemented engine takes the
AD-adjacent route: a hand-written reverse adjoint of the stable cashflow kernel, with both
primal and adjoint compiled by Numba. It produces exact first order without tracing or shape
specialization; its formulas must be kept in sync when pricing algebra changes.

**Recommended sequencing (cheapest first):**

1. **Now (implemented):** hand-written reverse adjoint through the Numba cashflow kernel —
   exact δ, one dtype/layout compile, float64, and ragged-shape-safe.
2. **When pricing algebra expands materially:** consider dual-number types or
   CoDiPack/Enzyme in `_core` to reduce manual adjoint maintenance.
3. **Keep bump-and-reprice** as the universal fallback for non-smooth products (American
   swaptions, barriers, MC).
4. **Drop JAX from the fast path** — keep it only as a cross-check oracle in tests.

Strata precedent again: OpenGamma uses hand-coded analytic point sensitivities per pricer
(the AAD family) with finite differences only as a test oracle — a deliberate rejection of
both pure-bump and trace-compiled AD for the production path.

**Aside — precision.** JAX runs float32 unless `JAX_ENABLE_X64=true` is set *before* the
first JAX import. With x64 on, the bump and AD engines agree to float64; without it, AD is
limited to ~1e-3 relative.
