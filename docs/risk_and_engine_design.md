# Risk and engine design — RiskTemplate, Treasury products, op-graph engine, cleanup

> **Status: DESIGN + IMPLEMENTATION GUIDE (2026-10-13). Nothing in §2–§6 is implemented.**
> Written from a read-only sweep of `main` at `2ef5589`. It supersedes §1 of
> [risk_map_and_dual_ad.md](risk_map_and_dual_ad.md) (the `pinv` risk map), expands
> [engine_op_graph_plan.md](engine_op_graph_plan.md), and is the companion to
> [pricing_architecture.md](pricing_architecture.md), [sofr_futures.md](sofr_futures.md) and
> [portfolio_and_scenarios.md](portfolio_and_scenarios.md).

The reader is the person implementing this, so the document is explicit about signatures, file
paths, array layouts, failure modes and tests. Where a number is a default rather than a derived
fact it says so.

Scope agreed for the library: SOFR swaps (any leg shape), SOFR futures, European SOFR
swaptions, US Treasury bonds (on-the-run and off-the-run, a fitted UST curve), Treasury bond and
note futures, and options on Treasury futures. The `RiskTemplate` is the production risk view
across all of them.

---

## 0. Verdict and plan

**Doable.** Everything below builds on the three things that already work and are tested: the
columnar IR (`KernelInputs` → `reprice`), the interpolator-generic Numba adjoint
(`ZeroCurve.log_df_affine` → CSR query table → `NumbaAdjointResult`), and the calibration layer
(`CalibrationInstrument` protocol, `CurveCalibrator`, `calibration_jacobian`). The new products
are mostly *new instruments and new quote measures* over that spine; the RiskTemplate is linear
algebra over gradients the adjoint already produces.

| Phase | Deliverable | Depends on | Effort (one person) |
|---|---|---|---|
| **0** | Cleanup + prerequisites: SIFMA calendar, Act/Act ICMA, 30/360 fix, required day count, `discount_spread` column, `FuturesProgram` fold, `leg_group` | — | 3–4 days |
| **R** | `RiskTemplate` over swaps / deposits / FRAs / SOFR futures / swaptions, `PillarRisk` contract, `RiskEngine` rewire, delete `partial_dv01` ×2 | 0 (partly) | 3–4 days |
| **B** | UST bonds: instrument, analytics (accrued, price↔yield, DV01/duration/convexity), curve pricing with per-bond spread, `BondHelper`, UST curve fit (overdetermined), bond template | 0 | 6–8 days |
| **F** | Treasury futures: contract table, conversion factors, basket, CTD, forward price, risk through the CTD, `TreasuryFuturesHelper`, futures template | B | 4–5 days |
| **O** | Options on Treasury futures: European Black-76 / Bachelier on the futures price, premium-upfront discounting, sticky-strike risk; American exercise as an explicit later model | F | 2 days (+3 for American) |
| **E** | Op-graph engine: `(primal, vjp)` registry, program as op list, one reverse sweep by query id, single `risk()` entry, backends as runners; retires the six risk doors and the hand-chained product risk | R (for the contract) | 10–15 days |
| **S** | Swaption production gaps: normal-vol cube, settlement styles, vega bucketing | R | 3–4 days |

Rough total: 7–9 weeks of focused work. The trading-relevant milestone — a production SOFR
curve with a par-swap `RiskTemplate` over a book of swaps, SOFR futures and swaptions — is
phases 0 + R, about 1.5 weeks. Bonds and Treasury futures are the next 2–3 weeks. The op-graph
engine is a refactor that makes the product risk code smaller; it is not on the critical path for
any product and should not block trading on phases R, B, F.

What is genuinely hard, and where the PnL risk lives:

1. **RiskTemplate conditioning.** A badly posed template produces saw-tooth bucket risk with
   alternating signs; hedging it is a loss on bid-offer with a wrong net. §2.5 refuses those
   templates instead of printing numbers.
2. **Treasury futures are discontinuous in the CTD.** First-order risk through the current CTD is
   correct until the basket switches. §3.3 reports the switch margin and the risk of the runner-up
   so the discontinuity is visible, not hidden.
3. **Bond conventions.** Accrued, Act/Act ICMA, the last-period simple-interest rule, settlement
   calendars. Each is a one-line formula that is easy to get subtly wrong; §3.2 pins every one to
   the Treasury's published formulas and to an external oracle in tests.
4. **Units and signs.** §2.6 fixes them once; nothing downstream flips a sign.

---

## 1. Where the code is today (sweep of 2026-10-13)

Product coverage and the facts the design relies on. File references are `path:line` on
`2ef5589`.

### 1.1 Products

| Product | Instrument | Pricer / program | Risk today | Calibration helper |
|---|---|---|---|---|
| Swap, Deposit, FRA | `Swap`/`Deposit`/`Fra` (`instruments/resolution/builders.py`), all `Priceable`, all `legs` | `LegPricer` → `PricingProgram` (`pricers/legs.py`, `pricers/base.py`) | `NumbaRisk`, `AutodiffRisk`, `Sensitivities`, `RiskEngine` | `SwapHelper`, `DepositHelper`, `FraHelper` |
| SOFR future (SR1/SR3) | `SofrFuture` (`resolution/futures.py`), frozen, not `Priceable`, `legs` = one unit leg | `FuturesProgram` (`pricers/futures.py`) — compiles with `leg_spec` directly, bypassing `LegPricer.compile` and `projection_curve_name` | `FuturesProgram.risk(market, curve)` → `(n_flows, P)` rate/price/pnl gradients, index role only; **recomputes DFs without the `pay_dates >= origin` mask** (`pricers/futures.py:78-81`) | `FuturesHelper` (`convexity` exogenous or callable, residual `implied − (quote − CA)`) |
| European swaption | `Swaption` (`resolution/swaption.py`), frozen, not `Priceable`, no `legs` | `SwaptionProgram` (`pricers/swaption.py`): unit-coupon swap program, forward = −float/fixed, annuity = |fixed|, Black/Bachelier on `(forward, annuity, vol)` | `SwaptionProgram.risk(market, curve, bp)` → chained through forward and annuity, sticky strike, no vega bucketing | none |
| UST bond | **none** (`ProductKind.Bond` is an unused enum member) | none | none | `QuoteKind.Yield = 3` is a label with no helper |
| Treasury future / option | **none** | none | none | none |

### 1.2 Facts the design depends on

- **Curve identity.** Curves are keyed by `CurveKey` (`finance/conventions/keys.py`):
  projection `"CCY.INDEX.TENOR"` (`USD.SOFR.ON`) and discount `"CCY.NAME"` (`USD.STDCSA`,
  `USD.UST`). `funding_curve_name(ccy, funding_id)` is literally the discount key; there is no
  alias from `STDCSA` to an index, the market binds the SOFR curve under `USD.STDCSA` itself
  (curves_and_market_plan.md PR C, done).
- **Parameters.** The engines differentiate with respect to `ZeroCurve.node_zero_rates`:
  continuously-compounded Act/365 zero rates at the non-origin pillars, `z_j = −ln DF_j / (x_j/365)`.
  `NumbaProgram.params_from_market` concatenates curves in `KernelInputs.curve_names` order
  (first appearance in the compiled book); `NumbaAdjointResult.curve_slice(name)` recovers a
  block. Gradients are `∂PV/∂z` in PV currency per unit of zero rate (not bp-scaled).
- **Adjoint output.** `NumbaAdjointResult.instrument_gradient(curve, role="index"|"funding"|"total")`
  → `(n_instruments, P_curve)`; `cashflow_gradient` → `(F, P_curve)`. The dense per-flow
  `(F, Z)` arrays are allocated inside `_execute` whenever the adjoint runs (`numba/program.py:120-121`).
- **Calibration Jacobian.** `engines/numba/calibration.calibration_jacobian(helpers, target_name, market, curve)`
  → `(n_helpers, P_target)`, rows in helper order (the calibrator passes them sorted by
  `pillar_date`), columns = target pillars, **other curves held fixed**. Deposit/FRA rows come
  from `log_df_affine`; swap rows from a fresh `NumbaProgram` plus the par-rate quotient rule;
  any other helper must expose `numba_gradient(market)` (only `FuturesHelper` does).
- **Calibration shape.** One curve per `calibrate` call, one node per helper, pillars strictly
  increasing, pillar set = helper `pillar_date`s. `GlobalSolver` is scipy `least_squares`
  (finite-difference Jacobian); `Bootstrapper` is per-pillar `brentq`, local schemes only.
  `CurveDefinition.interpolation` accepts an `Interpolator` instance, so `Mixed` is reachable
  through `GlobalSolver` even though `CurveInterpolator` has no `Mixed` entry.
- **Market context.** `MarketContext(as_of_date, curves: CurveNamespace, vols: object | None, conventions)`
  with `with_curve`, `yield_curve`, `zero_curve`, `convention`. `with_curve` rebuilds the
  namespace and **resets every curve version to 0** (`markets/context.py:30-43`), so namespace
  versions cannot key a cache. `ZeroCurve` compares by identity.
- **Caching idiom.** `PricingProgram.prepare(market, backend=Numba)` caches a `NumbaProgram` and
  rebuilds when `NumbaProgram.compatible(market)` is false (origin, scheme, node dates).
- **Risk doors (six).** `Sensitivities` (bump), `NumbaRisk` (adjoint; `partial_dv01 = g @ pinv(J)`),
  `AutodiffRisk` (JAX, LogLinearDF only; `partial_dv01 = g @ inv(J)`), `RiskEngine`
  (dispatches `PricingProgram` → `NumbaRisk`, `SwaptionProgram` → `.risk`; rejects
  `FuturesProgram`), `SwaptionProgram.risk`, `FuturesProgram.risk`. The `NumbaRisk = None`
  fallback in `risk/__init__.py` can never trigger (numba is imported lazily at call time).
- **Dates.** `DayCountMethod` has `ActualActual` but `period_fractions` does not implement it
  (`KeyError`). `Thirty360` computes 30E/360 (unconditional `min(D, 30)` on both ends,
  `day_counts/calculator.py:46-68`), not US 30/360. Only the `no_holidays` calendar exists.
  `CommonInstrument.day_count_method` defaults to `Unused`, which accrues **zero**.
- **Conventions.** Live package is `finance/conventions/` (`IndexFamily` with its published
  tenors, `RateIndex` = family + tenor, `MarketConventions` with required `swap`/`deposit`/`fra`
  sections, registry keyed per family `(ccy, name)`; `SOFR`, `FEDFUND` and `CME-TERM-SOFR`
  registered). `markets/conventions/` and `pricing/conventions/` are stale
  `__pycache__` only.
- **Prior design notes.** `risk_map_and_dual_ad.md` §1 defines partials as `s = g · J_B⁺` with
  the unhedged residual `‖g − s·J_B‖`; `research/risk_reconciliation.py::reconcile` is its
  harness. §2.3 below shows that formulation is one basis choice of the template and keeps it
  as an option.

---

## 2. RiskTemplate

### 2.1 What it is, what it is not

A `RiskTemplate` is a *view*: a list of real instruments (par swaps, deposits, futures, bonds,
bond futures) on one or more curves, with a rule for how the base curve moves when one template
quote moves. The risk engine re-expresses the book's pillar risk as risk to those template
quotes. Pricing, accruals, fixings and calibration residuals never leave the base curve; a
template changes no PV and triggers no recalibration.

It is the "one production curve, many risk views" model: the desk owns one snapped SOFR curve
(deposits + SR3 strip + par swaps, `Mixed` interpolation), and risk is reported on a template of
par swaps `1M 3M 6M 9M 12M 15M 18M 21M 24M 3Y 4Y 5Y 7Y 10Y 15Y 20Y 30Y`. SOFR futures positions
then show as par-swap risk by maturity overlap. A Treasury desk runs a second template of
on-the-run bonds on the UST curve, and Treasury futures show as on-the-run risk through their
CTD.

Not in scope: second-order risk on the template (gamma reprices on the base curve), templates
with more buckets than the base curve has degrees of freedom (refused), and templates that
change the base curve's values (that is a scenario, see `portfolio_and_scenarios.md`).

### 2.2 Notation and the math

Fixed market, as-of date `t₀`.

| Symbol | Shape | Meaning |
|---|---|---|
| `z` | `(P,)` | base pillar parameters of all curves the template spans, concatenated per curve: `z = (z₁ … z_C)`, `z_c ∈ ℝ^{P_c}` are that curve's `node_zero_rates` |
| `G` | `(N, P)` | book pillar risk, `G[i, j] = ∂PV_i/∂z_j` (one row per position) — from the adjoint |
| `t_k` | | template instrument `k = 1..K`, on curve `c(k)`, with node offset `x_k` (days from origin, from its `pillar_date`) and quote measure `p_k(z)` (par rate, simple rate, futures rate, yield, price) |
| `J_T` | `(K, P)` | template Jacobian, `J_T[k, j] = ∂p_k/∂z_j`, **on the base curve**, all curves the instrument touches |
| `B` | `(P, K)` | bucket basis: column `k` is the base-pillar move (in `z` units) for a unit move of template node `k`; block-diagonal by curve (`B_c ∈ ℝ^{P_c×K_c}`) |
| `M = J_T B` | `(K, K)` | template quote response to template node moves |
| `A = B M⁻¹` | `(P, K)` | transfer: base move per unit template-quote move; satisfies `J_T A = I_K` |
| `R = G A` | `(N, K)` | **quote risk**, `R[i, k] = ∂PV_i/∂q_k` with all other template quotes held fixed |
| `U = G − R J_T` | `(N, P)` | **unexplained pillar risk**: what remains after the template hedge is executed |
| `s_k` | `(K,)` | unit sensitivity of template instrument `k` to its own quote (§2.8) |

Derivation. Bumping template quote `k` by `δ` and holding the others fixed means finding a base
move `Δz` with `J_T Δz = δ e_k`. Restricting `Δz` to the span of the basis, `Δz = B y`, gives
`M y = δ e_k`, so `Δz = A e_k δ`. The PV change is `G A e_k δ = R[:, k] δ`. Computationally,

```
R = G A = G B M⁻¹      ⇒   solve  Mᵀ X = Bᵀ Gᵀ,   R = Xᵀ          (one LU of Mᵀ, multi-RHS)
```

Properties that matter:

- **Square case is basis-independent.** If `K = P` and `B` is invertible, `R = G J_T⁻¹`. The
  identity template (template instruments = calibration instruments) therefore reproduces
  today's `partial_dv01 = g @ inv(J)` exactly.
- **Exactness on the template.** `J_T A = I` holds for any basis: a template bump always
  reprices template instrument `k` to `q_k + δ` and the others to their quotes, to first order,
  even when base and template interpolate differently between pillars. The basis only decides
  how the base curve moves *between* template nodes, i.e. how pillar risk between nodes is
  attributed.
- **Hedge meaning.** By construction template instrument `k` (built at its quote) has quote
  risk `s_k e_k`. A book with quote risk `R[i, :]` is flattened on the template by
  `h_k = −R[i, k] / s_k` units of instrument `k`. The base-pillar risk of the hedged book is
  `U[i, :]`: `U` is literally the risk left after executing the template hedge (for a futures
  strip hedged with par swaps, `U` is the futures-swap basis exposure; it is not zero and must
  be shown).
- **Two parallels.** `Σ_j G[i, j]` is the zero-parallel DV01; `Σ_k R[i, k] = G A 1` is the
  template-parallel DV01 (all template quotes +1). They differ by a few percent (July 2026
  measurement: 2.5% on a 5Y par swap). Both are reported; neither is "the" DV01 without the
  qualifier.

Supersedes the July note: `risk_map_and_dual_ad.md` computes `s = g J_B⁺`. For `K ≤ P`,
`J_B⁺ = J_Bᵀ (J_B J_Bᵀ)⁻¹`, which is exactly `A` with the basis `B = J_Tᵀ`. That basis is kept
as `LeastSquaresBasis` (§2.3); it is no longer the default, and `K > P` is refused instead of
being pseudo-inverted.

### 2.3 Basis rules

`B` is built per curve from the base pillar offsets `x^base_c` (`zero_curve.x[1:]`, days) and the
template node offsets `x^T_c` on that curve, strictly increasing.

| Rule | `B_c[j, k]` | Character | When |
|---|---|---|---|
| **`TentBasis`** (default) | hat function of node `k` evaluated at base pillar `j`: linear in days between adjacent template nodes, `1` on `[0, x_1]` for the first node and on `[x_K, ∞)` for the last (flat extension) | local, non-negative, partition of unity. Strata's `CurveSensitivityUtils.linearRebucketing` rule. | always, unless there is a reason |
| `LeastSquaresBasis` | `J_T[k, j]` restricted to curve `c` (so `B = J_Tᵀ`, `A = J_Tᵀ (J_T J_Tᵀ)⁻¹`) | global; minimises `‖U‖₂`; reproduces `risk_map_and_dual_ad.md` and `research/risk_reconciliation.py::reconcile` | cross-check; reproducing the July numbers |
| `SchemeBasis(space, interpolator)` | `−diag(365 / x^base) · log_df_affine(x^base)` of a template-shaped `ZeroCurve` (origin + template nodes) | smooth bumps (cubic → overshoot = "curve noise" in buckets); near the origin the `ZeroRate`-space extrapolated node pollutes the first basis function | opt-in experiments only |

`TentBasis` is implemented directly with `np.interp` on node positions, not through a
`ZeroCurve`, so that the flat extension and the origin behave exactly as stated. Under
`TentBasis` the unit of a "template node move" is one unit of zero rate at the node; the
normalisation cancels in `A` and only affects conditioning.

Structural requirement under `TentBasis`: every template node must own at least one base pillar
strictly inside its support `(x_{k−1}, x_{k+1})` (with `x_0 = 0`, `x_{K+1} = ∞`), otherwise its
column of `B` is zero and the template is rank-deficient — "template finer than the base". This is
checked explicitly (§2.5) and the error names the node.

### 2.4 Multiple curves

The template spans the set of curves `{c(k)}` plus every curve a template instrument's quote
depends on (a Treasury-futures template on `USD.UST` also touches `USD.SOFR` for financing).
`J_T` is `(K, P)` over all of them; `B` is block-diagonal (a template node moves only its own
curve). `M = J_T B` is then block-triangular when template instruments of one curve depend on
another (Strata's "transition matrix"), and block-diagonal when they do not. The solve is the
same dense `K×K` LU in both cases — block structure is a conditioning fact, not a code path.

Consequence worth stating: with a SOFR swap template and a UST template of Treasury futures,
"SOFR 5Y +1bp with Treasury-futures prices held fixed" moves the UST curve to compensate the
financing effect. That is the correct reading of "risk to a SOFR quote with the other template
quotes fixed", and it is what the full solve delivers automatically.

Curves in the book's pillar risk that no template covers are reported raw (pillar ladder) and
flagged in `QuoteRisk.uncovered`; they are never silently dropped.

### 2.5 Production safety rules (the checklist `bind` enforces)

Every rule is a hard failure (`TemplateError` with a reason code) unless marked *warn*. Defaults
are parameters on `TemplateLimits`; raising a limit requires passing it explicitly.

| # | Rule | Reason code | Default |
|---|---|---|---|
| 1 | Template nodes strictly increasing per curve; no two template instruments share a node | `UNSORTED_NODES` | — |
| 2 | `K_c ≤ P_c` for every curve | `K_GT_P` | — |
| 3 | `rank(B_c) = K_c` (SVD); under `TentBasis` also the structural support check of §2.3 | `RANK_DEFICIENT` (names the node) | — |
| 4 | `cond₂(M) ≤ max_condition` | `ILL_CONDITIONED` | refuse `1e8`, *warn* above `1e4` |
| 5 | Leverage `ℓ_k = max_j |A[j, k]| ≤ max_leverage` (how far any base pillar moves per unit template move; ringing shows up here before it shows in `cond`) | `LEVERAGE` (names the node) | refuse `25`, *warn* above `5` |
| 6 | Template residuals `ρ_k = p_k(z) − quote_k` (when the template instrument carries a quote) within `max_abs_residual` | `RESIDUAL` | report only (`None`); a desk may set e.g. `2e-4` |
| 7 | Every curve referenced by a template instrument exists in the market and has ≥ 2 nodes | `MISSING_CURVE` | — |
| 8 | Template bound to the market it is used with (`BoundRiskTemplate.matches(market)`), else rebind | `STALE_BINDING` | automatic rebind in `RiskEngine` |

Reconciliation lines reported with every `QuoteRisk` (never thresholds, always visible):
zero-parallel DV01, template-parallel DV01, their difference, `‖U_i‖_∞` and `Σ_j U[i, j]` per
position, the naive tent rebucketing `G B` (no `J_T`; Strata's linear rebucketing) as a
cross-check column, and the template diagnostics (`cond`, leverage, residuals, warnings).

Why these particular defaults: a par-swap template on a swap-built curve gives `cond(M)` in the
tens; a futures-heavy front end legitimately reaches `1e3` (a 3M futures rate responds to its end
pillar with weight `t_end/τ ≈ 8`, a 10Y par rate to its end pillar with weight ≈ 0.6). Singular
templates sit at `1e15+`. `1e8` separates "unusual but real" from "garbage"; the warn level makes
the unusual visible. Leverage is the trader-facing version of the same fact: a 1bp template move
should move base pillars by about 1bp; 5bp is suspicious, 25bp is a distorted view.

### 2.6 Units and signs (fixed here, never adjusted downstream)

- `PillarRisk.values`, `J_T`, `R`, `U` are **derivatives per unit** (per 1.0 of zero rate or
  quote in decimal). `.dv01` properties multiply by `1e-4` (per +1bp of the quote). No other
  scaling exists.
- Sign is the raw derivative. A receiver swap has negative quote DV01 (PV falls when its par
  rate rises); a long bond has negative yield DV01; a long futures position has negative rate
  DV01 and positive price DV01. Reports may add a presentation sign (`RiskReport(sign=...)`),
  the engine never does.
- Quote units follow `QuoteKind`: `ParRate`, `SimpleRate`, `FuturesRate` and `Yield` are
  decimal rates; `FuturesPrice` and `CleanPrice` are price points per 100. A template may mix
  kinds; each column's unit is carried in `QuoteRisk.kinds`.
- Hedge notionals (§2.8) are in the template instrument's own unit: leg notional in currency for
  swaps/deposits/FRAs, face amount for bonds, contracts for futures. Sign positive = the
  "receive the quote"/long direction of the helper's unit instrument.

### 2.7 Objects and files

```
src/finance/pricing/risk/
    pillar.py      PillarRisk, pillar_risk(program, market, *, backend)        # contract engines → templates
    basis.py       BasisRule (Protocol), TentBasis, LeastSquaresBasis, SchemeBasis
    jacobian.py    template_jacobian(helpers, curves, market, *, backend)       # generalises calibration_jacobian
    template.py    TemplateLimits, TemplateError, TemplateDiagnostics, RiskTemplate, BoundRiskTemplate, QuoteRisk
    engine.py      RiskEngine (rewired: pillar_risk / quote_risk / ladder / dv01 / gamma)
    sensitivities.py   unchanged (bump oracle)
    adjoint.py, autodiff.py   partial_dv01 deleted; the rest unchanged until phase E
```

```python
# pillar.py
@dataclass(frozen=True)
class PillarRisk:
    """∂PV/∂z for a book, per curve block. The only thing a template consumes."""
    curves: tuple[str, ...]                 # block order
    param_offsets: np.ndarray               # (C+1,) intp, block j = values[:, off[j]:off[j+1]]
    pillar_dates: dict[str, np.ndarray]     # per curve, non-origin pillars, datetime64[D]
    values: np.ndarray                      # (N, P) float64
    pv: np.ndarray                          # (N,)
    labels: tuple[str, ...] | None = None   # position ids, optional

    def block(self, curve: str) -> np.ndarray: ...                      # (N, P_c) view
    def aligned(self, curves: tuple[str, ...], pillar_dates: dict) -> "PillarRisk":
        """Reorder/zero-fill blocks to another layout; raises if a curve's pillars differ."""

def pillar_risk(program, market, *, backend: Backend = Backend.Numba) -> PillarRisk:
    """Adapter over today's programs; replaced by the op-graph `risk()` in phase E.
    PricingProgram  -> prepare(Numba).value_and_grad(market).instrument_gradient(c, 'total') per curve
    FuturesProgram  -> pnl_gradient per curve (index role; margined, no discounting)
    SwaptionProgram -> risk(market, c).curve_gradient per curve
    """
```

```python
# basis.py
class BasisRule(Protocol):
    def matrix(self, base_x: FloatArray, node_x: FloatArray, *, jacobian_block: FloatArray) -> FloatArray:
        """(P_c, K_c). base_x, node_x in days from origin, both strictly increasing, > 0."""

@dataclass(frozen=True)
class TentBasis: ...                       # np.interp hats; flat outside [node_x[0], node_x[-1]]
@dataclass(frozen=True)
class LeastSquaresBasis: ...               # jacobian_block.T
@dataclass(frozen=True)
class SchemeBasis:
    space: CurveSpace = CurveSpace.ZeroRate
    interpolator: Interpolator = Linear()
```

```python
# jacobian.py
def template_jacobian(helpers: Sequence[CalibrationInstrument], curves: tuple[str, ...], market: MarketContext,
                      *, backend: Backend = Backend.Numba) -> np.ndarray:
    """(K, P) with P = Σ_c (pillars of c). Row k = ∂implied_k/∂z over every curve in `curves`
    (zeros for curves the helper does not touch). Rows in `helpers` order (NOT re-sorted).

    Batched by helper kind: all SwapHelpers in one LegPricer program (one value_and_grad, then
    the par-rate quotient rule per instrument); Deposit/Fra via log_df_affine on their curve;
    FuturesHelpers in one FuturesProgram; BondHelpers (phase B) in one bond program with the
    implicit-function rule; TreasuryFuturesHelpers (phase F) through their program.
    """

def calibration_jacobian(helpers, target_name, market, curve) -> np.ndarray:
    """Kept for CurveCalibrator: template_jacobian(helpers, (target_name,), market)."""
```

```python
# template.py
class TemplateError(ValueError):
    code: str          # UNSORTED_NODES | K_GT_P | RANK_DEFICIENT | ILL_CONDITIONED | LEVERAGE | RESIDUAL | MISSING_CURVE | STALE_BINDING
    node: str | None   # label of the offending template node when applicable

@dataclass(frozen=True)
class TemplateLimits:
    max_condition: float = 1e8
    warn_condition: float = 1e4
    max_leverage: float = 25.0
    warn_leverage: float = 5.0
    max_abs_residual: float | None = None

@dataclass(frozen=True)
class TemplateDiagnostics:
    condition: float
    leverage: np.ndarray            # (K,)
    residuals: np.ndarray           # (K,) p_k(z) − quote_k, NaN when no quote
    rank: dict[str, int]            # per curve
    warnings: tuple[str, ...]

@dataclass(frozen=True)
class RiskTemplate:
    instruments: tuple[CalibrationInstrument, ...]
    basis: BasisRule = TentBasis()
    limits: TemplateLimits = TemplateLimits()
    labels: tuple[str, ...] | None = None          # default: f"{helper.curve} {tenor-ish}" from the helper
    name: str = ""                                 # audit label, e.g. "SOFR par 1M-30Y v3"

    @property
    def curves(self) -> tuple[str, ...]: ...       # template curves, first appearance
    def bind(self, market: MarketContext, *, backend: Backend = Backend.Numba) -> "BoundRiskTemplate": ...
    @classmethod
    def identity(cls, instruments: Sequence[CalibrationInstrument], **kw) -> "RiskTemplate": ...
    @classmethod
    def from_spec(cls, spec: TemplateSpec, as_of: Date, **kw) -> "RiskTemplate": ...   # §2.9
    def to_spec(self) -> TemplateSpec: ...

@dataclass(frozen=True)
class BoundRiskTemplate:
    template: RiskTemplate
    curves: tuple[str, ...]                 # spanned curves (template curves ∪ touched curves)
    param_offsets: np.ndarray               # (C+1,)
    pillar_dates: dict[str, np.ndarray]
    node_x: dict[str, np.ndarray]
    jacobian: np.ndarray                    # J_T (K, P)
    basis: np.ndarray                       # B   (P, K)
    transfer: np.ndarray                    # A   (P, K)
    unit_sensitivity: np.ndarray            # s   (K,)
    quotes: np.ndarray                      # (K,) NaN where absent
    diagnostics: TemplateDiagnostics
    _lu: tuple                              # scipy.linalg.lu_factor(Mᵀ)
    _curve_refs: dict[str, ZeroCurve]       # identity anchors for matches()
    as_of: Date

    def matches(self, market: MarketContext) -> bool: ...
    def quote_risk(self, pillar: PillarRisk) -> "QuoteRisk": ...
    def rebucket(self, pillar: PillarRisk) -> np.ndarray: ...          # (N, K) = G B, naive cross-check
    def hedge(self, risk: "QuoteRisk") -> np.ndarray: ...              # (N, K) unit-instrument notionals
    def base_move(self, k: int, size: float) -> dict[str, np.ndarray]: ...   # A e_k·size per curve, for scenarios/tests

@dataclass(frozen=True)
class QuoteRisk:
    labels: tuple[str, ...]; kinds: tuple[QuoteKind, ...]; curves: tuple[str, ...]
    values: np.ndarray                      # (N, K) ∂PV/∂q
    unexplained: np.ndarray                 # (N, P) U
    rebucketed: np.ndarray                  # (N, K) naive G B
    pillar_parallel: np.ndarray             # (N,) Σ_j G
    template_parallel: np.ndarray           # (N,) Σ_k R
    uncovered: dict[str, np.ndarray]        # pillar blocks for curves with no template
    diagnostics: TemplateDiagnostics
    stamp: RiskStamp                        # as_of, template name+hash, curve names and pillar counts
    @property
    def dv01(self) -> np.ndarray: ...       # values * 1e-4
    def to_pandas(self): ...
```

```python
# engine.py (rewired; the public surface after phase R)
class RiskEngine:
    def __init__(self, program, market, *, template: RiskTemplate | None = None,
                 bp: float = 1e-4, method: str = "auto"): ...
    def pillar_risk(self) -> PillarRisk: ...
    def quote_risk(self, template: RiskTemplate | None = None) -> QuoteRisk: ...   # binds (identity-cached)
    def ladder(self, curve: str) -> CurveRiskReport: ...     # unchanged semantics, now a PillarRisk view
    def dv01(self, curve: str) -> np.ndarray: ...
    def gamma(self, curve: str, *, step: float = 1e-5) -> np.ndarray: ...
```

### 2.8 Algorithms

`RiskTemplate.bind(market)`:

1. Validate rule 1 and rule 7. Collect `curves` = template curves ∪ curves touched by any helper
   (the helper's instrument legs' discount/projection curves; the Jacobian routine reports which
   blocks are non-zero). Build `param_offsets`, `pillar_dates`, base `x` per curve.
2. `J_T = template_jacobian(instruments, curves, market)`.
3. Per curve `c`: `B_c = basis.matrix(x_base_c, node_x_c, jacobian_block=J_T[rows_c, cols_c])`;
   assemble block-diagonal `B`. Rules 2 and 3 (SVD rank per block; tent support check).
4. `M = J_T @ B`; `cond = np.linalg.cond(M)`; rule 4. `lu = scipy.linalg.lu_factor(M.T)`.
5. `A = B @ inv(M)` via `lu_solve` on `M` (not `M.T`): solve `M Y = I`, `A = B Y`. Leverage
   `ℓ_k = max_j |A[j, k]|`; rule 5.
6. Unit sensitivities: for each helper, build its *at-quote unit instrument*
   (`helper.unit_instrument(market)`: swap with `fixed_rate = implied`, notional 1, receiver;
   deposit at implied rate, notional 1; future long 1 contract; bond long face 1); compile per
   kind in one program; `G_T = pillar_risk(...)` `(K, P)`; `s = diag(G_T A)`. Also check the
   off-diagonal of `G_T A` is small relative to `s` (template instruments carry only their own
   quote risk at par); otherwise warn.
7. Residuals `ρ_k = helper.implied(market) − quote_k` (NaN quotes skipped); rule 6.
8. Return `BoundRiskTemplate` holding strong references to the involved `ZeroCurve` objects.

`BoundRiskTemplate.quote_risk(pillar)`:

1. `pillar = pillar.aligned(self.curves, self.pillar_dates)` (raise on pillar mismatch; move
   curves absent from the template into `uncovered`).
2. `X = lu_solve(self._lu, B.T @ G.T)` → `(K, N)`; `R = X.T`.
3. `U = G − R @ J_T`; `rebucketed = G @ B`; parallels; stamp; return `QuoteRisk`.

`BoundRiskTemplate.hedge(risk)` → `−risk.values / s[None, :]`.

`BoundRiskTemplate.base_move(k, size)` → `{c: A[block_c, k] * size}`; used by tests and by
`Scenario` perturbations (`template_quote({"5Y": +10bp})` becomes a geometry-preserving
`key_rate`-style perturbation on the base curve, so the whole scenario machinery of
`portfolio_and_scenarios.md` works on template quotes with no new engine code).

Complexity: one adjoint sweep per template helper kind (`K` instruments batched), one `K×K` LU,
`O(K² N + P N)` per book. For `K = 20`, `N = 10⁴`, this is milliseconds; the Jacobian dominates
and is computed once per market.

### 2.9 Binding, caching, serialization

- **Identity binding.** `matches(market)` is true iff `market.as_of_date == as_of` and
  `market.zero_curve(c) is self._curve_refs[c]` for every spanned curve. `with_curve` creates a
  new `YieldCurve` for the replaced curve and keeps the others by reference, so reprices on the
  same market hit the cache and any curve change (scenario, recalibration) rebinds. Strong
  references make `is` safe (ids cannot be recycled). This avoids the namespace version counter,
  which `with_curve` resets.
- **Geometry-preserving scenarios still rebind.** `J_T` depends on curve values, not only
  geometry; `RiskEngine` rebinds per scenario (cheap) rather than reusing `A`. A
  `rebind_on_scenario=False` escape hatch is deliberately not offered.
- **Serialization.** A template is desk configuration under version control. `TemplateSpec` is
  the JSON-able form: `{"name": ..., "basis": "tent", "limits": {...}, "curves": {"USD.SOFR":
  {"swap": ["1M", "3M", ..., "30Y"]}, "USD.UST": {"bond": ["<cusip>", ...]}, "USD.TY":
  {"treasury_future": ["TYZ6"]}}}`. `RiskTemplate.from_spec(spec, as_of)` builds helpers through
  the existing factories (`swap_helper(rate=nan, tenor=..., as_of=...)`) and, for bonds, a
  `BondUniverse` lookup (phase B). `QuoteRisk.stamp` carries the spec hash so a report is
  traceable to the template version.

### 2.10 Products on the template

| Position | Pillar risk source | Where it lands | Notes |
|---|---|---|---|
| Swap / deposit / FRA | `PricingProgram` adjoint, total role | SOFR template buckets by cashflow dates | identity template reproduces `partial_dv01` |
| SOFR future | `FuturesProgram` pnl gradient (index role) | par-swap buckets overlapping the reference window; `U` carries the futures–swap basis | the CA is exogenous and constant, so it carries no risk |
| Swaption | `SwaptionProgram.risk` per curve (sticky strike) | SOFR buckets through forward and annuity | vega is separate (phase S); no dvol/dforward |
| UST bond | bond program adjoint on `USD.UST` (per-bond spread held fixed) | on-the-run template buckets by maturity; spread risk reported separately as `IDIO` | phase B |
| Treasury future | chained through the CTD (§3.3) on `USD.UST` and the financing curve | on-the-run buckets + a financing line on the SOFR template | switch margin reported; phase F |
| Option on Treasury future | delta × futures gradient (+ discounting) | as the future | phase O |
| Template instrument itself | — | `s_k e_k` | tested |

### 2.11 Tests

`src/finance/tests/unit/pricing/risk/test_template.py` (unit) and
`src/finance/tests/regression/test_risk_template.py` (pinned numbers).

| Test | Asserts |
|---|---|
| `test_identity_template_reproduces_partial_dv01` | template = calibration helpers (square): `R == G @ inv(J)` to rtol 1e-10; `U ≈ 0` |
| `test_template_bump_oracle` | for each `k`: `base_move(k, ±ε)` applied with `with_curve`, central difference of PV equals `R[:, k]` (rtol 1e-6, ε = 1e-6); and every template helper's `implied` moves by `ε·δ_jk` (rtol 1e-8) — this is `J_T A = I` on the real pricer |
| `test_hedge_flattens_template_risk` | book + `hedge()` notionals of at-quote unit instruments: template risk of the combined book ≈ 0; pillar risk of the combined book ≈ `U` |
| `test_futures_strip_shows_as_swap_risk` | production-shaped base curve (deposits + SR3 strip + swaps, `Mixed`); long SR3 strip; all risk in the 3M–24M swap buckets, nothing beyond 2Y above 1% of total; `template_parallel` within 5% of `pillar_parallel`; `U` non-zero and reported |
| `test_too_fine_template_is_refused` | nodes 1M and 2M with no base pillar between → `TemplateError(code="RANK_DEFICIENT", node="USD.SOFR 2M")` |
| `test_k_greater_than_p_is_refused` | `TemplateError(code="K_GT_P")` |
| `test_condition_and_leverage` | identity template: leverage ≈ 1 (≤ 1.5); a deliberately near-duplicate node pair trips `LEVERAGE` before `ILL_CONDITIONED` |
| `test_least_squares_basis_matches_july_harness` | `LeastSquaresBasis` on the 21-FRA / 23-pillar case reproduces `research/risk_reconciliation.reconcile` partials and residuals |
| `test_tent_basis` | partition of unity at every base pillar, flat extension, `np.interp` equivalence, zero column detected |
| `test_multi_curve_block_structure` | FEDFUND curve funded on SOFR; templates on both; a FEDFUND swap shows index risk on the FEDFUND template and funding risk on the SOFR template; bump oracle holds across curves; `M` is block-triangular |
| `test_rebinding` | `matches` false after `with_curve`, true after a reprice; `RiskEngine` rebinds automatically |
| `test_signs_and_units` | receiver swap: `dv01 < 0`; long future: rate `dv01 < 0`; `dv01 == values * 1e-4` |
| regression `test_production_sofr_template` | pinned `R`, `U`, diagnostics for a fixed book on the production-shaped curve (`REGRESSION_UPDATE_EXPECTS=1` refresh, `expects/risk_template.py`) |

### 2.12 Implementation order (phase R)

1. **PR R1 — contract and Jacobian.** `pillar.py` (`PillarRisk`, adapters for the three program
   types, with the `FuturesProgram` origin-mask fix), `jacobian.py` (`template_jacobian`, batched
   by kind; `calibration_jacobian` becomes the wrapper). Tests: adapters agree with today's
   doors; `template_jacobian((target,))` equals `calibration_jacobian` bit-for-bit; multi-curve
   blocks agree with finite differences.
2. **PR R2 — template.** `basis.py`, `template.py` with `bind`, diagnostics, `quote_risk`,
   `hedge`, `base_move`, `from_spec`. Tests 1–9 and 11 above.
3. **PR R3 — engine and retirement.** `RiskEngine` rewire; delete `NumbaRisk.partial_dv01` and
   `AutodiffRisk.partial_dv01`; multi-curve and rebinding tests; regression expects; docs
   (`engine_selection.md` §7, `pricing_architecture.md` risk section, this document's status).

---

## 3. Products

### 3.1 Prerequisites shared by bonds and futures (phase 0)

| Item | Change | File | Why |
|---|---|---|---|
| SIFMA calendar | `StaticCalendar` named `"SIFMA"` with the SIFMA recommended holiday list (data table, year range 2000–2060, early closes ignored); register in `Calendars()`; `"USNY"` alias | `finance/dates/calendars/sifma.py` | UST settlement, coupon payment and futures delivery dates all run on it; today only `no_holidays` exists |
| Act/Act ICMA | new member `DayCountMethod.ActualActualICMA`; `period_fractions` gains keyword `reference` `(ref_starts, ref_ends, frequency)` and computes `days/(ref_days × freq)`; regular periods give exactly `1/freq`; implement `ActualActual` as ISDA at the same time | `finance/dates/day_counts/calculator.py` | UST accrued and coupon amounts |
| 30/360 fix | `Thirty360` → US rule (`D1=31→30`; `D2=31→30 if D1 ∈ {30,31}`; end-of-February rules optional flag); keep the current code as `ThirtyE360` | same | the current implementation is 30E/360 under the wrong name |
| No silent zero accrual | `CommonInstrument.day_count_method` has no default (required) — or validation rejects `Unused` on a leg with a non-zero coupon | `instruments/common_instrument.py` | a leg without a day count accrues zero today |
| Discount spread | `KernelInputs.discount_spread: FloatArray (F,)` default 0; kernels use `DF · exp(−s·τ_pay)` with `τ_pay = days(as_of→pay)/365`; adjoint unchanged (constant) | `kernels/inputs.py`, `kernels/compiler.py`, `engines/numba/program.py`, `engines/jax/program.py` | per-bond z-spread; later CSA/funding spreads |
| Conventions for a bond market | `MarketConventions.swap/deposit/fra` become `| None = None`; add `bond: BondConventions | None` (`coupon_frequency`, `day_count_method`, `settlement_lag`, `calendar`, `bdc`, `roll`, `yield_convention`, `price_quote`); register `("USD", "UST")` with `RateIndex(day_count=ActualActualICMA, fixing_calendar="SIFMA", tenor=None)` | `finance/conventions/` | curve name `"USD.UST"` and bond builders |
| `FuturesProgram` fold | `SofrFuture` priced through `LegPricer`-style compile (`projection_curve_name`, funding alias), keep `FuturesProgram` as a thin wrapper that adds convexity/point value; fix the origin mask | `pricers/futures.py` | one lowering path |

### 3.2 US Treasury bonds (phase B)

**Instrument** — `src/finance/instruments/resolution/bond.py`:

```python
@dataclass(kw_only=True)
class Bond(Priceable):
    """A fixed-coupon bullet bond. The leg is the truth the pricer reads (flat fields + schedule, as everywhere)."""
    leg: CommonInstrument            # Fixed coupon, PrincipalExchange.Final, ActualActualICMA, SemiAnnually, Direction.Backward
    issue_date: Date
    first_coupon_date: Date | None = None     # sets leg.first_regular_accrual_date for short/long first coupons
    security_id: str                          # CUSIP or ISIN; the key into a BondUniverse
    currency: str = "USD"
    index_name: str = "UST"                   # curve family: "USD.UST"
    funding_id: str = "UST"                   # discount curve = the bond curve itself
    settlement_lag: Term = Term(1, BusinessDays)
    settlement_calendar: str = "SIFMA"
    yield_convention: YieldConvention = YieldConvention.Street
    legs -> ((leg, 1.0),)

    @classmethod
    def ust(cls, *, coupon: float, maturity: Date, issue_date: Date, security_id: str, face: float = 100.0,
            first_coupon_date: Date | None = None, registry=default_registry) -> "Bond": ...
    def settlement_date(self, as_of: Date) -> Date: ...
    def coupon_dates(self) -> DateArray: ...          # unadjusted, from the leg schedule
```

Coupon dates are generated backwards from maturity (`Direction.Backward`), roll = maturity
day-of-month with `Roll.EOM` when maturity is a month end, payment adjusted `Following` on SIFMA,
accrual unadjusted (ICMA). `Frequency.SemiAnnually` exists.

**Analytics** — `src/finance/analytics/bonds.py` (pure functions, vectorised across bonds;
`YieldConvention = Street | USTreasury`):

- Accrued (per 100 face): `AI = (C/2) · (settle − prev)/(next − prev)` on actual days of the
  unadjusted coupon period; for an odd first period use the quasi-coupon reference period.
  (31 CFR 356 App. B: `A = [(s − r)/s]·(C/2)`.)
- Street price from yield, `N` remaining coupons, `w = (next − settle)/(next − prev)`:
  `Dirty = Σ_{j=1}^{N} (C/2)/(1 + y/2)^{j−1+w} + 100/(1 + y/2)^{N−1+w}`, `Clean = Dirty − AI`.
  Last coupon period (`N = 1`): simple interest, `Dirty = (100 + C/2)/(1 + w·y/2)`.
  Odd first coupon: the first cash flow is the ICMA-prorated amount at the first coupon date.
- US Treasury convention (auction formula, 31 CFR 356 App. B, II.A–C): simple interest over the
  fractional first period for every `N`:
  `P[1 + (r/s)(i/2)] = (C/2)(r/s) + (C/2)·a_n + 100·vⁿ` (regular/short first), long first:
  `P[1 + (r/s)(i/2)] = [(C/2)(r/s)]·v + (C/2)·a_n + 100·vⁿ`. Needed to reproduce auction
  prices; `Street` is the secondary-market default.
- Yield from price: vectorised Newton with analytic `dP/dy`, start from current yield
  `C/P + (100 − P)/(P·T)`, stop at `|ΔP| < 1e-10`, max 50 iterations, bracket fallback
  (`brentq`) on failure. `dP/dy`, modified duration, convexity closed-form from the same sums.
- Bills (optional, same file): `P = 100(1 − d·r/360)`; investment yield per App. B VI.D.

**Pricing off the curve** — `pricers/bonds.py`:

```python
class BondPricer(LegPricer):                    # registered "Bond"
    def compile(self, bonds, *, backend=None) -> BondProgram: ...

class BondProgram(PricingProgram):
    spreads: np.ndarray                         # (N,) per-bond z-spread → discount_spread column
    def reprice(self, market) -> BondResult     # dirty PV, clean PV, accrued, curve-implied yield
    def implied_yields(self, market) -> np.ndarray
```

`BondResult` adds `accrued` (from the analytics, not the kernel — the kernel has no accrual-to-date
column yet; `portfolio_and_scenarios.md` §5 plans one) and `clean_pv`.

**Spread model.** Each bond carries a spread `s_i` such that `PV_curve(z, s_i) = market dirty
price`. `fit_bond_spreads(bonds, prices, market) -> np.ndarray` (vectorised Newton; the
derivative is `−Σ τ_j · cash_j · DF_j e^{−s τ_j}`). Risk holds `s_i` fixed ("curve + idiosyncratic");
`∂PV/∂s_i` is reported as the bond's `IDIO01`.

**Calibration helper** — `pricing/calibration/instruments.py`:

```python
@dataclass
class BondHelper:
    bond: Bond
    quote: Quote                 # QuoteKind.Yield (decimal) or QuoteKind.CleanPrice (per 100); price converted to yield once
    curve: str                   # "USD.UST"
    spread: float = 0.0          # held fixed during the fit
    weight: float = 1.0          # on-the-runs vs off-the-runs
    pillar_date -> bond.leg.maturity
    implied(market) -> float     # curve-implied street yield (price off the curve with `spread`, then yield)
    residual(market) -> implied − quote_yield
    numba_gradient(market) -> (P,)   # ∂y/∂z = (∂P_curve/∂z)/(∂P/∂y): implicit-function rule
```

`QuoteKind` gains `CleanPrice = 5`, `FuturesPrice = 6`.

**UST curve fit.** `CurveDefinition(currency="USD", index_name="UST", interpolation=Mixed(Linear(), Cubic(), k) or LogCubicDF)`.
`CurveCalibrator` changes: `knots: Sequence[Date | Term] | None = None` (explicit pillar set;
helpers no longer need to be one per node), `GlobalSolver` accepts `K > P` (scipy
`least_squares` already does), weights from `Quote.weight`, and an optional Tikhonov penalty on
second differences of instantaneous forwards (`regularization: float = 0.0`). After the fit,
`fit_bond_spreads` gives each bond its spread; the OTR bonds are the natural template nodes (their
spreads are ≈ 0 by construction if they are knots).

**Oracles and tests** (`tests/unit/analytics/test_bonds.py`, `tests/parity/test_bonds_oracle.py`):
App. B worked examples (price/yield/accrued for regular, short and long first coupons; the
`N = 1` simple-interest case); QuantLib (`pytest.importorskip("QuantLib")`,
`ql.BondFunctions.yield`, `ActualActual(ISMA)`, `ql.UnitedStates(GovernmentBond)`) and rateslib
(`FixedRateBond(spec="us_gb")`) parity on a basket of 20 real CUSIPs; vectorised yield solver
agrees with scalar `brentq`; curve-priced bond at `spread = fit` reprices to market; identity
template on the OTRs; a 7-year off-the-run shows risk split between the 5Y and 10Y OTR buckets by
tents and `U` carries its curvature.

### 3.3 Treasury bond and note futures (phase F)

**Contract table** — `src/finance/instruments/resolution/treasury_futures.py`, data, not code
(the row values below are from CME's published specifications and must be re-verified against
the current CBOT rulebook chapters 18–21/26 when implemented; they are inputs, not derived facts):

| Code | Deliverable remaining term (from first day of delivery month) | Original term | CF rounding of `z` | Face | Last trading day | Last delivery day |
|---|---|---|---|---|---|---|
| ZT (TU) 2-Year | ≥ 1y9m and ≤ 2y | ≤ 5y3m | to the month | $200,000 | last business day of month | 3rd business day after last business day of month |
| Z3N 3-Year | per rulebook | per rulebook | to the month | $200,000 | as 2-Year | as 2-Year |
| ZF (FV) 5-Year | ≥ 4y2m | ≤ 5y3m | to the month | $100,000 | as 2-Year | as 2-Year |
| ZN (TY) 10-Year | ≥ 6y6m and < 8y | ≤ 10y | to the quarter | $100,000 | 7th business day before last business day of month | last business day of month |
| TN Ultra 10-Year | ≥ 9y5m and ≤ 10y | ≤ 10y | to the quarter | $100,000 | as 10-Year | as 10-Year |
| ZB (US) Bond | ≥ 15y and < 25y | — | to the quarter | $100,000 | as 10-Year | as 10-Year |
| UB Ultra Bond | ≥ 25y | — | to the quarter | $100,000 | as 10-Year | as 10-Year |

Point value is `face / 100` per price point ($1,000 for $100k face). `DeliveryRule` holds the
two date rules; `assumed_delivery` defaults to the last delivery day (positive carry → deliver
late), overridable.

**Conversion factor** (CME, *Calculating U.S. Treasury Futures Conversion Factors*, verified
2026-10-13 against the CME document):

```
factor = a · [ coupon/2 + c + d ] − b           rounded to 4 decimals
coupon : annual coupon, decimal
n      : whole years from the first day of the delivery month to maturity
z      : whole months beyond n, rounded DOWN to the nearest quarter (ZN, TN, ZB, UB) or month (ZT, Z3N, ZF)
v      : z            if z < 7
         z − 6        otherwise            (quarter contracts: 9 → 3)
a      : 1 / 1.03^(v/6)
b      : (coupon/2) · (6 − v) / 6
c      : 1 / 1.03^(2n)        if z < 7
         1 / 1.03^(2n+1)      otherwise
d      : (coupon/0.06) · (1 − c)
```

Worked example from the CME document (pinned as a test): Dec-2008 2-Year contract, the 1½% of
31-Oct-2010: remaining 1y 10m 30d → `n = 1`, `z = 10`, `v = 4`, `a = 0.980487`, `b = 0.002500`,
`c = 0.915142`, `d = 0.021215`, factor `0.922939 → 0.9229`.

```python
def conversion_factor(coupon: float, maturity: Date, delivery_month: tuple[int, int], *, rounding: CfRounding) -> float
def deliverable(bond: Bond, contract: str, delivery_month: tuple[int, int]) -> bool
def basket(universe: BondUniverse, contract: str, delivery_month) -> tuple[Deliverable, ...]   # (bond, cf)
```

**Instrument**:

```python
@dataclass(frozen=True, kw_only=True)
class TreasuryFuture:
    contract: str                               # "ZN"
    delivery_month: tuple[int, int]
    basket: tuple[Deliverable, ...]             # (bond, conversion_factor); supplied or built from a universe
    delivery_date: Date                         # assumed; default from DeliveryRule
    currency: str = "USD"; index_name: str = "UST"; funding_id: str = "STDCSA"   # financing (repo proxy) curve
    repo_spreads: tuple[float, ...] | None = None   # per basket bond, specials; default 0
    point_value: float; contracts: float = 1.0; entry_price: float | None = None
    net_basis: float = 0.0                      # price points; exogenous delivery-option value, like SofrFuture.convexity
```

**Pricing** — `pricers/treasury_futures.py`, `TreasuryFuturesProgram`:

For basket bond `i`, with settlement `t_s` (regular way from as-of), delivery `t_d`, financing
DFs `D(t) = DF_fund(t)/DF_fund(t_s)` (optionally times `e^{−spread_i (t − t_s)}`):

```
S_i        = dirty price at t_s           (curve: bond program at spread s_i; or market price if supplied)
Fwd_i      = [ S_i − Σ_{t_s < t_j ≤ t_d} c_j · D(t_j) ] / D(t_d) − AI_i(t_d)     clean forward
conv_i     = Fwd_i / CF_i
F_model    = min_i conv_i − net_basis                                            CTD = argmin
PV         = (F_model − entry_price) · point_value · contracts                   margined: no discounting
```

Reports per basket bond: gross basis `S_i^clean − F_mkt·CF_i`, net basis (`conv_i − F_mkt`),
implied repo (solve the financing rate that makes `conv_i = F_mkt`), DV01 of the future
(`DV01_CTD / CF_CTD`), and the **switch margin** `conv_(2) − conv_(1)` in price points plus the
yield move that would flip the CTD (linearised).

**Risk.** `∂F/∂z` through the CTD only: `∂conv_CTD/∂z_UST = (∂S/∂z_UST)/(CF · D(t_d))` from the
bond program adjoint; `∂conv/∂z_fund` through the `D(·)` factors via `log_df_affine` rows on
the financing curve. Hand-chained in phase F (about 60 lines), subsumed by the `min_basket` op
in phase E. `pillar_risk` returns blocks on both curves; the template shows on-the-run buckets
and a SOFR financing line. The runner-up's risk is reported alongside so a near switch is visible.

**Helper** — `TreasuryFuturesHelper(future, quote=Quote(price, FuturesPrice), curve="USD.UST")`:
implied = `F_model`, residual in price points; `numba_gradient` from the program. This lets a
futures strip be a UST-curve input and lets `ZN`/`ZB` be template instruments ("risk in TY
equivalents").

**Oracles and tests**: CME published conversion factors for the current baskets (pinned table);
rateslib `BondFuture(calc_mode="ust_long"/"ust_short").cfs`, `ctd_index`, `dlv()` parity; a
synthetic basket where the CTD switch is forced by a rate move, asserting the switch margin and
the discontinuity in `∂F/∂z` are reported; futures template bump oracle; "futures position hedged
with its CTD DV01/CF is flat on the UST template".

### 3.4 Options on Treasury futures (phase O)

CME Treasury options are American-style and exercise into the futures contract (verified on the
CME Ultra 10-Year options specification); their premium is settled as premium-style (full
premium at trade, the CME Clearing default). Model:

```python
@dataclass(frozen=True, kw_only=True)
class TreasuryFutureOption:
    future: TreasuryFuture; strike: float; expiry: Date; call: bool
    exercise: Exercise = Exercise.American
    premium: PremiumStyle = PremiumStyle.Upfront
    model: OptionModel = OptionModel.Black        # Black-76 on F (lognormal price vol) | Bachelier (normal price vol)
    vol_name: str | None = None                   # VolNamespace key, e.g. "USD.ZN.PRICEVOL"
    contracts: float = 1.0
```

European pricing (phase O): `unit = black(F, K, T, σ)` or `bachelier(...)` from the existing
kernels (`engines/numpy/option.py`, Numba twins), `T = (expiry − as_of)/365`,
`PV = DF_fund(t_settle, T_expiry) · unit · point_value · contracts` for `Upfront` premium (no
discounting for `FuturesStyle`). Risk: `delta · ∂F/∂z` (sticky strike) plus the discount term;
vega in the surface's units; gamma from the kernel. The `VolSurface` protocol is reused with
`tenor = 0` and `forward = F`; a `PriceVolSurface` implementation keyed per contract.

American exercise is **not approximated silently**: with `exercise=American` the pricer raises
unless `allow_european_approximation=True` is passed, until the CRR binomial on `F` (zero drift,
`n_steps` parameter, Richardson extrapolation) lands. For short-dated near-the-money options the
early-exercise premium is small; for deep in-the-money puts it is not, which is why the flag is
explicit.

### 3.5 Swaptions (phase S, production gaps)

Already correct for a European SOFR swaption with a flat vol. Gaps: a `NormalVolCube(expiries,
tenors, strike_offsets, vols)` with bilinear interpolation on `(expiry, tenor)` and linear on
strike offset (SABR as a later surface implementation behind the same protocol); settlement style
(`Physical` vs `CashCollateralised` — same annuity for SOFR, kept as an enum for later cash-price
settlement); premium date; vega bucketing to cube nodes (the same tent machinery in two
dimensions, `VolTemplate`); `Swaption` gains `legs` so it can route through `LegPricer`
infrastructure for the underlying.

### 3.6 SOFR futures (small items)

Fold `FuturesProgram` into the `LegPricer` lowering (§3.1); fix the origin mask in `risk`;
implement the `ConvexityModel` seam of `sofr_futures.md` §5 with Ho-Lee first
(`CA = ½σ²·T₁·T₂`) and Hull-White second; make `SofrFuture` `Priceable` and register it.

---

## 4. Engine: the op graph (phase E)

This section is the implementation plan behind `engine_op_graph_plan.md`. Decisions already
taken there stand: **not a tape, not JAX at runtime**, hand-written VJPs over a small op
vocabulary, JAX as the test oracle. The additions here are the exact representation, the op
vocabulary extended to bonds, futures and options, the reverse sweep's row map (so per-instrument
gradients cost one sweep, not `N`), the contract with the RiskTemplate, and the migration gates.

### 4.1 Representation

```python
# src/finance/pricing/engines/graph/program.py
@dataclass(frozen=True)
class Slot:
    name: str; shape: tuple[int, ...]; dtype: np.dtype
    kind: Literal["param", "static", "value"]      # param: z blocks, vol params; static: compiled columns; value: op outputs
    rows: str | None                                # row map for value slots: "flow" | "obs" | "leg" | "instrument" | "scalar"

@dataclass(frozen=True)
class OpInstance:
    kind: str                                       # registry key
    inputs: tuple[str, ...]                         # slot names
    outputs: tuple[str, ...]
    static: dict[str, np.ndarray | float | int]     # offsets, masks, CSR table, τ, point values

@dataclass(frozen=True)
class GraphProgram:
    slots: dict[str, Slot]
    ops: tuple[OpInstance, ...]                     # topological order; reverse order is the sweep
    params: ParamLayout                             # curve name -> slice of z; vol surface name -> slice (phase S)
    row_maps: dict[str, np.ndarray]                 # "flow"->instrument, "obs"->instrument, "leg"->instrument
    outputs: dict[str, str]                         # measure name -> slot ("instrument_pv", "flow_pv", "par_rate", "futures_price", ...)
    queries: QueryTable                             # today's _QueryTable, promoted
```

The compiler (`kernels/compiler.compile_portfolio` + `NumbaProgram.__init__`'s query dedup)
emits a `GraphProgram` once per book; `KernelInputs` stays as the columnar input the compiler
reads. Products add ops, never new engines.

### 4.2 Op vocabulary with reverse rules

Registry `engine_registry[(kind, Backend)] = OpImpl(primal, vjp)`. Cotangent of `x` is `x̄`.

| kind | primal | vjp (accumulate) | rows |
|---|---|---|---|
| `df` | `DF_q = exp(c_q + Σ_p coef_{qp} z_p)` (CSR) | `z̄_p += DF̄_q · DF_q · coef_{qp}` scattered **by query id** into the `(rows, P)` output | query |
| `df_spread` | `DF_f = DF_q(f) · exp(−s_f τ_f)` | `DF̄_q += DF̄_f · exp(−s τ)`; `s̄_f += −τ DF_f DF̄_f` | flow |
| `fwd_simple` | `r = (DF_s/DF_e − 1)/τ` | `DF̄_s += r̄/(τ DF_e)`; `DF̄_e += −r̄ DF_s/(τ DF_e²)` | flow / obs |
| `fix` | overlay historical fixings on `r` (mask) | `r̄ *= 1 − mask` | obs / flow |
| `prep_obs`, `shape_float`, `shape_period` | `kernels/shaping.*` | elementwise masks; floor/cap: `1[inside]` subgradient | obs / flow |
| `compounded` | `R = (Π(1 + r_m w_m) − 1)/W` segmented | `r̄_m += R̄ · Π/(1 + r_m w_m) · w_m / W` (segmented prefix products, as the current adjoint) | flow |
| `averaged` | `R = Σ r_m w_m / W` | `r̄_m += R̄ w_m / W` | flow |
| `cash` | `cash_f = N_f · r_f · τ_f` | `r̄_f += cash̄_f N τ` | flow |
| `dcf` | `pv_f = cash_f · DF_f · sign_f` | `cash̄ += pv̄ DF sign`; `DF̄ += pv̄ cash sign` | flow |
| `reduce` | segmented sum `flow → leg → instrument`, or weighted sum to scalar | broadcast | leg / instrument / scalar |
| `par_rate` | `p = −PV_float / PV_fixed_unit` | quotient rule | instrument |
| `black`, `bachelier` | value, delta, vega (existing kernels) | `F̄ += v̄ Δ`; `σ̄ += v̄ vega`; `ann̄ += v̄ unit` | instrument |
| `bond_fwd` | `Fwd = (S − Σ c_j D_j)/D_d − AI` | `S̄ += F̄/D_d`; `D̄_j += −F̄ c_j/D_d`; `D̄_d += −F̄ (S − Σ c D)/D_d²` | instrument (basket bond) |
| `min_basket` | `F = min_i x_i / CF_i` over a segment | `x̄_{argmin} += F̄ / CF` (subgradient; the runner-up index and margin are side outputs) | instrument |
| `yield_from_price` | `y = Y(P)` (Newton inside the primal) | `P̄ += ȳ / (dP/dy)` (implicit function; `dP/dy` is a primal side output) | instrument |
| `affine` | `y = a·x + b` (point values, convexity, net basis, bp scaling) | `x̄ += a ȳ` | any |

The reverse sweep's **row map** is what keeps per-instrument gradients at one sweep: every value
slot belongs to exactly one instrument (flows, observations, legs, basket bonds, options), so the
`df` VJP scatters `DF̄_q · DF_q · coef` into row `instrument_of(q)` of a `(rows, P)` buffer. A
scalar seed (book PV, one par rate) uses a one-row map; a per-leg request uses the leg map. There
is no `(F, Z)` intermediate anywhere; `cashflow_gradient` becomes an on-request sweep with the
flow row map (only reports ask for it).

### 4.3 Runners

`NumbaRunner`, `NumpyRunner`, `JaxRunner` execute the same op list. Per-op Numba kernels replace
the fused `_execute`; the Python dispatch of ~20 ops per sweep costs tens of microseconds and
loses some fusion. **Gate:** primal reprice of the 10k-swap benchmark within 2× of today's fused
kernel; adjoint time ≤ today; adjoint memory `O(F + NNZ + N·P)` instead of `O(F·Z)`. If the
primal gate fails, fuse the hot path (`df → fwd → shape → compounded → cash → dcf`) into one
composite op with its own VJP — the representation allows it, and parity tests are per op either
way. `JaxRunner` builds a `jax.jit` function from the op list for oracle tests only (never in
production; `jax_enable_x64` set in the test harness).

### 4.4 The single risk entry

```python
# src/finance/pricing/risk/engine.py (phase E form)
def risk(program: GraphProgram, market: MarketContext, *, seed: str = "instrument_pv",
         rows: Literal["instrument", "leg", "flow", "scalar"] = "instrument",
         wrt: tuple[str, ...] = ("curves",)) -> PillarRisk | VolRisk
```

`PillarRisk` (§2.7) is the output for curve parameters; `VolRisk` for vol slots (phase S). The
`pillar_risk()` adapters of phase R become this one function; `SwaptionProgram.risk`,
`FuturesProgram.risk`, `NumbaRisk`, `AutodiffRisk` and `RiskEngine._adjoint_ladder` are retired;
`Sensitivities` (bump) stays as the universal cross-check. `template_jacobian` becomes
`risk(template_program, market, seed="quote")` with the quote measures as ops (`par_rate`,
`fwd_simple`, `yield_from_price`, `min_basket`).

### 4.5 Migration order and gates

1. **E1 — registry and parity harness.** `OpImpl(primal, vjp)`; register the existing numpy and
   Numba kernels as primal-only; a per-op parity test template (numpy vs Numba vs JAX finite
   differences and `jax.vjp`).
2. **E2 — `GraphProgram` emitted by the compiler** (no engine change yet): slots, ops, row maps,
   query table. Test: the op list evaluated by `NumpyRunner` equals `compiler.reprice` bit-for-bit
   on the regression suite (`test_leg_shapes`).
3. **E3 — VJPs and the reverse sweep** in `NumbaRunner`; `risk()` returns `PillarRisk`. Gates:
   equals `NumbaAdjointResult.instrument_gradient` to 1e-12 relative on every existing adjoint
   test; the memory/time gate of §4.3.
4. **E4 — retire the doors.** `RiskEngine` and the RiskTemplate consume `risk()`; delete
   `NumbaRisk`, `AutodiffRisk`, `SwaptionProgram.risk`, `FuturesProgram.risk`, the dense `(F, Z)`
   arrays, `engines/jax/program.py`'s monolith (replaced by `JaxRunner`). `Sensitivities` stays.
5. **E5 — products as ops.** `black`/`bachelier` with a vol slot (swaption inside the IR),
   `bond_fwd`, `min_basket`, `yield_from_price`; the hand-chained risk code of phases B/F/O is
   deleted. Gate: product regression expects unchanged.
6. **E6 — `JaxRunner` as the oracle**, `engines/jax/calibration.py` deleted (the JAX calibration
   Jacobian is `jax.jacobian` over the runner).

---

## 5. `leg_group` and the cashflow report edge

Joined legs (`finance.instruments.join_legs`) are separate legs to the engine and separate rows
in `CashflowReport`. A trader wants one row per economic leg.

- `LegSpec.group: int` and `KernelInputs.leg_group: IntArray (L,)` (default = leg index).
  `LegPricer.compile` assigns one group per `legs` entry that came from a join (the join returns
  its pairs tagged; `join_legs` gains `group: int | None`).
- `CashflowReport` gains `leg_group`, `instrument`, and the per-period compile-time metadata the
  compiler already computes and currently drops: `coupon_type` (`int8`, the governing event's
  `CouponType`), `spread`, `index_floor`, `cap`, `floor`, `margin`. These are captured into
  `KernelInputs` as report-only columns (the kernels ignore them) — the "capture, don't lower"
  rule of `engine_op_graph_plan.md` §Reporting edges.
- `PricingResult.cashflows(net=True)` nets flows by `(instrument, leg_group, pay_date)` with
  `np.add.reduceat` over the group offsets (joined legs are adjacent, so groups are contiguous).
  The leg-level `CouponType` reported for a group is `Custom` when any member leg is `Custom` or
  when members differ, else the shared type.
- `regression/test_leg_shapes.py` example 10 (monthly-then-quarterly fixed) gains a netted-report
  assertion: one row per pay date, PVs equal to the sum of the two legs' rows.

---

## 6. Cleanup inventory (keep / delete / fold)

From the read-only sweep. Evidence is the grep counts and file references the sweep produced;
"prod" means non-test `src/` outside the defining package.

### 6.1 Delete (no consumers, or superseded)

| Item | Where | Evidence | Action |
|---|---|---|---|
| Stale bytecode directories | `src/finance/markets/conventions/`, `src/finance/pricing/conventions/`, `src/finance/coupons/`, `src/finance/tests/unit/coupons/`, `instruments/enums/__pycache__/leg_type*.pyc`, `markets/curves/_curve_impl/__pycache__/interpolators*.pyc` | untracked `__pycache__` only; sources moved in `8a35eba` / deleted in `1bf5c1d`, `d46bb7b` | `rm -r` from disk (not a git change); add `find src -name __pycache__ -prune -exec rm -r {} +` to `bin/` |
| `src/finance/pricing/kernels/foo.txt` | tracked, 144 lines, a legacy `fixedfloatswap` sketch | commit `92f6f72` | `git rm` |
| `ProductKind` | `pricing/types.py:45` | 0 consumers | delete; `pricer_registry` keys by class name |
| `AmortizationType` and `CommonInstrument.{original_notional, amortization_type, amortization_start, amortization_end}` | `instruments/enums/amortization.py`, `common_instrument.py:71-74` | never read | delete now; when amortization lands, reintroduce as `amortization_type` generator **or** `Custom` + `notional_schedule`, mirroring the coupon design (settled 2026-10-13) |
| `ScheduledNotional`, `RateDependentNotional`, `NotionalProvider.is_rate_dependent` | `kernels/inputs.py:58-85` | only re-exported | delete; `StaticNotional` stays until the amortization design above |
| `MarginTreatment.Flat`, `MarginTreatment.Unused` | `instruments/enums/margin.py` | 0 uses; no kernel branch handles them (`compiler.py:291,307`) | delete both; validation rejects anything but `Inclusive`/`Exclusive` |
| `SofrFuture.rate_kind` | `resolution/futures.py:79` | 0 readers, lazy import | delete |
| `CouponSchedule.add_event` | `schedules/coupon_schedule.py:103` | 0 callers | delete |
| `PricingRequest` and the `requests` argument of `Priceable.__call__` | `instruments/priceable.py:25,41` | only `"cashflows"` has an effect | replace with `with_cashflows: bool` |
| `NumbaRisk = None` fallback and `RiskEngine._resolve`'s use of it | `risk/__init__.py:6-9`, `risk/engine.py:65-72` | cannot trigger (numba imported lazily) | delete the fallback; `_resolve("auto")` checks `importlib.util.find_spec("numba")` |
| `AutodiffRisk._total_z_pv` | `risk/autodiff.py:117` | never called | delete |
| `NumbaRisk.partial_dv01`, `AutodiffRisk.partial_dv01` | `risk/adjoint.py:87`, `risk/autodiff.py:156` | replaced by the identity `RiskTemplate` (§2) | delete in PR R3 |
| `engines/rust/` | docstring-only stub; no Cargo/PyO3 anywhere | `Backend.Rust` tested only as "absent" | delete the package and the enum member; the op registry readmits a backend when one exists |
| `markets/paths/` (`SinglePath`, `FlatPath`, `InvertedPath`) | no consumers outside its test; `InvertedPath.__post_init__` mutates the caller's array | listed "NOT ported" in `curve_seam_porting.md` | delete with its test and the `markets/__init__` re-exports |
| `ccy_indices/amrs/usd.py::DEFINED` | `ccy_indices` | test-only | delete (`FundingIndex` deleted in PR C; the discount key `CCY.NAME` is the funding model) |
| `DayCountMethod.{ThirtyE365, Bus252}` | enum members without implementation | 0 uses | delete; `ThirtyE360`, `ThirtyE360ISDA` get implementations with the 30/360 fix (§3.1) |
| `DayCountMethod.is_supported`, `SupportedIntEnum.is_supported` call sites | always true, never called | — | delete |
| `research/pricing_libs/numba_kernel.py`, `numba_perf.py` lines calling `NumbaProgram.from_program` | removed in `cfd013d` | fail at runtime | fix or delete the scripts |
| `src/common/factories/` | empty package | no consumers anywhere | delete |
| `src/finance/tests/unit/dates/test_enums.py` | 0-byte test module (commit `db7a6ed`) | collects nothing | delete or write the enum tests |
| Declared-but-unused pytest markers `market_data`, `instruments`, `calculators`, `slow`; `IntegrationTest` base never subclassed | `pyproject.toml`, `common/testing/__init__.py` | 0 uses | delete the markers; keep `IntegrationTest` only if the parity oracles of §3.2–3.3 use it (they should: `tests/parity/` = integration) |

### 6.2 Fold (two things that should be one)

| Item | Fold into | Notes |
|---|---|---|
| `FuturesProgram` lowering (`leg_spec` directly, no `projection_curve_name`) | `LegPricer` path (§3.1) | keeps convexity/point value as a thin layer |
| Six risk doors | `pillar_risk()` adapters (phase R) → `risk()` (phase E) | `Sensitivities` survives as the bump oracle |
| `calibration_jacobian` | `template_jacobian` (phase R) | wrapper kept for `CurveCalibrator` until phase E |
| `engine_registry` (primal-only, no production caller: `compiler.reprice` and `NumbaProgram` import kernels directly) | op registry `(kind, Backend) → OpImpl` (phase E1) | the registry becomes load-bearing |
| `STDCSA` constant vs seven `"STDCSA"` literals | one constant in `resolution/resolver.py`, imported everywhere | `futures.py:73,113`, `swaption.py:39`, `calibration/instruments.py:214,228,242,263` |
| `Priceable` for `SofrFuture`, `Swaption`, `Bond`, `TreasuryFuture`, `TreasuryFutureOption` | register all in `pricer_registry` | today only Swap/Deposit/Fra are registered |
| Duplicate `class Legs` wrappers in `tests/regression/test_leg_shapes.py:52` and `research/one_leg_examples.py:50` | a `finance.instruments.Legs` product (signed legs, `Priceable`) | the natural home of `join_legs` output |

### 6.3 Fix (latent defects found by the sweep)

| Defect | Where | Fix |
|---|---|---|
| `FuturesProgram.risk` recomputes DFs without the `pay_dates >= origin` mask | `pricers/futures.py:78-81` | use the primal's `df` (the adjoint result carries it) |
| `Thirty360` is 30E/360 | `dates/day_counts/calculator.py:46-68` | §3.1 |
| `day_count_method=Unused` accrues zero silently | `instruments/common_instrument.py:61` | §3.1 (required field or validation) |
| `MarginTreatment.Flat` silently drops the spread on compounded legs | `compiler.py:291,307` | deleted above |
| `VolNamespace` docstring claims `snapshot()`; `MarketContext.vols: object | None` | `markets/vols/__init__.py`, `markets/context.py:27` | type it `VolNamespace | None`; add `snapshot()`; `with_vols()` sibling of `with_curve` |
| `CurveInterpolator` has no `Mixed` entry but `CurveDefinition` accepts a `Mixed` instance; `is_local` treats `Mixed(Linear, Linear)` as global | `markets/curves/types.py`, `calibration/calibrator.py` | `is_local` inspects `Mixed` components; document that the parse table is for strings only |
| Stale comments: `types.py:17` ("numba/rust are reserved seams"), `types.py:58-59` (bachelier/black "not implemented"), `engine_selection.md:125` ("exact JAX ∂implied/∂z" — it is Numba), `research/pricing_libs/jax_basis_risk.py:84,101` ("the JAX hook") | — | edit |
| `CommonInstrument.first/last_regular_accrual_date` are plumbed but never set | `legs.py:61` | **keep** — bonds with odd first coupons use them (§3.2) |
| `PrincipalExchange.Initial` unused | `enums/principal.py` | **keep** — needed for a funded-leg representation (`Both` on deposits exists) |
| JAX not declared in `pyproject.toml`; x64 never enabled in `src/` | packaging | add an `oracle` extra (`jax`), enable x64 in the test harness only (JAX is test-only by decision) |
| `quant_toolkit/__init__.py` is `from finance import *` over an empty `finance/__init__.py`, so the published package exports nothing | `src/quant_toolkit/__init__.py`, `src/finance/__init__.py` | curated `__all__` in `finance/__init__.py` (dates, instruments, markets, pricing, calibration, risk, template) — the public API surface this document defines |
| `common/testing/__init__.py:30-32` assigns `df.copy_columns = …` on the input frame (typo for `df_copy.columns`) | `common/testing/__init__.py` | fix; it only affects the markdown helper's `else` branch |
| `pytest`/`pytest-cov` are runtime dependencies; `tabulate` (needed by `DataFrame.to_markdown` in `common/testing`) undeclared; ruff lint keys under `[tool.ruff]` (deprecated placement); `--cov` omits `quant_toolkit_xl` although it is in `testpaths` | `pyproject.toml` | move test deps to `dev`, add `tabulate` to `dev`, move lint keys to `[tool.ruff.lint]`, add `--cov=quant_toolkit_xl` |
| `quant_toolkit_xl/README.md:118` says `interpolation` is the 5th positional argument of `qBuildCurve`; it is the 3rd (`functions.py:20`) | README | edit |

### 6.3b Documentation that no longer matches the code

| Doc | Stale statement | Fix |
|---|---|---|
| `pricing_architecture.md:195` | "`coupons/calculators.py` and `coupons/rates.py` consume these columns" — both deleted in `1bf5c1d`; same stale name in `tests/unit/pricing/test_shaping.py:3-4` | point at `kernels/compiler.py` and `engines/numpy/rates.py` |
| `sofr_futures.md` | names `ResolvedSofrFuture` and `ConvexityModel` (neither exists; the contract is `SofrFuture`, the seam is `FuturesHelper.convexity: float | callable`); status line calls `Roll.IMM` a follow-up but `Roll.IMM = -2` exists and is tested | rename in the doc; update the status line |
| `curve_seam_porting.md` | references `InterpolatedSegment`, `DFLogLinear`, `DFLogCubic`, `build_interpolator`, the deleted `_curve_impl/interpolators.py`, the old `RateLinear = -1…` values and the old `ZeroCurve.__init__(node_dates, …)` signature | mark the document historical (porting guide for the pre-`Line1d` curves) or rewrite against `Line1d` |
| `curve_line1d_refactor_plan.md:3` | says PR 2/3 are on `feature/carlos/line_curve_refactor`; merged in PR #19 (`9d7599c`) | status line |
| `engine_op_graph_plan.md:88,105` | refers to "the math-upgrades doc", which is not in the repository | replace with a pointer to this document (§2 is item 1, §4 is item 2) |
| `engine_selection.md:125` | "the exact JAX ∂implied/∂z" — the default `jacobian_backend` is Numba | edit |
| `portfolio_and_scenarios.md` §5 | `MARKET_QUOTE_01` sourced from `partial_dv01 ∘ CalibrationResult.jacobian` | after phase R: `RiskTemplate.identity(...)` / `QuoteRisk` |

### 6.4 Keep as is (looked like debt, is not)

- `finance/dates` `dt_based` / `np_based` / `_core` trio and the JAX engine: the deliberate
  reference ladder (simple algorithm is the truth, fast ones must reproduce it).
- `Swaption.european` with only test callers: it is the user-facing builder.
- `imm_date`, `next_quarterly_imm`, `roll_spot`, `resolve_conventions` exports: small, used by
  builders and research.
- The `ruff format`-unclean tree: intentional compact formatting; match surrounding style.

---

## 7. Order of work

| # | PR | Phase | Gate |
|---|---|---|---|
| 1 | Cleanup §6.1 deletions + §6.3 stale comments + `FuturesProgram` origin-mask fix | 0 | suite green; no behaviour change except the futures risk mask |
| 2 | SIFMA calendar; Act/Act ICMA + ISDA; 30/360 fix; required day count; `discount_spread` column | 0 | day-count tests vs QuantLib oracle; bit-identical PVs on the regression suite for Act/360 books |
| 3 | `leg_group` + report columns + netted cashflows | 0 | example 10 netted assertion |
| 4 | R1 `PillarRisk` + `template_jacobian` | R | parity with existing doors and `calibration_jacobian` |
| 5 | R2 `RiskTemplate` | R | §2.11 tests 1–9, 11 |
| 6 | R3 `RiskEngine` rewire, delete `partial_dv01`s, regression expects, docs | R | §2.11 tests 10, 12; `research/risk_reconciliation.py` reproduced via `LeastSquaresBasis` |
| 7 | Bond conventions entry, `Bond`, analytics, oracle tests | B | App. B examples; QuantLib/rateslib parity |
| 8 | `BondPricer`/`BondProgram`, spreads, `BondHelper`, calibrator `knots`/weights/`K > P` | B | curve-priced bonds reprice to market at fitted spreads; OTR identity template |
| 9 | Treasury futures: contract table, CF, basket, program, risk, helper | F | CME factor table; rateslib parity; switch-margin test |
| 10 | Options on Treasury futures (European), `PriceVolSurface` | O | Black/Bachelier parity; bump oracle |
| 11 | E1–E3 op graph (registry, program, sweep) | E | §4.5 gates |
| 12 | E4–E6 retire doors, products as ops, JAX runner | E | regression expects unchanged |
| 13 | Swaption production gaps | S | cube interpolation tests; vega bucketing bump oracle |
| 14 | American exercise for Treasury options (CRR) | O | binomial converges to Black for European; early-exercise premium ≥ 0 |

Phases B, F, O and S can interleave with E; only the retirement steps E4/E5 touch product code
and they are mechanical once `risk()` exists.

---

## Appendix A — Formulas pinned by tests

**Treasury price/yield (31 CFR 356, Appendix B).** Variables: `P` price per 100 (rounded to six
places), `C` annual coupon per 100, `i` yield (semiannual-pay nominal), `n` full semiannual
periods from issue/settlement to maturity, `r` days from settlement to the next coupon, `s` days
in that coupon period, `v = 1/(1 + i/2)`, `a_n = (1 − vⁿ)/(i/2)`.
Regular and short first coupon: `P[1 + (r/s)(i/2)] = (C/2)(r/s) + (C/2)a_n + 100vⁿ`.
Long first coupon: `P[1 + (r/s)(i/2)] = [(C/2)(r/s)]v + (C/2)a_n + 100vⁿ`.
Accrued: `A = [(s − r)/s](C/2)`. Bills: `P = 100(1 − d·r/360)`; investment rate (≤ half year)
`i = [(100 − P)/P](y/r)`.

**Street convention** (secondary market, the library default): compound fractional-period
discounting `(1 + y/2)^{j−1+w}` with `w = r/s`, and simple interest `(100 + C/2)/(1 + w·y/2)` in
the final coupon period.

**Conversion factor**: §3.3, with the CME worked example `0.9229`.

**Cheapest to deliver**: `conv_i = (clean forward_i)/CF_i`, CTD = argmin; implied repo for bond
`i` at futures price `F`: the simple Act/360 rate `ρ` solving
`(F·CF_i + AI_i(t_d)) + Σ c_j (1 + ρ·(t_d − t_j)/360) = (S_i)(1 + ρ·(t_d − t_s)/360)`.

**Quote risk**: `R = G B (J_T B)⁻¹`, `U = G − R J_T`, `h = −R/s`.

## Appendix B — References

- CME Group, *Calculating U.S. Treasury Futures Conversion Factors* (cmegroup.com, Interest Rate
  Resource Center); *U.S. Treasury Futures Conversion Factor Lookup Tables*; CBOT Rulebook
  chapters 18 (T-Bond), 19 (10-Year), 20/21 (5-/2-Year), 26 (options); Ultra 10-Year options
  contract specifications (American exercise); CME Clearing Advisory 15-260 (premium-style vs
  futures-style options).
- 31 CFR Part 356, Appendix B — *Formulas and Tables* (price, yield, accrued interest, bills).
- OpenGamma Strata: `RatesCurveCalibrator.jacobianDirect`, `JacobianCalibrationMatrix`,
  `CurveSensitivityUtils.linearRebucketing`, `MarketQuoteSensitivityCalculator`.
- rateslib: `Solver` (`grad_s_vT = pinv(J)`, `Solver.jacobian(other)`), `FixedRateBond(spec="us_gb")`,
  `BondFuture` (`calc_mode="ust_short"|"ust_long"`, `cfs`, `ctd_index`, `dlv`, `implied_repo`,
  `net_basis`).
- QuantLib: `BondFunctions`, `ActualActual(ISMA)`, `UnitedStates(GovernmentBond)` — parity oracle.
- This repository: `docs/risk_map_and_dual_ad.md` (superseded §1), `docs/engine_op_graph_plan.md`,
  `research/risk_reconciliation.py`.
