# Porting the Curve + Calibration Layer into an External Platform

A step-by-step plan for seaming this repo's curve implementations, calibration routines,
and helper concepts into an existing pricing & risk platform **without** porting the
kernel/compiler, product pricers, risk analytics, or this repo's instrument layer.
The target platform keeps its own instrument representation; adapters bridge the gap.

---

## 1. Why this is feasible

The calibration layer was built around two deliberate seams:

1. **`CalibrationInstrument` is a protocol, not a base class.** The `CurveCalibrator`
   only ever touches four members — `quote`, `curve`, `pillar_date`, `implied(market)` —
   and never reaches into instrument internals. Anything that can compute "the model
   value of my quoted measure off a trial curve" plugs in, regardless of which
   instrument library sits underneath.
2. **Trial evaluation is a market rebind, not a rebuild.** The solver loop is
   `market.with_curve(name, trial_curve)` → re-evaluate residuals. The curve is
   *injected* per iteration; no instrument object is ever reconstructed.

Because of (1), the target platform's instruments do not need to be re-implemented to
calibrate curves. Because of (2), the integration cost concentrates in one place: the
platform must be able to price a calibration instrument off an injected curve.

---

## 2. What flows through, what gets adapted, what stays behind

### Ports as-is (the "port set")

| Piece | Source | External deps | Notes |
|---|---|---|---|
| Line / interpolator layer | `finance/markets/curves/_curve_impl/interpolators.py` | numpy, scipy (`PPoly`) | The real engine: `Interpolator` hierarchy (log-linear/log-cubic DF, rate linear/quadratic/cubic) + `InterpolatedSegment`, all in float day-space. **This must flow through** — `ZeroCurve` is a thin date wrapper over it. |
| `ZeroCurve` | `_curve_impl/zero_curve.py` | interpolators, `finance.dates.Term` | `Term` is only used for the dual-segment cutover convenience; either vendor `Term` or restrict the ported API to `datetime64` cutovers to drop the dependency. |
| `CurveInterpolator` / `RateExtrapolator` enums | `markets/curves/types.py` | `common.containers.enums.SupportedIntEnum` | Trivial; vendor the enum base or replace with `IntEnum`. |
| `CurveNamespace` | `markets/curves/namespace.py` | none beyond curves | Optional but cheap; gives versioned name→curve binding. |
| Solvers | `pricing/calibration/solvers.py` | numpy, scipy | Deliberately domain-blind: residual closure + `x0` in, solution out. Copy verbatim. |
| `CurveCalibrator` + `CurveDefinition` | `pricing/calibration/calibrator.py` | curves, solvers, market shim | Strip the `vol_shim` / `FlatVolSurface` block and the `jacobian=True` hook (JAX-coupled, see §7) for the first cut. |
| `CalibrationInstrument` protocol + `Quote`/`QuoteKind` | `pricing/calibration/instruments.py` (top ~60 lines) | none | The seam itself. Port the protocol and quote types only — **not** the concrete helpers below them. |

### Re-implemented as thin adapters (against the platform's instruments)

| Piece | Effort | How |
|---|---|---|
| Market context shim | small | The calibrator needs an object answering `as_of_date`, `with_curve(name, curve)`, `discount_factor(name, dates)`, and simple projection (`(DF_s/DF_e − 1)/τ`). Either port `MarketContext` minus fixings/vols/conventions, or make the platform's market object satisfy that interface. |
| Deposit / FRA helpers | trivial | Closed-form: one DF-ratio projection over the accrual window. Rewrite against the platform's deposit/FRA objects (they only need to supply effective, maturity, day count). |
| Swap (OIS) helper | the real work | Two options, in order of preference: **(a) closed-form par**: with projection = discount, the compounded float leg telescopes, so `par = (DF_eff − DF_mat) / Σ τᵢ·DF_payᵢ` — needs only the fixed-leg schedule from the platform's instrument, no pricer at all (this is effectively QuantLib's `OISRateHelper`); **(b) pricer adapter**: `implied(market)` calls the platform's own swap pricer for annuity and float PV off the trial curve. |

### Stays behind entirely

- `pricing/kernels/` (columnar IR + compiler), `pricing/pricers/`, `pricing/engines/`
  (numpy/jax/numba/rust), `pricing/risk/` (bump + autodiff)
- This repo's instrument layer (`instruments/`, schedules, `Priceable`)
- Fixings overlay (`SinglePath`), vols, and the full conventions bundle (see §5)

---

## 3. Steps

**Phase 0 — carve the port set.**
Extract the table-1 modules into the platform's tree (or a shared package). Resolve the
three small dependencies: `Term` (vendor or drop the Term-cutover API), `SupportedIntEnum`
(vendor), `finance.dates` day-count fractions for the market shim's `project`
(the platform already has day-count utilities — use theirs).
*Acceptance: curves construct and round-trip DF/rate/forward against saved fixtures from
this repo.*

**Phase 1 — curves behind the platform's curve interface.**
Drop `ZeroCurve` + interpolators in behind whatever curve interface the platform's
pricers consume. This is standalone — no instrument or calibration coupling — and
replaces the disliked curve implementations immediately.
*Acceptance: existing platform pricing regression suite passes with the new curve
implementation bound in.*

**Phase 2 — the market rebind shim.**
Provide the `with_curve`-style trial rebind (fresh namespace, shared everything else —
never mutate the input market). If platform pricers take a curve in the constructor,
this is the moment to add an injectable-curve path; it is the same fix this repo's
architecture doc describes and it is the single highest-value refactor available to the
platform regardless of this port.
*Acceptance: rebinding a curve and repricing an existing platform trade gives the same
answer as constructing fresh.*

**Phase 3 — solvers + calibrator.**
Port verbatim (minus vol shim / jacobian hook). Wire the calibrator's `build_curve`
closure to the ported `ZeroCurve`.
*Acceptance: the synthetic round-trip test from
`finance/tests/unit/pricing/calibration/test_curve_calibrator.py` — generate quotes from
a known curve, recalibrate from an empty market, recover node DFs to 1e-9 — ported and
green.*

**Phase 4 — helper adapters.**
Implement `DepositHelper`/`FraHelper` closed-form against platform instruments; implement
the OIS swap helper closed-form first (option a), and only fall back to the
pricer-adapter (option b) for products where telescoping genuinely breaks (basis swaps,
exotic float shaping). Keep one node per instrument, strictly increasing pillar dates —
the calibrator enforces both.
*Acceptance: calibrate a full SOFR curve from live platform quotes with both solvers;
max|residual| < 1e-10; GlobalSolver and Bootstrapper node DFs agree to ~1e-12.*

**Phase 5 — validation against the incumbent.**
Reprice a book snapshot off the new curve vs the platform's existing curve build.
Differences decompose into (i) interpolation-scheme changes (intended), (ii) helper
convention gaps (pay delay, day count — investigate each), (iii) bugs.
*Acceptance: signed-off diff report; agreed tolerance per tenor bucket.*

---

## 4. The Line layer — yes, it flows through

`ZeroCurve` is deliberately thin: date conversion at the boundary, then everything —
interpolation, extrapolation, dual-segment cutover, terminal-forward flat extrapolation —
happens in the `Interpolator`/`InterpolatedSegment` layer on float days-since-origin.
Porting `ZeroCurve` without it is not meaningful; porting it alone (if the platform
wants to keep its own date-facing curve class) is entirely viable — the layer's only
dependencies are numpy and scipy, and its x-axis is just floats. Either way, the Line
implementation is part of the port set, and it is also the natural place for the
platform to add its own interpolation schemes later without touching the calibrator.

---

## 5. Conventions: link the *market* tier to curves, keep the *product* tier out

Agreed — with one refinement.

This repo's `MarketConventions` is currently a monolithic bundle per `(currency, index)`:
`RateIndex` (index day count, fixing calendar, fixing style) **plus** swap/deposit/FRA
leg conventions. Two consequences for the port:

1. You cannot register an index without supplying full product conventions.
2. The bundle imports `CouponType` from `finance.instruments.enums` (via
   `FloatLegConventions.coupon_type`), so the conventions package silently depends on the
   instrument layer — exactly the coupling the port is trying to avoid.

The right split is **two tiers**:

- **Market/index tier** (ports with the curves): `RateIndex` — index label, projection
  day count, fixing calendar, fixing style — plus curve naming
  (`curve_name(currency, index)` → `"USD.SOFR"`, `funding_curve_name`). This is what the
  curve/market/calibration layer needs: which curve a name resolves to, and how the
  index it projects accrues. Depends only on `dates`.
- **Product tier** (stays with whichever instrument layer is in play): leg shapes,
  frequencies, BDCs, spot lags, payment delays. On the platform side, *their*
  instrument builders own this; this repo's product conventions never cross the seam.

One refinement worth holding to: link conventions to curves **at the market layer, not
inside the curve object**. `ZeroCurve` today is pure math — node dates, values,
interpolation — and that purity is what makes it portable and what keeps a calibration
trial a cheap rebind. The index association belongs beside the name binding (the
namespace / market context maps `name → (curve, index metadata)`), not as a field on the
curve. Same end result — a curve reachable with its index conventions and no product
conventions in sight — without giving the curve container a reason to know about
markets. (This mirrors Strata: `Curve` is dumb, `RatesProvider` owns the
index→curve association; and rateslib similarly keeps `Curve` free of instrument
knowledge.)

Applying the same split back in *this* repo (registry accepts an index-tier-only
registration; product tier layered on top) would be a small, worthwhile PR on its own —
it makes the port set a clean subtree instead of a surgical extraction.

---

## 6. Port skeleton — pseudo-modules ↔ implemented code

The target-platform layout below is what the "port set" + adapters become on the other
side. Each pseudo-module is annotated with the concrete source it derives from
(`← src: …`). **`[port]`** = copy with trivial dep edits; **`[adapter]`** = re-implement
against the platform's instruments.

```
platform_curverisk/
  curves/
    interpolators.py    [port]     ← src: finance/markets/curves/_curve_impl/interpolators.py
    zero_curve.py       [port]     ← src: finance/markets/curves/_curve_impl/zero_curve.py
    enums.py            [port]     ← src: finance/markets/curves/types.py
    namespace.py        [port]     ← src: finance/markets/curves/namespace.py
  calibration/
    protocol.py         [port]     ← src: pricing/calibration/instruments.py  (top ~60 lines)
    solvers.py          [port]     ← src: pricing/calibration/solvers.py
    calibrator.py       [port]     ← src: pricing/calibration/calibrator.py
  adapters/
    market_shim.py      [adapter]  ← src: finance/markets/context.py  (subset)
    helpers.py          [adapter]  ← src: pricing/calibration/instruments.py  (concrete helpers)
```

### curves/interpolators.py  `[port]`
```python
# ← src: interpolators.py — the real engine, float days-since-origin x-axis. numpy+scipy only.
class Interpolator(ABC):                      # ← class Interpolator(ABC)
    def df(self, x: FloatArray) -> FloatArray: ...
class DFLogLinear(Interpolator): ...          # ← DFLogLinear   (CurveInterpolator.LogLinearDF)
class DFLogCubic(Interpolator): ...           # ← DFLogCubic    (LogCubicDF)
class RateLinear(Interpolator): ...           # ← RateLinear / RateQuadratic / RateCubic
def build_interpolator(kind, x, df, ...) -> Interpolator: ...   # ← build_interpolator(...)
class InterpolatedSegment: ...                # ← InterpolatedSegment (dual-segment cutover)
```

### curves/enums.py  `[port]`
```python
class CurveInterpolator(IntEnum):             # ← src: types.py::CurveInterpolator
    LogLinearDF = 1; LogCubicDF = 2; RateLinear = -1; RateQuadratic = -2; RateCubic = -3
class RateExtrapolator(IntEnum):              # ← src: types.py::RateExtrapolator
    Flat = 1; NotAllowed = 2
```

### curves/zero_curve.py  `[port]`
```python
class ZeroCurve:                              # ← src: zero_curve.py::ZeroCurve (thin date wrapper)
    def __init__(self, node_dates, node_values,          # node_values are DISCOUNT FACTORS
                 interpolation=CurveInterpolator.LogLinearDF,
                 *, interpolation_long=None, interpolation_cutover=None): ...
    # properties: origin, node_dates, node_dfs, max_date, interpolation
    def discount_factor(self, dates) -> FloatArray: ...   # ← discount_factor
    def rate(self, dates) -> FloatArray: ...              # ← rate (cont-comp, Act/365)
    def forward_rate(self, starts, ends) -> FloatArray: ...
```

### curves/namespace.py  `[port]`
```python
class CurveNamespace:                         # ← src: namespace.py::CurveNamespace
    def bind(self, name, curve): ...          # bind (raises if present) / rebind (overwrite+version)
    def rebind(self, name, curve): ...
    def resolve(self, name) -> ZeroCurve: ...
    def version(self, name) -> int: ...        # snapshot(), __contains__ also ported
```

### calibration/protocol.py  `[port]` — the seam itself
```python
class QuoteKind(IntEnum):                     # ← src: instruments.py::QuoteKind
    ParRate = 1; SimpleRate = 2; Yield = 3    # swaps / deposits+FRAs / bonds (Yield = designed seam)
@dataclass
class Quote:                                   # ← instruments.py::Quote
    value: float; kind: QuoteKind
class CalibrationInstrument(Protocol):        # ← instruments.py::CalibrationInstrument
    quote: Quote                              # the ONLY four members the calibrator touches:
    curve: str                                #   quote, curve, pillar_date, implied(market)
    @property
    def pillar_date(self) -> np.datetime64: ...
    def implied(self, market) -> float: ...
```

### calibration/solvers.py  `[port]` — domain-blind, copy verbatim
```python
@dataclass
class SolverResult: ...                        # ← src: solvers.py::SolverResult
class Solver(Protocol):                        # ← Solver
    def solve(self, residual_fn: ResidualFn, x0: FloatArray) -> SolverResult: ...
class GlobalSolver:  ...                        # ← GlobalSolver   (scipy least_squares, all nodes)
class Bootstrapper: ...                         # ← Bootstrapper   (sequential brentq; LOCAL interp only)
```

### calibration/calibrator.py  `[port]` — strip vol-shim + `jacobian=True` for first cut
```python
@dataclass(frozen=True)
class CurveDefinition:                          # ← src: calibrator.py::CurveDefinition
    name: str; interpolation: CurveInterpolator = CurveInterpolator.LogLinearDF
@dataclass
class CalibrationResult:                        # ← calibrator.py::CalibrationResult
    curve: ZeroCurve; market; solver_result: SolverResult
    pillar_dates; residuals; jacobian=None       # jacobian stays behind (JAX-coupled, §7)
@dataclass
class CurveCalibrator:                          # ← calibrator.py::CurveCalibrator
    instruments: Sequence[CalibrationInstrument]; solver: Solver; target: CurveDefinition
    def calibrate(self, market, *, jacobian=False) -> CalibrationResult:
        # node dates = sorted instrument pillar_dates; residual_fn rebinds trial curve:
        #   trial = ZeroCurve(pillars, x_to_dfs(x), target.interpolation)
        #   resid = [inst.implied(market.with_curve(target.name, trial)) - inst.quote.value ...]
        # solver.solve(residual_fn, x0) -> node DFs
```

### adapters/market_shim.py  `[adapter]` — platform market must answer these
```python
class MarketShim:                               # ← src: context.py::MarketContext (subset only)
    as_of_date: Date
    def with_curve(self, name, curve) -> "MarketShim": ...   # fresh rebind, never mutate self
    def discount_factor(self, name, dates) -> FloatArray: ...
    def project(self, name, starts, ends, day_count) -> FloatArray:
        # simple forward off the curve:  (DF(starts)/DF(ends) - 1) / tau
    def discount(self, name) -> ZeroCurve: ...
    # NOT ported: fixings (SinglePath), vols, product conventions bundle
```

### adapters/helpers.py  `[adapter]` — re-implement against platform instruments
```python
# ← src: instruments.py::{DepositHelper, FraHelper, SwapHelper} — closed-form, no pricer needed
class DepositHelper / FraHelper:                # each implements the CalibrationInstrument protocol
    quote: Quote; curve: str
    def pillar_date(self): return self.instrument.maturity
    def implied(self, market):                  # one DF-ratio projection over the accrual window
        return market.project(self.curve, [start], [end], day_count)[0]
class SwapHelper:                               # OIS: closed-form par (projection == discount)
    def implied(self, market):                  #   par = (DF_eff - DF_mat) / Σ τᵢ·DF_payᵢ
        ...                                      #   fall back to platform swap pricer only if telescoping breaks
```

### End-to-end wiring (mirrors `research/sofr_curve_calibration.py`)
```python
helpers = [DepositHelper(...), SwapHelper(...), ...]            # one node per pillar, increasing dates
calibrator = CurveCalibrator(instruments=helpers, solver=GlobalSolver(),
                             target=CurveDefinition("USD.SOFR", CurveInterpolator.LogLinearDF))
result = calibrator.calibrate(MarketShim(as_of_date=AS_OF, curves=CurveNamespace()))
curve = result.curve                                            # bind into the platform's RatesProvider
```

---

## 7. Known losses and caveats

- **`calibrate(jacobian=True)` does not port.** The exact quote-Jacobian rides on the
  JAX twin of the pricing kernel. Bump-based par DV01 against the calibrated curve works
  without it; revisit only if the platform wants algorithmic partial DV01s.
- **Closed-form OIS telescoping is an approximation** under payment delay (pay date ≠
  observation end) and lookback/lockout. Sub-bp for standard USD SOFR conventions, but
  bound it explicitly in Phase 4 acceptance rather than assuming.
- **Performance depends on the platform's residual cost.** The bootstrapper evaluates
  the full residual vector per bracketing step (~230 evaluations for a 23-pillar curve
  in this repo). Closed-form helpers make that free; pricer-adapter helpers on a
  curve-in-constructor pricer make it O(rebuild)·O(evals) — functional, but the
  injectable-curve fix in Phase 2 is what makes it fast.
- **One instrument per pillar, strictly increasing pillar dates.** Watch short-end
  quotes whose maturities collide after business-day adjustment (this bit us in this
  repo: 1D and 2D deposits both rolling to the same Monday under a T+2 spot lag —
  the short end had to be built as O/N-style deposits anchored at the market date).
