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
2. **Trial evaluation is a market rebind, not a rebuild.** The solver loop composes the
   trial `ZeroCurve` with index conventions, calls `market.with_curve(yield_curve)`, and
   re-evaluates residuals. The curve is
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
| `YieldCurve` + `CurveNamespace` | `markets/curves/yield_curve.py`, `namespace.py` | conventions | Composes pure curve state with an index definition; the namespace binds these named market objects. |
| `RateGenerator` | `markets/rate_generator.py` | market context, day counts | Owns simple/continuous/compounded/averaged rates and par-rate generation; these do not belong on `ZeroCurve`. |
| Solvers | `pricing/calibration/solvers.py` | numpy, scipy | Deliberately domain-blind: residual closure + `x0` in, solution out. Copy verbatim. |
| `CurveCalibrator` + `CurveDefinition` | `pricing/calibration/calibrator.py` | curves, solvers, market shim | Returns a `ZeroCurve`; the optional exact quote Jacobian uses the Numba adjoint by default. |
| `CalibrationInstrument` protocol + `Quote`/`QuoteKind` | `pricing/calibration/instruments.py` (top ~60 lines) | none | The seam itself. Port the protocol and quote types only — **not** the concrete helpers below them. |

### Re-implemented as thin adapters (against the platform's instruments)

| Piece | Effort | How |
|---|---|---|
| Market context shim | small | The calibrator needs `as_of_date`, registered convention-aware curves, and `with_curve(yield_curve)`. Projection is supplied separately by `RateGenerator`. |
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
(vendor), and `finance.dates` day-count fractions for `RateGenerator.simple_rate`
(the platform already has day-count utilities — use theirs).
*Acceptance: curves construct and round-trip DF/log-DF/zero against saved fixtures from
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
Port verbatim (omit the optional Jacobian hook if the target does not port Numba AAD). Wire the calibrator's `build_curve`
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

This repo uses an explicit composition: `ZeroCurve` owns mathematical state and
`YieldCurve` attaches the `MarketConventions` bundle for one `(currency, index)`. The
`YieldCurve`, not the `ZeroCurve`, is registered in `MarketContext`. Two consequences for
the port remain:

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

Hold the same boundary on the target platform: `ZeroCurve` remains pure math, while a thin
market object such as `YieldCurve` owns the association to index metadata. This mirrors the
implemented architecture and keeps a calibration trial cheap to compose and rebind.

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
    yield_curve.py      [port]     ← src: finance/markets/curves/yield_curve.py
    enums.py            [port]     ← src: finance/markets/curves/types.py
    namespace.py        [port]     ← src: finance/markets/curves/namespace.py
  market/
    rate_generator.py   [port]     ← src: finance/markets/rate_generator.py
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
    def log_discount_factor(self, dates) -> FloatArray: ...
    def zero_rate(self, dates) -> FloatArray: ...         # cont-comp, Act/365
```

### curves/namespace.py  `[port]`
```python
@dataclass(frozen=True)
class YieldCurve:
    zero_curve: ZeroCurve; conventions: MarketConventions
    @property
    def name(self) -> str: ...
    def with_zero_curve(self, zero_curve) -> "YieldCurve": ...

class CurveNamespace:                         # ← src: namespace.py::CurveNamespace
    def bind(self, curve: YieldCurve): ...    # canonical name comes from the market definition
    def rebind(self, curve: YieldCurve): ...
    def resolve(self, name) -> YieldCurve: ...
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
    currency: str; index_name: str
    interpolation: CurveInterpolator = CurveInterpolator.LogLinearDF
@dataclass
class CalibrationResult:                        # ← calibrator.py::CalibrationResult
    zero_curve: ZeroCurve; solver_result: SolverResult
    pillar_dates; residuals; jacobian=None       # optional; requires the Numba adjoint (§7)
@dataclass
class CurveCalibrator:                          # ← calibrator.py::CurveCalibrator
    instruments: Sequence[CalibrationInstrument]; solver: Solver; target: CurveDefinition
    def calibrate(self, market, *, jacobian=False) -> CalibrationResult:
        # node dates = sorted instrument pillar_dates; residual_fn rebinds trial curve:
        #   trial = ZeroCurve(pillars, x_to_dfs(x), target.interpolation)
        #   yc = target.compose(trial, market)
        #   resid = [inst.implied(market.with_curve(yc)) - inst.quote.value ...]
        # solver.solve(residual_fn, x0) -> node DFs
```

### adapters/market_shim.py  `[adapter]` — platform market must answer these
```python
class MarketShim:                               # ← src: context.py::MarketContext (subset only)
    as_of_date: Date
    def with_curve(self, curve: YieldCurve) -> "MarketShim": ...  # fresh rebind
    def yield_curve(self, name) -> YieldCurve: ...
    def zero_curve(self, name) -> ZeroCurve: ...
    # NOT ported: fixings (SinglePath), vols, product conventions bundle

class RateGenerator:                           # ← src: markets/rate_generator.py
    def simple_rate(self, name, starts, ends, day_count=None) -> FloatArray: ...
    def continuous_forward_rate(self, name, starts, ends) -> FloatArray: ...
```

### adapters/helpers.py  `[adapter]` — re-implement against platform instruments
```python
# ← src: instruments.py::{DepositHelper, FraHelper, SwapHelper} — closed-form, no pricer needed
class DepositHelper / FraHelper:                # each implements the CalibrationInstrument protocol
    quote: Quote; curve: str
    def pillar_date(self): return self.instrument.maturity
    def implied(self, market):                  # one DF-ratio projection over the accrual window
        return RateGenerator(market).simple_rate(self.curve, [start], [end], day_count)[0]
class SwapHelper:                               # OIS: closed-form par (projection == discount)
    def implied(self, market):                  #   par = (DF_eff - DF_mat) / Σ τᵢ·DF_payᵢ
        ...                                      #   fall back to platform swap pricer only if telescoping breaks
```

### End-to-end wiring (mirrors `research/sofr_curve_calibration.py`)
```python
helpers = [DepositHelper(...), SwapHelper(...), ...]            # one node per pillar, increasing dates
calibrator = CurveCalibrator(instruments=helpers, solver=GlobalSolver(),
                             target=CurveDefinition("USD", "SOFR", CurveInterpolator.LogLinearDF))
base = MarketShim(as_of_date=AS_OF, curves=CurveNamespace())
result = calibrator.calibrate(base)
yield_curve = YieldCurve(result.zero_curve, sofr_conventions)
market = base.with_curve(yield_curve)
```

---

## 7. Known losses and caveats

- **`calibrate(jacobian=True)` requires an adjoint port.** The default exact quote-Jacobian
  rides on the Numba pricing adjoint. Omit it if the target only needs bump-based curve risk.
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
