# Curve refactor: `Line1d` as the foundation of `ZeroCurve`

Status: PR 1 in progress. This document is the working plan; tick items as they land and
keep the "Deferred" section current so design notes are not lost between sessions.

## Why

The curve stack's layering (`ZeroCurve` -> `YieldCurve` -> `CurveNamespace`/`MarketContext`
-> `RateGenerator`) is the same decomposition as OpenGamma Strata
(`Curve` -> `DiscountFactors` -> `IndexRates` -> `RatesProvider`) and is not the problem.
The complication is concentrated in `ZeroCurve.__init__`, and `common.math.line.Line1d`
was never able to sit underneath it. Specifically:

- `ZeroCurve` took nine constructor parameters, five of them for an experimental
  dual-segment ("short/long/cutover") feature that belongs in an interpolator.
- `alpha`/`beta` were bare floats whose meaning changed per scheme (second derivative for
  the cubics, first derivative for the quadratic, ignored for the linears).
- `Line1d` interpolated on `y` directly while every curve scheme interpolates on a transform
  (`ln DF` or the zero rate), so there was no seam to plug one into the other.
- `Line1d` extrapolated the same way on both sides and signalled "not allowed" with NaN
  sentinels it then scanned for.
- `Line1d`'s flat interpolation returned the **next** node's value between nodes. For a
  fixings series that gives Saturday and Sunday Monday's print instead of Friday's.
- Curve construction ran three Python loops (Thomas solver, PPoly build, zero-rate loop) on
  the calibration hot path, which rebuilds the curve every solver iteration.

## Target design

```text
common.math.interpolation   Interpolator (config) --fit(x, y)--> BoundInterpolator
                            Flat | Linear | Cubic(bc_type) | Quadratic(slopes) | Mixed(short, long, switch_node)
common.math.line            Line1d(x, y, interpolator, left, right)   left/right: Extrapolation enum
                            line(xq), line.derivative(xq), line.with_y(y), line.coefficients

finance.markets.curves      ZeroCurve(x, dfs, space, interpolator, extrapolation)
                              = Line1d in LogDF or ZeroRate space + DF invariants; x-only, no dates
                            YieldCurve(origin, zero_curve, conventions, fixings)
                              = the only place dates and Terms are converted to x
```

Decisions and their reasons:

| Decision | Reason |
|---|---|
| Interpolators are frozen instances, not an enum + registry | Parameters travel with the scheme; `alpha`/`beta` ambiguity disappears. Strata passes interpolator instances; QuantLib templates on them. |
| `Cubic(bc_type=...)` mirrors scipy's `CubicSpline` contract | Same information as QuantLib's `(condition, value)` pair per side without inventing classes. Default `"natural"` = today's `alpha = beta = 0`. |
| `Quadratic(left_slope, right_slope)` | Only one kind of end condition exists for a C1 quadratic, so plain floats are unambiguous. |
| `Flat` is previous-hold only | Fri fixing covers Sat/Sun; Mon takes over on Mon. No use found for next-hold. |
| `Mixed(short, long, switch_node)` on a node index | QuantLib's `MixedLinearCubicInterpolation` precedent. `YieldCurve` resolves a `Term`/date to the index. C0 at the join (segments fit independently); C1 join is a later option. |
| Extrapolation stays an enum, applied per side | `NotAllowed` / `Flat` / `Linear` carry no parameters. Left and right independent, as Strata. |
| Bounds checked up front, raise with count | No NaN sentinel, no allocate-then-scan. |
| Fit with scipy, evaluate with `PPoly`, export `coefficients` | Fit runs once per construction and never needs to be jit-compatible. Evaluation is searchsorted + Horner, which the Numba/JAX engines already replicate; `coefficients` is the hand-off. |
| numpy in `common/`, Numba only under `finance/pricing/engines` | Consumers call the curve once with large deduplicated arrays; Numba wins on scalar calls, which nothing on the hot path does. Keeps Numba optional. |
| Origin moves to `YieldCurve` | `ZeroCurve` becomes pure x-space math (Strata `Curve`); `YieldCurve` owns the valuation date (Strata `DiscountFactors`). |
| DF = 1 at x = 0 and DF > 0 stay on `ZeroCurve` | Finance invariants, not line invariants. |

## PR 1: `Line1d` foundation (`common/math`) — additive

- [x] `common/math/interpolation.py`: `Interpolator` / `BoundInterpolator` protocols;
      `Flat`, `Linear`, `Cubic`, `Quadratic`, `Mixed`; `BoundPPoly` (scipy) and `BoundFlat`.
- [x] `common/math/line.py`: `Line1d(x, y, interpolator, left, right)`, `Extrapolation`
      enum (`NotAllowed`, `Flat`, `Linear`), `__call__`, `derivative`, `with_y`,
      `coefficients`; `get_value` kept as an alias.
- [x] Migrate `HistoricalFixings` and `SinglePath`. Fixings now previous-hold.
- [x] Tests: node/extrapolation/rebinding/validation; each scheme against a reference
      (scipy for `Cubic`, the old loop sweep for `Quadratic`, `np.interp` for `Linear`);
      `Mixed` continuity and padding; hypothesis knot-recovery; the Fri/Sat/Sun/Mon case.
- [x] `docs/curve_architecture.md`: fixings paragraph updated.

## PR 2: `ZeroCurve` on `Line1d`; origin to `YieldCurve`

- [ ] `ZeroCurve(x, dfs, *, space=CurveSpace.LogDF, interpolator=Linear(), extrapolation=RateExtrapolator.FlatForward)`.
      `space` owns the transform to/from DF. Left is always rejected. Flat-forward uses the
      interpolant's end derivative, not a one-day finite difference.
- [ ] `with_dfs(dfs)` delegates to `Line1d.with_y`. `geometry` exports x, values-in-space,
      space, coefficients for the engines.
- [ ] Remove `interpolation_long` / `interpolation_cutover` / `alpha*` / `beta*`; `Mixed`
      covers the use case. Delete `_curve_impl/interpolators.py`.
- [ ] `YieldCurve` gains `origin`; `discount_factor(dates)`, `log_discount_factor(dates)`,
      `zero_rate(dates)` convert dates -> x here. `node_index(Term | date)` for `Mixed`.
      `YieldCurve.build(...)` convenience for dates-in construction.
- [ ] Consumers: `pricing/kernels/compiler.py`, `pricers/futures.py`, `engines/numba/program.py`,
      `engines/numba/calibration.py`, `engines/jax/{program,curves,calibration}.py`,
      `risk/{sensitivities,engine}.py`, `calibration/calibrator.py` (`build_curve`),
      `quant_toolkit_xl` UDFs that construct curves, `research/` scripts.
- [ ] Regression expects captured from the current implementation before the rewrite
      (`common/testing/expects_loader.py`) so DF/zero/forward numbers are pinned.

## PR 3: cleanup and docs

- [ ] `CurveInterpolator` enum kept only as a parse table for Excel/config ->
      `(CurveSpace, Interpolator)`. Negative values dropped; `SupportedIntEnum` unused here.
- [ ] `CLAUDE.md` (currently describes a log-DF-storing `ZeroCurve` on `common/containers/curve1d.py`,
      neither exists) and `docs/pricing_architecture.md`, `docs/curve_architecture.md`.
- [ ] Performance tests: curve construction (calibration hot path) and bulk query.

## Deferred — design notes to re-examine

**Composite curves (turns, basis spreads).** Both are the same mechanism: `ln DF_total(x) =
sum of components`, each component a `Line1d` in log-DF space (or rate space with a
transform). `ZeroCurve.components: tuple[Line1d, ...]` is a two-line generalisation of the
single-`Line1d` version and can be added without breaking PR 2.

- *Turns.* A year-end / quarter-end turn is a spread `s` on the overnight rate over a window
  `[a, b]`: a log-DF overlay with nodes at `a` and `b`, `Linear` between them, `Flat` both
  sides (`-s * (b - a) / 365` beyond `b`). Base curve stays smooth; instruments spanning the
  turn see the accrual, others don't. Calibration is sequential: size turns from a Dec/Jan
  futures spread or an OIS pair or a trader mark, then bootstrap the base with the overlay
  held fixed (solve one component's `y`, others constant). Reference: Burgess, SSRN 3898069
  (not fetched; SSRN blocks automated access).
- *Rate-level spreads are not curves.* Prime = FF + 300bp on the published rate; ISDA fallback
  adds its spread to the compounded-in-arrears rate after compounding; EONIA = €STR + 8.5bp
  was an index redefinition. These belong in conventions (a `RateIndex` naming a base index
  and a spread) and are applied by `RateGenerator` at the point the convention says.
  Representing Prime as a DF curve is wrong by a compounding cross-term and puts the spread in
  the wrong place for compounded coupons.
- *Curve-level spreads (basis curves)* — FF/SOFR basis, term vs OIS, CSA discount over OIS,
  constant continuous spread (two nodes, `Flat` both sides) — are log-DF components.
  QuantLib: `ZeroSpreadedTermStructure`, `PiecewiseZeroSpreadedTermStructure`. Strata does
  not compose curves (each basis curve is calibrated as independent nodes).
- *Dependency direction.* Keep `ZeroCurve` by-value. Put "spread curve S depends on base B"
  in `CurveNamespace` as a derived-curve definition and recompose dependents eagerly in
  `MarketContext.with_curve` when a base is rebound. Keeps the dependency graph in one place
  rather than inside curve objects (the QuantLib handle-graph problem).

**Futures convexity.** Already modelled the QuantLib/Strata way: a per-contract input
(`SofrFuture.convexity`; `FuturesHelper.convexity: float | callable(market)`), added by the
futures pricer, consumed on the quote side by calibration. It is a property of futures
settlement mechanics, so it stays on the future/pricer and does **not** move into
`RateGenerator`. Where the number comes from is a separate provider concern:
`HullWhiteConvexity(sigma, mean_reversion)` (closed form; Mercurio/Henrard variants for 1M
averaged vs 3M compounded SOFR) or `MarketImpliedConvexity` (futures minus OIS-implied
forward, when the OIS strip is liquid; note that residual also contains turn effects). Rough
magnitude at 1% vol: ~0.05bp at 3M, ~2bp at 2Y, ~4.5bp at 3Y.

**Curve identity vs index identity.** `YieldCurve.name` is always `CCY.INDEX`; funding
curves resolve through the `_FUNDING_ALIASES` dict in `instruments/resolution/resolver.py`
and `FundingIndex` is an unused stub. Strata separates `Currency -> DiscountFactors` from
`Index -> forward curve`. Closing this is a `MarketContext` concern and pairs naturally with
the derived-curve definitions above.

**`CurveNamespace` versioning.** `version()` is read by nothing, and `MarketContext.with_curve`
rebuilds the namespace so versions reset anyway. Remove `bind`/`rebind`/`version` in a
separate small PR; the immutable rebind path already provides staleness.

**Monotone / shape-preserving cubics.** Add as separate interpolators (`MonotoneCubic`,
`Pchip`) following Strata's naming, not as flags on `Cubic`. The Numba AAD engine needs
matching analytic weights before any non-log-linear scheme is enabled there.

**`Mixed` C1 join.** Constrain the long segment's left end slope to the short segment's end
slope (for `Cubic`: `bc_type=((1, slope), right)`); the `Mixed` fit already has the short
bound interpolator's `derivative` available.
