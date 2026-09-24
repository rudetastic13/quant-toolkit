# Curve and Rate Architecture

Responsibilities and boundaries. For usage, snippets and figures see
[curves_line1d.md](curves_line1d.md).

The curve stack separates mathematical state from market meaning. A calibrated curve is not
itself a SOFR curve: it becomes one only after SOFR's index conventions are attached.

```text
calibration quotes
       |
       v
   Line1d                common.math: nodes + interpolator + per-side extrapolation
       |
       | in log-DF or zero-rate space, DF invariants
       v
   ZeroCurve             mathematical DF / log-DF / zero-rate state on day offsets
       |
       | + origin date + USD SOFR conventions
       v
   YieldCurve            dates, conventions + optional historical fixings
       |
       v
   MarketContext         registry of yield curves and vols
       |
       v
   RateGenerator         forwards, compounded/averaged rates, and par rates
       |
       v
   PricingProgram        NumPy / Numba / JAX valuation and risk
```

## Responsibilities

`ZeroCurve` is pure curve math on float day offsets (`x[0] == 0`, `dfs[0] == 1`). It is a
`common.math.line.Line1d` in a declared `CurveSpace` (`LogDF` or `ZeroRate`) with any
`common.math.interpolation` scheme (`Linear`, `Cubic(bc_type)`, `Quadratic`, `Mixed`, ...),
plus the discount-factor invariants. Its public queries all take day offsets:

- `discount_factor(x)`, `log_discount_factor(x)`
- `zero_rate(x)` (continuously compounded on the curve's Act/365 clock)
- `instantaneous_forward(x)`
- `with_dfs(dfs)` rebinds node values without refitting the geometry checks (the
  calibration and bump-and-reprice hot path); `line.coefficients` is the engine hand-off

It has no dates, currency, index, fixing, forward-rate, coupon, or par-rate behavior.
`CurveInterpolator` (`LogLinearDF`, `RateCubic`, ...) is a parse table for config and Excel;
`resolve()` gives the `(space, interpolator)` pair.

`YieldCurve` composes an `origin` date and one `ZeroCurve` with a `MarketConventions`
definition and optional `HistoricalFixings`. It is the only layer that sees dates:
`discount_factor(dates)` etc. convert through `dates_to_x(origin, dates)`; `node_dates` and
`node_index(Term | date)` map pillars back to the calendar. Its canonical name comes from the
index definition, for example `USD.SOFR`, and it is the object stored by `CurveNamespace`
and `MarketContext`. `YieldCurve.build(node_dates, dfs, ...)` is the dates-in constructor;
`with_zero_curve(...)` creates a scenario curve while preserving origin, index definition and
fixing history.

`MarketContext` owns registered market data. It resolves `yield_curve(name)` and
`zero_curve(name)` and produces immutable curve scenarios through `with_curve(yield_curve)`.
It does not calculate rates.

`RateGenerator` gives financial meaning to discount-factor ratios. It generates simple
index rates (including historical-fixing overlays), continuous interval forwards,
compounded and averaged observation rates, and par swap rates. Pricing and calibration
helpers use this layer rather than asking the curve or market to project a rate.

Historical/projected selection uses the curve origin as its only cutover:

```text
fixing date < YieldCurve.origin   -> historical fixing (locked, zero index risk)
fixing date >= YieldCurve.origin  -> projected rate (curve-sensitive)
payment date < YieldCurve.origin  -> expired flow (DF = PV = all risk = 0)
```

`HistoricalFixings` is a date/value series backed by `Line1d` with `Flat` (previous-hold)
interpolation and flat extrapolation on both sides: a Friday print covers Saturday and
Sunday, Monday's print takes over on Monday. Daily compounded coupons may therefore contain both locked observations
and projected observations. Their index delta is naturally the remaining, or "stubbed",
projection risk. A fully historical but unpaid coupon retains funding risk because its future
payment is discounted, while its index risk is zero.

`YieldCurveGroup` is intentionally deferred. `MarketContext` already supports multiple
registered yield curves, which is sufficient for SOFR discounting, Fed Funds projection,
and other current multi-curve cases. A currency-family group should be introduced only when
it has additional behavior or invariants to own.

## Calibration and registration

Calibration returns mathematical state plus its origin. Registration is explicit:

```python
from finance.dates import Date
from finance.markets.context import MarketContext
from finance.markets import HistoricalFixings
from finance.markets.curves import CurveInterpolator, CurveNamespace, YieldCurve
from finance.pricing.calibration import (
    CurveCalibrator,
    CurveDefinition,
    GlobalSolver,
    deposit_helper,
    swap_helper,
)

as_of = Date(2026, 6, 3)
helpers = [
    deposit_helper(rate=0.0361, tenor="3M", as_of=as_of),
    swap_helper(rate=0.0394, tenor="2Y", as_of=as_of),
    swap_helper(rate=0.0399, tenor="7Y", as_of=as_of),
    swap_helper(rate=0.0409, tenor="10Y", as_of=as_of),
]
base = MarketContext(as_of_date=as_of, curves=CurveNamespace())
definition = CurveDefinition("USD", "SOFR", CurveInterpolator.LogLinearDF)
result = CurveCalibrator(helpers, GlobalSolver(), definition).calibrate(
    base,
    jacobian=True,
)

sofr_curve = YieldCurve.from_registry(
    result.origin,
    result.zero_curve,
    currency="USD",
    index_name="SOFR",
)
# Optional for a seasoned book:
# sofr_curve = sofr_curve.with_historical_fixings(
#     HistoricalFixings(fixing_dates, fixing_rates)
# )
market = base.with_curve(sofr_curve)
```

The same boundary is used for scenarios:

```python
scenario_zero = result.zero_curve.with_dfs(new_discount_factors)
scenario_market = market.with_curve(sofr_curve.with_zero_curve(scenario_zero))
```

## Pricing and rate generation

```python
from finance.instruments.resolution import Swap
from finance.markets import RateGenerator
from finance.pricing.pricers import SwapPricer
from finance.pricing.risk import NumbaRisk
from finance.pricing.types import Backend

template = Swap.fixed_float_swap(
    notional=100e6,
    rate_index="SOFR",
    fixed_rate=0.0,
    tenor="7Y",
    as_of=as_of,
)
par = RateGenerator(market).par_swap_rate(template)
swap = Swap.fixed_float_swap(
    notional=100e6,
    rate_index="SOFR",
    fixed_rate=par,
    tenor="7Y",
    as_of=as_of,
)

program = SwapPricer().compile([swap], backend=Backend.Numba)
price = program.price(market)
risk = NumbaRisk(program, market)
zero_dv01 = risk.zero_ladder("USD.SOFR")
index_dv01 = risk.index_ladder("USD.SOFR")
funding_dv01 = risk.funding_ladder("USD.SOFR")
partial_dv01 = risk.partial_dv01("USD.SOFR", result.jacobian)
gamma = risk.gamma("USD.SOFR")
```

For a complete runnable calibration, single-swap report, randomized N-swap aggregation,
and no-recompile scenario recalculation, see
[`research/sofr_curve_calibration.py`](../research/sofr_curve_calibration.py).

## Interpolation scope

`ZeroCurve` accepts any `common.math.interpolation` scheme in either space, including
`Mixed(short, long, switch_node)` for a log-linear front end with a spline beyond a chosen
pillar (`YieldCurve.node_index(Term("2Y"))` resolves the switch). The Numba and JAX
pricing/adjoint geometry model log-linear DF only (`ZeroCurve.is_log_linear`); requesting
another scheme fails explicitly and the risk engine falls back to bump-and-reprice. Every
bound interpolator exports PPoly `coefficients`, so extending an engine kernel to a new scheme
is a searchsorted + Horner branch on that array, plus matching analytic weights for the
adjoint.

Same pillar discount factors under the five named schemes. Zero rates agree at the pillars;
the forward panel is where the scheme choice shows:

![ZeroCurve schemes](img/curves/zero_curve_schemes.png)

Log-linear front end, natural cubic beyond the 2Y pillar via `Mixed`; the forward is
continuous in DF at the join, not in slope:

![Mixed yield curve](img/curves/yield_curve_mixed.png)
