# Curve and Rate Architecture

The curve stack separates mathematical state from market meaning. A calibrated curve is not
itself a SOFR curve: it becomes one only after SOFR's index conventions are attached.

```text
calibration quotes
       |
       v
   ZeroCurve             mathematical DF / log-DF / zero-rate state
       |
       | + USD SOFR conventions
       v
   YieldCurve            conventions + optional historical fixings
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

`ZeroCurve` is pure curve math. It owns immutable copies of pillar dates and discount
factors, interpolation/extrapolation configuration, and only these public numerical queries:

- `discount_factor(dates)`
- `log_discount_factor(dates)`
- `zero_rate(dates)` (continuously compounded on the curve's Act/365 clock)

It has no currency, index, fixing, forward-rate, coupon, or par-rate behavior.

`YieldCurve` composes one `ZeroCurve` with a `MarketConventions` definition and optional
`HistoricalFixings`. Its canonical name comes from that definition, for example `USD.SOFR`.
It is the object stored by `CurveNamespace` and `MarketContext`. `with_zero_curve(...)`
creates a scenario curve while preserving both the index definition and fixing history.

`MarketContext` owns registered market data. It resolves `yield_curve(name)` and
`zero_curve(name)` and produces immutable curve scenarios through `with_curve(yield_curve)`.
It does not calculate rates.

`RateGenerator` gives financial meaning to discount-factor ratios. It generates simple
index rates (including historical-fixing overlays), continuous interval forwards,
compounded and averaged observation rates, and par swap rates. Pricing and calibration
helpers use this layer rather than asking the curve or market to project a rate.

Historical/projected selection uses the curve origin as its only cutover:

```text
fixing date < ZeroCurve.origin   -> historical fixing (locked, zero index risk)
fixing date >= ZeroCurve.origin  -> projected rate (curve-sensitive)
payment date < ZeroCurve.origin  -> expired flow (DF = PV = all risk = 0)
```

`HistoricalFixings` is a date/value series backed by `Line1d` with flat interpolation and
flat extrapolation. Daily compounded coupons may therefore contain both locked observations
and projected observations. Their index delta is naturally the remaining, or "stubbed",
projection risk. A fully historical but unpaid coupon retains funding risk because its future
payment is discounted, while its index risk is zero.

`YieldCurveGroup` is intentionally deferred. `MarketContext` already supports multiple
registered yield curves, which is sufficient for SOFR discounting, Fed Funds projection,
and other current multi-curve cases. A currency-family group should be introduced only when
it has additional behavior or invariants to own.

## Calibration and registration

Calibration returns mathematical state. Registration is explicit:

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

zero_curve = result.zero_curve
sofr_curve = YieldCurve.from_registry(
    zero_curve,
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
scenario_zero = zero_curve.with_node_dfs(new_discount_factors)
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

The generic `ZeroCurve` supports the existing interpolation choices, including natural
cubic interpolation of log discount factors. The Numba pricing/adjoint geometry currently
supports `LogLinearDF` only; requesting another interpolation fails explicitly. A future
monotone-convex implementation belongs behind the `ZeroCurve` interpolation strategy and
must add matching analytic interpolation weights before it is enabled in the Numba AAD
engine.
