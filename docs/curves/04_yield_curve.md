# 05 · `YieldCurve`: the landing place

A `YieldCurve` binds three things: an `origin` date, a `DiscountCurve` (the field is still named
`zero_curve`, but any protocol implementation sits there), and the `MarketConventions` of one
index. Optionally it carries `HistoricalFixings`. It is the only layer that sees dates, and it is
the object `MarketContext` registers and `RateGenerator` reads.

```text
DiscountCurve (02–04)  +  origin  +  (currency, index) conventions  +  fixings
        │
        ▼
    YieldCurve  ──registered by name──▶  CurveNamespace / MarketContext  ──▶  RateGenerator, pricers
```

Continues from [01](01_discount_curve.md) and [03](03_composite_curve.md) (`sofr`, `fedfund`,
`dfs`).

## Construction

`from_registry` composes an existing discount curve with the registered conventions for
`(currency, index_name)`; `build` is the dates-in convenience that constructs the `ZeroCurve`
for you (the first node is the origin).

```python
from finance.dates import Date, Term
from finance.markets.curves import CurveNamespace, YieldCurve, dates_to_x

as_of = Date(2026, 6, 1)
origin = as_of.to_numpy()
sofr_yc = YieldCurve.from_registry(origin, sofr, currency="USD", index_name="SOFR")
sofr_yc
# YieldCurve(name='USD.SOFR', origin=2026-06-01, zero_curve=ZeroCurve(nodes=6, max_x=3650, space=LogDF, interpolator=Linear(), extrapolation=FlatForward))
sofr_yc.name
# 'USD.SOFR'
sofr_yc.node_dates
# ['2026-06-01' '2026-08-30' '2027-06-01' '2028-05-31' '2031-05-31' '2036-05-29']
sofr_yc.index
# RateIndex(currency='USD', name='SOFR', day_count_method=<DayCountMethod.Actual360: 2>, fixing_calendar='no_holidays', ...)

built = YieldCurve.build(sofr_yc.node_dates, dfs, currency="USD", index_name="SOFR", interpolator=Cubic())
built
# YieldCurve(name='USD.SOFR', origin=2026-06-01, zero_curve=ZeroCurve(nodes=6, max_x=3650, space=LogDF, interpolator=Cubic(bc_type='natural'), extrapolation=FlatForward))
```

The canonical name `CCY.INDEX` comes from the conventions, never from the caller: a curve cannot
be registered under a name its index does not own.

## Dates to `x`

Every date query converts through `dates_to_x(origin, dates)`, float days from the origin, and
hands the result to the discount curve.

```python
dates = np.array(["2026-07-01", "2027-06-01", "2031-06-01"], dtype="datetime64[D]")
sofr_yc.to_x(dates)
# [  30.  365. 1826.]
sofr_yc.discount_factor(dates)
# [0.996636 0.960789 0.814547]
sofr_yc.zero_rate(dates)
# [0.041    0.04     0.041002]
dates_to_x(origin, sofr_yc.node_dates)
# [   0.   90.  365.  730. 1825. 3650.]
```

`node_index` maps a pillar back to its position, by date or by `Term` from the origin. It is
strict: the pillar must exist, which is how a `Mixed` switch node is placed safely.

```python
sofr_yc.node_index(np.datetime64("2028-05-31"))
# 3
sofr_yc.node_index(Term.from_str("2Y"))
# ValueError: no curve pillar on 2028-06-01; pillars are [...]      (this curve's 2Y pillar is 730 days, not a calendar 2Y)
```

## Fixings

`HistoricalFixings` is a date series held flat between prints (Friday's print covers the
weekend). `with_historical_fixings` attaches it without touching the curve. The origin is the
only cutover: a fixing date before `origin` is historical and carries no index risk; on or after
it is projected from the curve.

```python
from finance.markets import HistoricalFixings

fixings = HistoricalFixings(
    np.array(["2026-05-27", "2026-05-28", "2026-05-29"], dtype="datetime64[D]"),
    np.array([0.0428, 0.0431, 0.0429]),
)
seasoned = sofr_yc.with_historical_fixings(fixings)
seasoned.historical_fixings.get_value(np.array(["2026-05-30", "2026-05-29"], dtype="datetime64[D]"))
# [0.0429 0.0429]
```

## A spread curve as its own index

`USD.FEDFUND` is registered with `funding_id="SOFR"`: a Fed Funds swap projects off
`USD.FEDFUND` and discounts off `USD.SOFR`. Its discount curve is the labelled spread composite from
[03](03_composite_curve.md), sharing the SOFR atom by reference.

```python
fedfund_yc = YieldCurve.from_registry(origin, fedfund, currency="USD", index_name="FEDFUND")
fedfund_yc
# YieldCurve(name='USD.FEDFUND', origin=2026-06-01, zero_curve=CompositeCurve(base=+1*ZeroCurve(nodes=6, ...), spread=+1*ZeroCurve(nodes=2, ...)))
fedfund_yc.node_dates                              # the composite's union of pillars
# ['2026-06-01' '2026-08-30' '2027-06-01' '2028-05-31' '2031-05-31' '2036-05-29']
```

## In the market

`CurveNamespace` keys yield curves by canonical name; `MarketContext` is the container pricers
read. `with_curve` returns a new market with one curve replaced; the original is untouched.

```python
from finance.markets import RateGenerator
from finance.markets.context import MarketContext

ns = CurveNamespace()
ns.bind(seasoned)
ns.bind(fedfund_yc)
market = MarketContext(as_of, ns)
ns
# CurveNamespace({USD.SOFR(v0), USD.FEDFUND(v0)})
market.zero_curve("USD.FEDFUND")
# CompositeCurve(base=+1*ZeroCurve(nodes=6, ...), spread=+1*ZeroCurve(nodes=2, ...))
```

`RateGenerator` gives discount-factor ratios their financial meaning. A window that starts before
the origin is read from the fixings; one that starts after it is projected.

```python
gen = RateGenerator(market)
starts = np.array(["2026-05-29", "2026-09-01"], dtype="datetime64[D]")
ends = np.array(["2026-06-01", "2026-12-01"], dtype="datetime64[D]")
gen.simple_rate("USD.SOFR", starts, ends)          # [historical print, projected Act/360 simple rate]
# [0.0429   0.039323]
gen.simple_rate("USD.FEDFUND", starts[1:], ends[1:])
# [0.038825]

scenario_market = market.with_curve(sofr_yc.with_zero_curve(sofr.with_params(sofr.params + 0.0010)))
RateGenerator(scenario_market).simple_rate("USD.SOFR", starts[1:], ends[1:]) - gen.simple_rate("USD.SOFR", starts[1:], ends[1:])
# [0.000996]                                        # +10bp continuous ≈ +9.96bp simple Act/360
market.zero_curve("USD.SOFR") is sofr
# True
```

Note what the scenario did *not* do: it did not change `USD.FEDFUND`. Its spread composite still
holds a reference to the original `sofr` object. Moving the SOFR atom "through" every composite
that references it is the job of the shared-atom parameter layout
([risk_and_engine_design.md](../risk_and_engine_design.md) §4.1), not of `with_curve`; until
then, a scenario on a shared atom rebuilds each dependent composite with `with_component("base", ...)`.

## What `YieldCurve` deliberately does not do

It does not project rates, compute par rates, or price. `RateGenerator` owns forwards,
compounded and averaged observation rates and par swap rates; pricers own PV. It does not
calibrate: `CurveCalibrator` returns a `ZeroCurve` plus its origin, and registration through
`from_registry` is an explicit step ([curve_architecture.md](../curve_architecture.md)).

Next: [05 · Risk: Jacobian and Hessian](05_risk_jacobian_hessian.md).
