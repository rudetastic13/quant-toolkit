# 05 · `YieldCurve`: the landing place

A `YieldCurve` binds three things: an `origin` date, a `DiscountCurve` (the field is still named
`zero_curve`, but any protocol implementation sits there), and a curve `key`. Optionally it
carries `HistoricalFixings`. It is the only layer that sees dates, and it is the object
`MarketContext` registers and `RateGenerator` reads.

```text
DiscountCurve (02–04)  +  origin  +  key (+ index conventions for a projection key)  +  fixings
        │
        ▼
    DiscountCurve  ──market[key] = curve──▶  MarketContext  ──yield_curve(key)──▶  RateGenerator, pricers
```

Keys follow one grammar (`finance.conventions.CurveKey`). The slot count says what the curve is:

| form | kind | examples |
|---|---|---|
| `CCY.INDEX.TENOR` | projection: what a floating leg resets off; `ON` for overnight | `USD.SOFR.ON`, `USD.FEDFUND.ON`, `USD.CME-TERM-SOFR.3M` |
| `CCY.NAME` | discounting: a funding or collateral regime, or an issuer; no tenor | `USD.STDCSA`, `GBP.USDSTDCSA`, `USD.UST` |

A projection key carries its index family's conventions (day count, fixing rules, published
tenors); a discount key carries none, because nothing resets off it. A market registers every key
its trades name: a swap with `funding_id="STDCSA"` discounts off `USD.STDCSA`, and "discount at
SOFR" is the SOFR curve object bound under that key as well. There is no implicit default.

Continues from [01](01_discount_curve.md) and [03](03_composite_curve.md) (`sofr`, `fedfund`,
`dfs`).

## Construction

`from_registry` composes an existing discount curve with a key, looking a projection key's
conventions up in the registry; `build` is the dates-in convenience that constructs the
`ZeroCurve` for you (the first node is the origin). Name the curve by `key=` or, for a projection
curve, by `currency=` and `index_name=` (`"SOFR"` is the overnight family, `"CME-TERM-SOFR.3M"` a
term tenor).

```python
from finance.dates import Date, Term
from finance.markets.curves import YieldCurve, dates_to_x

as_of = Date(2026, 6, 1)
origin = as_of.to_numpy()
sofr_yc = YieldCurve.from_registry(origin, sofr, currency="USD", index_name="SOFR")
sofr_yc
# YieldCurve(name='USD.SOFR.ON', origin=2026-06-01, zero_curve=ZeroCurve(nodes=6, max_x=3650, space=LogDF, interpolator=Linear(), extrapolation=FlatForward))
sofr_yc.name
# 'USD.SOFR'
sofr_yc.node_dates
# ['2026-06-01' '2026-08-30' '2027-06-01' '2028-05-31' '2031-05-31' '2036-05-29']
sofr_yc.key
# CurveKey('USD.SOFR.ON')
sofr_yc.index
# RateIndex('USD.SOFR.ON')
sofr_yc.index.family
# IndexFamily('USD.SOFR', tenors=('ON',), day_count=Actual360, fixing_calendar='no_holidays', fixing_type=Arrears)

built = YieldCurve.build(sofr_yc.node_dates, dfs, currency="USD", index_name="SOFR", interpolator=Cubic())
built
# YieldCurve(name='USD.SOFR.ON', origin=2026-06-01, zero_curve=ZeroCurve(nodes=6, max_x=3650, space=LogDF, interpolator=Cubic(bc_type='natural'), extrapolation=FlatForward))

csa = sofr_yc.with_key("USD.STDCSA")               # the same state as a discount curve
csa.name, csa.index
# ('USD.STDCSA', None)
```

A projection key is validated against the conventions: its family must be registered and must
publish the tenor (`USD.SOFR.3M` is rejected). A discount key needs no conventions at all.

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

A Fed Funds swap projects off `USD.FEDFUND.ON` and discounts off the market's `USD.STDCSA`
curve. Its projection curve is the labelled spread composite from [03](03_composite_curve.md),
sharing the SOFR atom by reference.

```python
fedfund_yc = YieldCurve.from_registry(origin, fedfund, currency="USD", index_name="FEDFUND")
fedfund_yc
# YieldCurve(name='USD.FEDFUND.ON', origin=2026-06-01, zero_curve=CompositeCurve(base=+1*ZeroCurve(nodes=6, ...), spread=+1*ZeroCurve(nodes=2, ...)))
fedfund_yc.node_dates                              # the composite's union of pillars
# ['2026-06-01' '2026-08-30' '2027-06-01' '2028-05-31' '2031-05-31' '2036-05-29']
```

## In the market

`MarketContext` is a `Mapping[str, DiscountCurve]` keyed by curve key and the container pricers
read. Assigning the SOFR curve a second time under `USD.STDCSA` is how the market says "discount
at SOFR": one object, two keys. A `YieldCurve` can be assigned too (its fixings come with it);
what is stored is always the mathematical curve, and `yield_curve(key)` is the derived view.

```python
from finance.markets import RateGenerator
from finance.markets.context import MarketContext

market = MarketContext(as_of)
market["USD.SOFR.ON"] = seasoned                  # a YieldCurve: its ZeroCurve and its fixings
market["USD.STDCSA"] = market["USD.SOFR.ON"]      # the same object under the discount key
market["USD.FEDFUND.ON"] = market["USD.SOFR.ON"] - 5 * bp
market
# MarketContext(as_of=2026-06-01, curves=['USD.SOFR.ON', 'USD.STDCSA', 'USD.FEDFUND.ON'], mutable)
market["USD.FEDFUND.ON"]
# CompositeCurve(base=+1*ZeroCurve(nodes=6, ...), spread=+1*ZeroCurve(nodes=2, ...))
{k: id(v) for k, v in market.items()}             # one object under two keys
market.owned("USD.SOFR.ON"), market.owned("USD.STDCSA"), market.owned("USD.FEDFUND.ON")
# ((ZeroCurve(nodes=6, ...),), (), (ZeroCurve(nodes=2, ...),))
```

Ownership is by shortest path: an atom belongs to the key reaching it by the fewest composite
levels, a projection key beats a discount key at equal depth, and the first registered wins
among registered objects. So the SOFR atom belongs to `USD.SOFR.ON` wherever else it appears
(under `USD.STDCSA`, inside the Fed Funds composite), and `USD.FEDFUND.ON` owns only its spread.
The engines lay parameters out by that ownership ([05](05_risk_jacobian_hessian.md) §6): one
block per owning key, and a ladder asked under an alias is its owner's ladder.

`RateGenerator` gives discount-factor ratios their financial meaning. A window that starts before
the origin is read from the fixings; one that starts after it is projected.

```python
gen = RateGenerator(market)
starts = np.array(["2026-05-29", "2026-09-01"], dtype="datetime64[D]")
ends = np.array(["2026-06-01", "2026-12-01"], dtype="datetime64[D]")
gen.simple_rate("USD.SOFR.ON", starts, ends)          # [historical print, projected Act/360 simple rate]
# [0.0429   0.039323]
gen.simple_rate("USD.FEDFUND.ON", starts[1:], ends[1:])
# [0.038825]

scenario_market = market.bump("USD.SOFR.ON", 0.0010)        # +10bp on every SOFR pillar
RateGenerator(scenario_market).simple_rate("USD.SOFR.ON", starts[1:], ends[1:]) - gen.simple_rate("USD.SOFR.ON", starts[1:], ends[1:])
# [0.000996]                                        # +10bp continuous ≈ +9.96bp simple Act/360
market["USD.SOFR.ON"] is sofr
# True                                              # the original market is untouched
scenario_market["USD.STDCSA"] is scenario_market["USD.SOFR.ON"]
# True
scenario_market["USD.FEDFUND.ON"]["base"] is scenario_market["USD.SOFR.ON"]
# True                                              # Fed Funds was rebuilt on the moved SOFR atom
RateGenerator(scenario_market).simple_rate("USD.FEDFUND.ON", starts[1:], ends[1:]) - gen.simple_rate("USD.FEDFUND.ON", starts[1:], ends[1:])
# [0.000996]                                        # ... so it moved by the same amount; the basis did not
```

A scenario propagates by object identity. Bumping `USD.SOFR.ON` replaces the SOFR atom wherever
it appears: under `USD.STDCSA`, which holds the same object, and inside the Fed Funds composite,
which is rebuilt on the moved atom with its spread untouched. `bump("USD.FEDFUND.ON", ...)` moves
only the spread, because that is the only parameter Fed Funds owns. `with_curve(key, curve)` is
the same move with a replacement curve: on the key that owns the object it propagates the same
way; on an alias such as `USD.STDCSA` it changes that key alone. Both return frozen copies.

## What `YieldCurve` deliberately does not do

It does not project rates, compute par rates, or price. `RateGenerator` owns forwards,
compounded and averaged observation rates and par swap rates; pricers own PV. It does not
calibrate: `CurveCalibrator` returns a `ZeroCurve` plus its origin, and registration through
`from_registry` is an explicit step ([curve_architecture.md](../curve_architecture.md)).

Next: [05 · Risk: Jacobian and Hessian](05_risk_jacobian_hessian.md).
