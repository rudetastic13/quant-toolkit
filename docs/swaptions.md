# Swaptions

A European swaption is the right to enter a fixed/float swap on the expiry date. This page
shows how to build one, price it and read its risk. Every number below comes from running the
snippets in order with `PYTHONPATH=src`.gvbhffn

> **Setup:** European exercise, physical settlement, long positions only. Bachelier (normal
> vols) and Black (lognormal vols) are implemented. No smile model exists yet, so the vol does
> not depend on strike.

## 1. Markett
A swaption needs a curve and a vol cube. A `ModelVolCube` is a surface plus the units its vols quote convention.

```python
import numpy as np
from quant_toolkit.finance.dates import Date
from quant_toolkit.finance.markets.context import MarketContext
from quant_toolkit.finance.markets.curves import YieldCurve
from quant_toolkit.finance.markets.vols import FlatVolSurface, GridVolSurface, ModelVolCube, VolNamespace, VolUnits

as_of = Date(2026, 6, 1)
origin = as_of.to_numpy()
dates = np.array([origin + np.timedelta64(d, "D") for d in (0, 365, 3 * 365, 7 * 365, 12 * 365)])
t = (dates.astype(np.int64) - dates[0].astype(np.int64)) / 365.0
zeros = np.array([0.0, 0.035, 0.040, 0.045, 0.047])
sofr = YieldCurve.build(dates, np.exp(-zeros * t), key="USD.SOFR.1D")

# normal vols on an expiry x tenor grid (rows: 1Y, 2Y, 5Y expiry; columns: 2Y, 5Y, 10Y tenor)
grid = GridVolSurface(
    expiries=np.array([1.0, 2.0, 5.0]),
    tenors=np.array([2.0, 5.0, 10.0]),
    values=np.array([
        [0.0105, 0.0100, 0.0092],
        [0.0102, 0.0097, 0.0090],
        [0.0094, 0.0090, 0.0085],
    ]),
)

vols = VolNamespace()
vols.bind("USD.SOFR.1D", ModelVolCube(grid, VolUnits.Normal))                      # for Bachelier
vols.bind("USD.SOFR.LN", ModelVolCube(FlatVolSurface(0.24), VolUnits.Lognormal))   # for Black

market = MarketContext(as_of, vols=vols)
market["USD.SOFR.1D"] = sofr     # projection curve
market["USD.STDCSA"] = sofr      # the same curve used for discounting
```

The grid is bilinear between nodes and flat beyond its edges. A cube is looked up by name; by
default a swaption uses the cube named after its rate index.

## 2. Construction

`Swaption.european` builds the option and its underlying swap from a handful of terms.

```python
from quant_toolkit.finance.instruments.resolution import Swaption, SwaptionModel

payer = Swaption.european(
    notional=10e6, strike=0.045, expiry="1Y", swap_tenor="5Y", as_of=as_of,
    payer=True, model=SwaptionModel.Bachelier,
)
receiver = Swaption.european(
    notional=10e6, strike=0.045, expiry="18M", swap_tenor="7Y", as_of=as_of,
    payer=False, model=SwaptionModel.Bachelier,
)
black = Swaption.european(
    notional=10e6, strike=0.045, expiry="1Y", swap_tenor="5Y", as_of=as_of,
    payer=True, model=SwaptionModel.Black, vol_name="USD.SOFR.LN",
)
```

- `payer=True` is the right to pay fixed; `payer=False` is the right to receive it.
- `strike` becomes the fixed rate of the underlying swap.
- `model` must match the units of the cube it reads: Bachelier needs `Normal`, Black needs
  `Lognormal`. A mismatch raises.
- `vol_name` picks a cube other than the default.

The underlying is an ordinary `Swap`, available as `payer.underlying`. For the payer above the
option expires 2027-06-01 and the swap runs 2027-06-03 to 2032-06-03.

## 3. Pricing

Compile a list of swaptions once, then reprice against any market.

```python
from quant_toolkit.finance.pricing.pricers import SwaptionPricer

program = SwaptionPricer().compile([payer, receiver, black])
result = program.price(market)
```

Each field is an array with one entry per swaption.

| | payer 1Y×5Y | receiver 18M×7Y | payer 1Y×5Y (Black) |
|---|---:|---:|---:|
| `result.pv` | 205,659.89 | 187,609.87 | 221,845.57 |
| `result.forward` | 4.6519% | 4.7923% | 4.6519% |
| `result.annuity` | 42,894,784 | 55,981,116 | 42,894,784 |
| `result.volatility` | 0.9999% | 0.9549% | 24.00% |

- `forward` is the forward swap rate and `annuity` is the PV of 1.0 of fixed coupon on the
  notional. Both come from pricing the underlying swap.
- `volatility` is what the cube returned. The 18M×7Y point sits between grid nodes, so it is
  interpolated.
- `pv` is `annuity × option value`, where the option value is Bachelier or Black on the forward.

`result.greeks` holds the monetary greeks with respect to the forward rate and the vol.

| | payer 1Y×5Y | receiver 18M×7Y | payer 1Y×5Y (Black) |
|---|---:|---:|---:|
| `greeks.delta` | 24,037,068 | −22,469,539 | 25,819,351 |
| `greeks.gamma` | 1.692e9 | 1.850e9 | 1.482e9 |
| `greeks.vega` | 16,916,216 | 26,524,499 | 769,935 |

These are per unit move (1.0) in the forward or the vol. Multiply delta by `1e-4` for a 1bp
move in the forward. Multiply vega by `1e-4` for 1bp of normal vol, or by `0.01` for one point
of lognormal vol. `vanna` and `volga` are also available.

## 4. Risk

`risk` returns curve risk and vol risk in one call.

```python
risk = program.risk(market, "USD.SOFR.1D")
```

### Curve risk

`risk.dv01` has one row per swaption and one column per curve pillar. Each entry is the PV
change for a 1bp rise in that pillar's zero rate.

| `risk.dv01` | 1Y | 3Y | 7Y | 12Y | total |
|---|---:|---:|---:|---:|---:|
| payer 1Y×5Y | −529.57 | 480.47 | 2,441.94 | 0.00 | 2,392.84 |
| receiver 18M×7Y | 278.10 | 127.75 | −1,755.31 | −1,064.25 | −2,413.71 |
| payer 1Y×5Y (Black) | −568.85 | 515.93 | 2,622.82 | 0.00 | 2,569.90 |

A payer gains when rates rise and a receiver loses. `risk.curve_gradient` is the same table
per unit move instead of per basis point.

The vol is held fixed while the curve moves (sticky strike).

### Vol risk

`risk.vol_gradient` is a dict keyed by cube name. Each value has one row per swaption and one
column per cube parameter, and is the PV change per unit move in that parameter.

```python
node_vega = risk.vol_gradient["USD.SOFR.1D"] * 1e-4     # per 1bp of normal vol
node_vega[0].reshape(3, 3)                              # payer, laid out like the grid
```

Payer 1Y×5Y, per 1bp of normal vol at each node:

| expiry \ tenor | 2Y | 5Y | 10Y |
|---|---:|---:|---:|
| 1Y | 0.00 | 1,689.77 | 1.85 |
| 2Y | 0.00 | 0.00 | 0.00 |
| 5Y | 0.00 | 0.00 | 0.00 |

Receiver 18M×7Y:

| expiry \ tenor | 2Y | 5Y | 10Y |
|---|---:|---:|---:|
| 1Y | 0.00 | 792.11 | 530.49 |
| 2Y | 0.00 | 796.46 | 533.40 |
| 5Y | 0.00 | 0.00 | 0.00 |

- The payer sits almost exactly on the 1Y×5Y node, so nearly all of its vega lands there. The
  small 10Y entry is because the swap starts two days after expiry, making its tenor slightly
  over five years.
- The receiver sits between four nodes and its vega is split across them by the interpolation
  weights.
- Each swaption's node vegas add up to its own vega: 1,691.62 and 2,652.45 here, which are
  `greeks.vega × 1e-4`.

A swaption priced off a different cube has a zero row. The Black swaption appears under its own
cube instead, which has a single parameter:

```python
risk.vol_gradient["USD.SOFR.LN"][2] * 0.01     # [7699.35], per one point of lognormal vol
```

## 5. What is not here yet

- **Smile.** The cube accepts a `SmileModel`, but none is implemented, so strike does not
  affect the vol.
- **Smile-adjusted delta.** With a smile model the vol moves when the forward moves. The plan
  is to report that as its own column next to the fixed-vol DV01 shown above.
- **Short positions and cash settlement.**
