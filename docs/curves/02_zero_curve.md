# 02 · `ZeroCurve`: the atom

A `ZeroCurve` is a `common.math.line.Line1d` in a declared `CurveSpace`, plus the
discount-factor invariants. It knows nothing about dates, indices, fixings or coupons; its axis is
float days from the origin. `YieldCurve` ([05](05_yield_curve.md)) adds the calendar.

Continues from the setup in [01](01_discount_curve.md) (`x`, `z`, `dfs`, `sofr`, `xq`).

## Space

| `CurveSpace` | Node value interpolated | DF recovered as |
|---|---|---|
| `LogDF` (default) | `y = ln DF(x)` | `exp(y)` |
| `ZeroRate` | `y = r(x)`, Act/365 continuously compounded | `exp(-r · x / 365)` |

Linear in `LogDF` is piecewise-constant forwards (the market standard for a bootstrapped
front end). `ZeroRate` with a spline gives smooth forwards. The choice is a modelling decision;
the engines take either because of the affine contract.

```python
from finance.markets.curves import Cubic, CurveInterpolator, Linear, Mixed, Quadratic, RateExtrapolator

rate_cubic = ZeroCurve(x, dfs, space=CurveSpace.ZeroRate, interpolator=Cubic())
rate_cubic
# ZeroCurve(nodes=6, max_x=3650, space=ZeroRate, interpolator=Cubic(bc_type='natural'), extrapolation=FlatForward)

np.allclose(rate_cubic.discount_factor(x), sofr.discount_factor(x))    # same pillars ...
# True
sofr.zero_rate(xq), rate_cubic.zero_rate(xq)                            # ... different between them
# (array([0.041  , 0.03946, 0.04354]), array([0.041218, 0.039546, 0.043281]))
sofr.instantaneous_forward(xq), rate_cubic.instantaneous_forward(xq)
# (array([0.041, 0.038, 0.045]), array([0.041109, 0.037985, 0.044042]))
```

## Interpolators

Any `common.math.interpolation` scheme: `Linear`, `Cubic(bc_type)`, `Quadratic`, `Flat`, and
`Mixed(short, long, switch_node)` for one scheme up to a pillar and another beyond it. The
parameters travel with the frozen interpolator instance, so `Cubic("not-a-knot")` and
`Cubic("natural")` are different schemes. Details and figures: [maths/01](maths/01_interpolation.md)
and [curves_line1d.md](../curves_line1d.md).

```python
mixed = ZeroCurve(x, dfs, interpolator=Mixed(Linear(), Cubic(), switch_node=3))   # log-linear to 2Y, cubic after
mixed
# ZeroCurve(nodes=6, max_x=3650, space=LogDF, interpolator=Mixed(short=Linear(), long=Cubic(bc_type='natural'), switch_node=3), extrapolation=FlatForward)
mixed.instantaneous_forward(xq)
# [0.041    0.038    0.045833]
```

`CurveInterpolator` is a parse table for config and Excel; `resolve()` returns the
`(space, interpolator)` pair. Code that constructs curves directly passes the objects.

```python
space, interpolator = CurveInterpolator.LogCubicDF.resolve()
space, interpolator
# (<CurveSpace.LogDF: 1>, Cubic(bc_type='natural'))
ZeroCurve(x, dfs, space=space, interpolator=interpolator).instantaneous_forward(xq)
# [0.041075 0.037757 0.045281]
```

## Construction and validation

`ZeroCurve(x, dfs, *, space, interpolator, extrapolation)`. Checks: `x` 1-D, at least two
nodes, strictly increasing, `x[0] == 0`; `dfs` finite, positive, `dfs[0] == 1`. Everything else
is the `Line1d`'s job.

## Rebinding: `with_dfs` and `with_params`

Both return a new curve with the same pillars, space, interpolator and extrapolation. `with_dfs`
takes discount factors (the calibrator's natural output); `with_params` takes pillar zero rates
(the risk engine's natural input). They are the same operation in different coordinates.

```python
new_dfs = dfs.copy()
new_dfs[3] *= np.exp(-0.001 * x[3] / 365.0)     # 2Y zero rate +10bp
rebound = sofr.with_dfs(new_dfs)
rebound.node_zero_rates - sofr.node_zero_rates
# [0.    0.    0.001 0.    0.   ]
np.allclose(rebound.dfs, sofr.with_params(rebound.params).dfs)
# True
```

`node_zero_rates` and `params` are the same array; the first is the finance name, the second the
protocol name.

## Extrapolation

Queries before the origin always raise. Beyond `max_x`, `RateExtrapolator.FlatForward`
(default) continues the terminal instantaneous forward; `NotAllowed` raises. In `LogDF` space a
constant forward is a straight line in `ln DF`, so `Line1d`'s linear extrapolation does it
directly; in `ZeroRate` space the curve applies the flat-forward continuation itself.

```python
sofr.instantaneous_forward(np.array([3650.0, 4000.0, 9000.0]))
# [0.045 0.045 0.045]

strict = ZeroCurve(x, dfs, extrapolation=RateExtrapolator.NotAllowed)
strict.discount_factor(np.array([4000.0]))
# ValueError: 1 query point(s) beyond the last node x=3650.0 (furthest: 4000.0); extrapolation is not allowed.
```

More in [maths/02](maths/02_extrapolation.md).

## As a `DiscountCurve`

| Member | `ZeroCurve` |
|---|---|
| `x`, `dfs`, `max_x` | the pillars as given |
| `params` / `n_params` | `node_zero_rates`, `len(x) - 1` |
| `scheme` | `(space, interpolator)` |
| `with_params` | `with_dfs(exp(-params · x[1:] / 365))` |
| `log_df_affine` | probed one pillar at a time, reconstruction checked |
| `line` | the underlying `Line1d`; `line.coefficients` are PPoly coefficients for a kernel that wants to evaluate the curve itself |

Next: [03 · `SpreadCurve`](03_spread_curve.md).
