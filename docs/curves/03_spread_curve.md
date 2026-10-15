# 03 · `SpreadCurve`: base plus spread

`SpreadCurve(base, spread)` is a discount curve whose log-DF is the sum of two others:

```
ln DF(x) = ln DF_base(x) + ln DF_spread(x)        ⇔        DF = DF_base · DF_spread
z(x)     = z_base(x)     + z_spread(x)
f(x)     = f_base(x)     + f_spread(x)
```

The spread is itself a `DiscountCurve`: its zero rate at `x` *is* the continuously compounded
Act/365 spread to `x`. Parameters are `base.params` followed by `spread.params`, so risk on a
spread curve arrives already split into a base block and a spread block.

Continues from the setup in [01](01_discount_curve.md).

## Constant spread: `SpreadCurve.flat`

The common case: an index that trades at a roughly fixed margin to another. The spread curve is a
two-node log-linear `ZeroCurve` on `[0, horizon]` whose single parameter is the spread;
flat-forward extrapolation continues it unchanged for every `x`, so `horizon` only names the
pillar.

```python
fedfund = SpreadCurve.flat(sofr, -0.0005)         # Fed Funds 5bp under SOFR in this snapshot
fedfund
# SpreadCurve(base=ZeroCurve(nodes=6, max_x=3650, space=LogDF, interpolator=Linear(), extrapolation=FlatForward),
#             spread=ZeroCurve(nodes=2, max_x=3650, space=LogDF, interpolator=Linear(), extrapolation=FlatForward))
fedfund.params
# [ 0.041   0.04    0.039   0.041   0.043  -0.0005]
fedfund.base_slice, fedfund.spread_slice
# (slice(0, 5, None), slice(5, 6, None))

fedfund.spread_rate(xq)                            # the spread component's zero rate
# [-0.0005 -0.0005 -0.0005]
fedfund.zero_rate(xq) - sofr.zero_rate(xq)
# [-0.0005 -0.0005 -0.0005]
np.allclose(fedfund.discount_factor(xq), sofr.discount_factor(xq) * fedfund.spread.discount_factor(xq))
# True
```

## Term structure of spread

Any `DiscountCurve` can be the spread. A funding spread that widens with tenor:

```python
sx = np.array([0.0, 365.0, 1825.0, 3650.0])
sz = np.array([0.0, 0.0008, 0.0012, 0.0015])      # 8bp at 1Y, 12bp at 5Y, 15bp at 10Y
sdfs = np.exp(-sz * sx / 365.0)
sdfs[0] = 1.0
funding = SpreadCurve(sofr, ZeroCurve(sx, sdfs))

funding.x                                          # union of base and spread pillars
# [   0.   90.  365.  730. 1825. 3650.]
funding.n_params                                   # 5 base + 3 spread
# 8
funding.spread_rate(xq)
# [0.0008   0.000935 0.001581]
funding.instantaneous_forward(xq) - sofr.instantaneous_forward(xq)    # forward spread, piecewise constant
# [0.0008 0.0013 0.0018]
```

## Scenarios on one role

`with_base` and `with_spread` replace a role with a curve of any geometry; `with_params` moves
both blocks from one flat vector. The untouched role is shared by reference, not copied.

```python
wider = funding.with_spread(funding.spread.with_params(funding.spread.params + 0.0005))
wider.spread_rate(xq) - funding.spread_rate(xq)
# [0.0005 0.0005 0.0005]
wider.base is sofr
# True
```

## Risk splits by block

The affine map is the base's map and the spread's map side by side. The base block is the
atom's own Jacobian, unchanged; the spread block is the spread atom's Jacobian.

```python
offset_s, jac_s = funding.log_df_affine(xq)
jac_s.shape
# (3, 8)
np.allclose(jac_s[:, funding.base_slice], sofr.log_df_affine(xq)[1])
# True
jac_s[:, funding.spread_slice]
# [[ -0.082192   0.         0.      ]
#  [ -0.907534  -0.462329   0.      ]
#  [  0.         3.69863  -17.39726 ]]
```

So a book priced on `funding` reports a ladder over SOFR pillars (index or funding risk, depending
on the role) and a separate ladder over the spread pillars (basis risk). A spread curve registered
as its own index (`USD.FEDFUND` over the SOFR atom) does this through the engine with no extra
code; [06](06_risk_jacobian_hessian.md) shows the output.

## When is a spread a curve, and when is it a convention?

Two different things look like "index A plus a margin":

- **A risk factor.** Fed Funds over SOFR, a funding or CSA spread, a z-spread on a bond. The
  spread moves, the desk wants to see and bump it, hedges depend on it. That is a `SpreadCurve`:
  a curve with parameters.
- **A contractual redefinition.** An index that is *defined* as another index plus a fixed
  number (a fallback-rate index, or a legacy index quoted as a benchmark plus a fixed spread
  adjustment). The spread has no risk because it cannot change. That belongs on the leg or the
  `RateIndex` convention, in the rate generator, not in a curve, or a parameter would be reported
  with zero risk forever.

The library does not decide for you; both are expressible. Prefer the convention when the
spread is fixed by contract, the curve when it is a market quote.

## As a `DiscountCurve`

`SpreadCurve` wraps a `CompositeCurve((base, spread))` with weights `(1, 1)` and delegates every
protocol member to it; `.composite` exposes the wrapped object. It adds the role accessors
(`base`, `spread`, `base_slice`, `spread_slice`), `spread_rate`, `with_base` / `with_spread`,
and the `flat` constructor. `scheme` is `("spread", composite.scheme)`.

Next: [04 · `CompositeCurve`](04_composite_curve.md).
