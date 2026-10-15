# maths/04 · Log discount factors

`ln DF(t)` is the natural coordinate for a discount curve. Three facts make it so.

Continues from [maths/03](03_discount_factors.md) (`df1`, `df2`, `t1`, `t2`) and
[03](../03_spread_curve.md) (`fedfund`).

## 1. It turns the chaining rule into addition

```
ln DF(t₁, t₂) = ln DF(t₂) − ln DF(t₁)
```

```python
np.allclose(np.log(df2 / df1), sofr.log_discount_factor(t2) - sofr.log_discount_factor(t1))
# True
np.log(dfs)                                        # the node values the default ZeroCurve interpolates
# [ 0.      -0.01011 -0.04    -0.078   -0.205   -0.43   ]
```

## 2. Its slope is minus the instantaneous forward

```
f(t) = −d ln DF / dt            (annualised on the curve's Act/365 clock)
```

So a straight line in `ln DF` is a constant forward, and `Linear` interpolation in `LogDF` space
is the piecewise-flat-forward curve of a classic bootstrap:

```python
sofr.instantaneous_forward(np.array([400.0, 600.0, 700.0]))      # anywhere inside the 1Y–2Y interval
# [0.038 0.038 0.038]
```

Flat-forward extrapolation is linear extrapolation of this line ([maths/02](02_extrapolation.md)).
The forward's full derivation is in [maths/06](06_forward_rates.md).

## 3. It is additive across curves

A spread, a basis, a turn: each is a discount curve of its own, and combining it with a base is
a *sum* in log-DF, a *product* in DF:

```
ln DF_total = ln DF_base + ln DF_spread          ⇔          DF_total = DF_base · DF_spread
```

```python
np.allclose(fedfund.log_discount_factor(xq), sofr.log_discount_factor(xq) + fedfund.spread.log_discount_factor(xq))
# True
```

This is the whole mathematics of `SpreadCurve` and `CompositeCurve` ([03](../03_spread_curve.md),
[04](../04_composite_curve.md)): `ln DF = Σ wᵢ ln DFᵢ`.

## Linearity in the parameters

Put the three facts together with [maths/01](01_interpolation.md). In `LogDF` space the node
values are `ln DF_j = −z_j x_j / 365`, linear in the pillar zero rates `z`; the interpolant is
linear in its node values; extrapolation is linear; a composite is a weighted sum. Therefore

```
ln DF(x) = offset(x) + J(x) · z           exactly, for every x, every scheme, every composite
```

which is `DiscountCurve.log_df_affine` and the single contract the engines are built on
([01](../01_discount_curve.md)). The same holds in `ZeroRate` space because `ln DF = −r(x) x/365`
and `r` is the interpolant of the `z_j` themselves.

Next: [maths/05 · Continuously compounded rates](05_continuous_rates.md).
