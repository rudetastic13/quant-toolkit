# maths/03 · Discount factors

The discount factor `DF(t)` is the value today of one unit of currency paid at time `t`, with
certainty, in the currency and collateral regime the curve represents. It is the primitive of
every interest-rate calculation in the library: rates are ratios of discount factors, present
values are sums of cash flows times discount factors, and risk is the derivative of those sums.

Continues from [01](../01_discount_curve.md) (`sofr`).

## Invariants

```
DF(0) = 1                         money today is worth itself
DF(t) > 0                         a positive amount has positive value
DF(t) ≤ 1 for t > 0               under non-negative rates (not enforced: negative rates exist)
DF decreasing in t                under positive forwards (not enforced, for the same reason)
```

`ZeroCurve` enforces the first two at construction and rebinding and leaves the others to the
calibrator and the market.

```python
t1, t2 = np.array([365.0]), np.array([730.0])
df1, df2 = sofr.discount_factor(t1), sofr.discount_factor(t2)
df1, df2
# (array([0.960789]), array([0.924964]))
sofr.discount_factor(np.array([0.0]))
# [1.]
np.all(np.diff(sofr.dfs) < 0)
# True
```

## Forward discount factors

The value at `t₁` of one unit paid at `t₂ > t₁` is the ratio

```
DF(t₁, t₂) = DF(t₂) / DF(t₁)
```

This is the no-arbitrage chaining rule: discounting from `t₂` to today equals discounting from
`t₂` to `t₁` and then from `t₁` to today. Every forward rate in [maths/06](06_forward_rates.md)
is a function of this ratio.

```python
df2 / df1
# [0.962713]
```

## Present value

For cash flows `cᵢ` at times `tᵢ`:

```
PV = Σᵢ cᵢ · DF(tᵢ)
```

```python
100.0 * df2                                        # PV of 100 paid in two years
# [92.496443]
```

Linear in the cash flows, and, for fixed times, linear in the discount factors. That linearity is
why the gradient of a fixed leg with respect to the curve parameters is a weighted sum of the
discount factors' gradients ([05](../05_risk_jacobian_hessian.md) §2).

## Why the curve does not interpolate discount factors directly

Interpolating `DF` linearly between pillars gives a forward rate that jumps *and* drifts within
each interval, and it can produce a non-monotone curve from monotone pillars once rates are low.
Interpolating `ln DF` linearly gives a piecewise-constant forward, the shape the market uses to
bootstrap; interpolating the zero rate gives yet another. `CurveSpace` names the transform
([maths/04](04_log_discount_factors.md), [maths/05](05_continuous_rates.md)); the discount factor
is always recovered exactly at the pillars whatever the space.

Next: [maths/04 · Log discount factors](04_log_discount_factors.md).
