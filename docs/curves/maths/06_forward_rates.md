# maths/06 · Forward rates

A forward rate is the rate for a period that starts in the future, implied today by the curve.
Everything a floating leg projects is a forward; the instantaneous forward is the curve's most
sensitive diagnostic, where an interpolation scheme shows its character.

Continues from [01](../01_discount_curve.md) (`sofr`) and [02](../02_zero_curve.md)
(`rate_cubic`), [maths/03](03_discount_factors.md) (`df1`, `df2`, `t1`, `t2`).

## Instantaneous forward

The rate for an infinitesimal period at `t`:

```
f(t) = −d ln DF(t) / dt
```

In `LogDF` space it is minus the slope of the interpolated line. In `ZeroRate` space, with
`ln DF = −z(t) t`, the product rule gives the transform on the continuously compounded rate:

```
f(t) = z(t) + t · z′(t)
```

The zero rate is the *average* of the forwards to `t`; the forward is the zero rate plus a term
that says how fast the average is changing. A rising zero curve has forwards above it, a falling
one below.

```python
h = 1e-3
probe = np.array([500.0, 1000.0, 3000.0])
fd = -(sofr.log_discount_factor(probe + h) - sofr.log_discount_factor(probe - h)) / (2.0 * h) * 365.0
sofr.instantaneous_forward(probe), fd
# (array([0.038   , 0.042333, 0.045   ]), array([0.038   , 0.042333, 0.045   ]))

zq = rate_cubic.zero_rate(probe)
dz = (rate_cubic.zero_rate(probe + h) - rate_cubic.zero_rate(probe - h)) / (2.0 * h)
rate_cubic.instantaneous_forward(probe), zq + probe * dz          # f = z + t z′ (t and z′ both in days, product in years)
# (array([0.037985, 0.039867, 0.044529]), array([0.037985, 0.039867, 0.044529]))
```

`Line1d.derivative` supplies `z′` exactly from the fitted piecewise polynomial, so
`instantaneous_forward` is analytic, not a finite difference.

## Interval forwards

The rate for `[t₁, t₂]` is set by the forward discount factor `DF(t₁, t₂) = DF(t₂)/DF(t₁)`
([maths/03](03_discount_factors.md)), written in whichever convention the instrument needs:

```
continuous (Act/365):   F = −(ln DF(t₂) − ln DF(t₁)) / (t₂ − t₁)        the average of f over the interval
simple (day count τ):   L = (DF(t₁)/DF(t₂) − 1) / τ                      what an index fixing is
```

```python
cont_fwd = -(sofr.log_discount_factor(t2) - sofr.log_discount_factor(t1)) / ((t2 - t1) / 365.0)
simple_fwd = (df1 / df2 - 1.0) / ((t2 - t1) / 360.0)                   # Act/360, like a SOFR term rate
cont_fwd, simple_fwd
# (array([0.038]), array([0.038201]))
np.mean(sofr.instantaneous_forward(np.linspace(365.0, 730.0, 20001)))
# 0.03800021665583385
```

`RateGenerator.simple_rate` and `continuous_forward_rate` are these two formulas over dates,
with the index's day count and the historical-fixing overlay ([05](../05_yield_curve.md)).

## Compounded and averaged forwards

An overnight index swap coupon compounds daily forwards: `(Π(1 + Lₘ τₘ) − 1) / Σ τₘ`. Because each
`Lₘ τₘ = DF(tₘ)/DF(tₘ₊₁) − 1`, the product telescopes to `DF(t_start)/DF(t_end) − 1` when there
are no spreads, floors or weights, which is why a plain compounded leg has the same value as a
single forward over the whole period and why the floating leg of a par swap in
[06](../06_risk_jacobian_hessian.md) §4 collapses to `1 − DF(T)`. Weights, spreads, caps and
fixings break the telescoping, and the engine then does the product explicitly.

## Forwards of a composite

Forwards add with the composite's weights, `f_total = Σ wᵢ fᵢ`, because the derivative of a sum
is the sum of derivatives. A flat spread adds a constant to every forward; a turn adds a spike in
one window and nothing elsewhere ([04](../04_composite_curve.md)).

## Forwards and risk

The forward is the first derivative of `ln DF` in *time*. Risk is the first derivative of `ln DF`
in the *parameters*. The two are unrelated except through the scheme: a global scheme (cubic)
makes both the forward and the parameter Jacobian non-local, which is why a cubic curve shows
smooth forwards and dense risk ladders while a log-linear curve shows stepped forwards and
banded ladders. [06](../06_risk_jacobian_hessian.md) takes the parameter derivative from here.
