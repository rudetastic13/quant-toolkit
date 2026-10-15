# maths/05 · Continuously compounded rates

The **zero rate** (spot rate) to time `t` is the single constant rate that, compounded over
`[0, t]`, reproduces the discount factor. With continuous compounding on the library's Act/365
clock:

```
DF(t) = exp(−z(t) · t)          ⇔          z(t) = −ln DF(t) / t,     t in years = days / 365
```

It is the curve's `zero_rate`, and in `ZeroRate` space it is the quantity interpolated. At the
pillars it is the parameter vector of every `ZeroCurve` (`params`, `node_zero_rates`).

Continues from [maths/03](03_discount_factors.md) (`df2`, `t2`).

```python
t_years = 730.0 / 365.0
z_2y = -np.log(df2[0]) / t_years
z_2y, sofr.zero_rate(t2)[0]
# (0.039, 0.039)
np.exp(-z_2y * t_years)
# 0.9249644265435393
```

## Why continuous compounding is the internal convention

It is the one convention in which rates **add**. A spread of `s` over a base is
`exp(−(z + s) t) = exp(−z t) · exp(−s t)`: the spread is its own discount factor and the log-DF
sum of [maths/04](04_log_discount_factors.md) is exact. Under annual or simple compounding a
"spread" is not a clean factor and composites would not be affine in their parameters. Market
quotes in other conventions are converted at the edge (the rate generator and the instruments),
never inside the curve.

## Converting to other conventions

Each convention is a different way of writing the same discount factor:

| Convention | Rate from `DF` over `t` years | `DF` from rate |
|---|---|---|
| continuous | `−ln DF / t` | `exp(−z t)` |
| annual | `DF^(−1/t) − 1` | `(1 + y)^(−t)` |
| semi-annual | `2 (DF^(−1/(2t)) − 1)` | `(1 + y/2)^(−2t)` |
| simple | `(1/DF − 1) / t` | `1 / (1 + r t)` |

```python
simple = (1.0 / df2[0] - 1.0) / t_years
annual = df2[0] ** (-1.0 / t_years) - 1.0
semi = 2.0 * (df2[0] ** (-1.0 / (2.0 * t_years)) - 1.0)
{"continuous": z_2y, "annual": annual, "semi-annual": semi, "simple": simple}
# {'continuous': 0.039, 'annual': 0.03977, 'semi-annual': 0.039383, 'simple': 0.040561}
```

Same discount factor, four numbers. The simple rate over a two-year window is the largest
because simple interest does not earn interest on interest; continuous is the smallest for the
opposite reason. Note that these are over the curve's Act/365 clock; a market simple rate also
carries its own day count (SOFR is Act/360), which the rate generator applies.

## The zero rate at the origin

`−ln DF / t` is `0/0` at `t = 0`. The limit is the instantaneous forward at the origin, and that
is what `zero_rate(0)` returns; in `ZeroRate` space the origin node's value is set the same way
so the interpolant has a sensible left end.

```python
sofr.zero_rate(np.array([0.0])), sofr.instantaneous_forward(np.array([0.0]))
# (array([0.041]), array([0.041]))
```

## Zero rates of a composite

Because `ln DF` adds across components and `z = −ln DF / t` is linear in `ln DF`, zero rates add
with the same weights: `z_total(t) = Σ wᵢ zᵢ(t)`. For a spread composite, `curve["spread"].zero_rate(t)` is the
continuously compounded spread to `t`, and `curve.zero_rate(t) - curve["base"].zero_rate(t)`
recovers the same number ([03](../03_composite_curve.md)).

Next: [maths/06 · Forward rates](06_forward_rates.md).
