# maths/02 · Extrapolation

Extrapolation is what the function does beyond its first or last node. `Line1d` handles each
side independently with an `Extrapolation` mode; `ZeroCurve` fixes the left side (queries before
the origin are an error) and exposes the right side as `RateExtrapolator`.

Continues from [maths/01](01_interpolation.md) (`nodes_x`, `nodes_y`) and
[01](../01_discount_curve.md) (`x`, `dfs`, `sofr`).

## `Line1d`: three modes

| `Extrapolation` | Value beyond the end node | Slope |
|---|---|---|
| `NotAllowed` (default) | raises `ValueError` naming the furthest offender | — |
| `Flat` | holds the end node's value | 0 |
| `Linear` | continues the fitted interpolant's end slope | the end slope |

```python
beyond = np.array([7.0, 8.0])
for mode in (Extrapolation.Flat, Extrapolation.Linear):
    line = Line1d(nodes_x, nodes_y, Cubic(), right=mode)
    print(mode.name, line(beyond), line.derivative(beyond))
# Flat   [2. 2.]                 [0. 0.]
# Linear [2.860119 3.720238]     [0.860119 0.860119]

Line1d(nodes_x, nodes_y, Cubic())(beyond)
# ValueError: 2 query point(s) beyond the last node x=6.0 (furthest: 8.0); extrapolation is not allowed.
```

`Linear` uses the *exact* end slope of the fitted spline (the derivative of the last piece at the
end node), not a difference quotient of the last two nodes, so a cubic extrapolates tangentially.

## `ZeroCurve`: flat forward

The financially meaningful rule for a discount curve is to continue the **terminal instantaneous
forward**: beyond the last pillar, money accrues at the last forward the curve knows. In the two
spaces that is implemented differently and means the same thing:

- **`LogDF` space.** A constant forward `f` is a straight line in `ln DF` with slope `-f/365`,
  so `Extrapolation.Linear` on the line *is* flat-forward.
- **`ZeroRate` space.** A constant forward is not a straight line in the zero rate (the zero rate
  decays towards the forward as `z(T) = (z_N x_N + f (T − x_N)) / T`), so the curve holds the line
  flat as a placeholder and applies the flat-forward continuation in the DF transform itself.

```python
far = np.array([3650.0, 4000.0, 9000.0])
rate_linear = ZeroCurve(x, dfs, space=CurveSpace.ZeroRate, interpolator=Linear())
sofr.instantaneous_forward(far)                    # LogDF/Linear: last segment's forward, 4.5%
# [0.045 0.045 0.045]
rate_linear.instantaneous_forward(far)             # ZeroRate/Linear: f = z + x z' at the end, 4.7%
# [0.047 0.047 0.047]
rate_linear.zero_rate(np.array([3650.0, 9000.0, 36500.0]))      # the zero rate decays to the forward
# [0.043    0.045378 0.0466  ]
```

Same pillar discount factors, different terminal forwards: the scheme owns the slope at the last
node, and the slope is what gets continued. This is one reason the choice of space is a modelling
decision and not a detail.

`RateExtrapolator.NotAllowed` turns the right side into an error, for curves that must never be
queried past their last instrument.

## Why extrapolation is part of the affine map

Both `Flat` and `Linear` are linear in the node values: `Flat` returns `y_N`, `Linear` returns
`y_N + y'(x_N)(x − x_N)` where `y'(x_N)` is linear in `y` ([maths/01](01_interpolation.md)). The
flat-forward continuation in `ZeroRate` space is `ln DF_N − f_N (x − x_N)/365` with `f_N` linear
in the node rates. So a query beyond the last pillar still has an exact affine row; in
[01](../01_discount_curve.md) the 5000-day query showed it, loading the 5Y and 10Y pillars with
large opposite weights because the terminal forward is a difference quotient of the last two
nodes.

## Composites

A `CompositeCurve` does not extrapolate; each component does, by its own rule. A two-node
flat-forward spread atom therefore continues its constant spread for ever, while a base with
`NotAllowed` still rejects a query past its last pillar ([03](../03_composite_curve.md)).

Next: [maths/03 · Discount factors](03_discount_factors.md).
