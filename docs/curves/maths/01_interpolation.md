# maths/01 · Interpolation

Interpolation fills in a function between known nodes. In this library it lives in
`common.math.line.Line1d` (nodes + an interpolator + per-side extrapolation) and
`common.math.interpolation` (the schemes). A curve never interpolates discount factors directly:
it interpolates a *transform* of them, `ln DF` or the zero rate, chosen by `CurveSpace`
([maths/04](04_log_discount_factors.md), [maths/05](05_continuous_rates.md)).

## Schemes

Given nodes `(xᵢ, yᵢ)`, each scheme defines `y(x)` on `[x₀, xₙ]`:

| Scheme | `y(x)` | Continuity | Locality |
|---|---|---|---|
| `Flat` | `yᵢ` on `[xᵢ, xᵢ₊₁)` | C⁻¹ (steps) | local |
| `Linear` | straight line between neighbours | C⁰ | local: a node moves only its two intervals |
| `Quadratic(left_slope, right_slope)` | piecewise quadratic, knot slopes from blended forward/backward sweeps | C⁰, C¹ where the sweeps agree | semi-local |
| `Cubic(bc_type)` | cubic spline; `natural` (zero end curvature), `clamped`, `not-a-knot`, or prescribed end derivatives | C² | global: every node moves the whole curve |
| `Mixed(short, long, switch_node)` | one scheme up to a node, another after; fit independently | C⁰ at the switch | per segment |

```python
from common.math.line import Extrapolation, Line1d
from finance.markets.curves import Cubic, Linear, Mixed, Quadratic

nodes_x = np.array([0.0, 1.0, 2.0, 4.0, 6.0])
nodes_y = np.array([1.0, 0.5, 1.5, 1.0, 2.0])
mid = np.array([0.5, 1.5, 3.0, 5.0])
for scheme in (Linear(), Cubic(), Cubic(bc_type="not-a-knot"), Quadratic(), Mixed(Linear(), Cubic(), switch_node=2)):
    line = Line1d(nodes_x, nodes_y, scheme)
    print(scheme, line(mid), line.derivative(mid))
# Linear()                        [0.75 1.   1.25 1.5 ]                      [-0.5   1.   -0.25  0.5 ]
# Cubic(bc_type='natural')        [0.577009 0.956473 1.497768 1.229911]      [-0.615327  1.201637 -0.512649  0.59003 ]
# Cubic(bc_type='not-a-knot')     [0.367187 1.007812 1.5625   0.8125  ]      [-0.369792  1.130208 -0.416667  0.333333]
# Quadratic(0.0, 0.0)             [0.875   0.78125 1.75    1.1875 ]          [-0.5   1.   -0.25  0.5 ]
# Mixed(Linear, Cubic, 2)         [0.75     1.       1.109375 1.359375]      [-0.5      1.       -0.296875  0.546875]
```

Every scheme reproduces its nodes exactly:

```python
np.allclose(Line1d(nodes_x, nodes_y, Cubic())(nodes_x), nodes_y)
# True
```

Figures comparing the schemes, and the forward panels they produce on a real curve, are in
[curves_line1d.md](../../curves_line1d.md).

## Piecewise-polynomial form

Every scheme except `Flat` is a scipy `PPoly` underneath. `coefficients` has shape
`(degree + 1, n_intervals)`, highest power first, each piece in powers of `x - xᵢ`. An engine that
wants to evaluate the curve itself does `searchsorted` + Horner on that array.

```python
Line1d(nodes_x, nodes_y, Linear()).coefficients
# [[-0.5   1.   -0.25  0.5 ]          slopes
#  [ 1.    0.5   1.5   1.  ]]         values at the left node of each interval
```

## The property everything rests on: linearity in `y`

For a fixed set of `x` nodes and a fixed scheme (including fixed boundary conditions), every
scheme above is a **linear map from node values to the interpolant**: interpolating `y + y'`
gives the sum of the two interpolants.

```python
y2 = np.array([0.2, -0.1, 0.4, 0.0, 0.3])
a = Line1d(nodes_x, nodes_y, Cubic())
b = Line1d(nodes_x, y2, Cubic())
c = Line1d(nodes_x, nodes_y + y2, Cubic())
np.allclose(a(mid) + b(mid), c(mid))
# True
```

Why it holds: `Linear` and `Flat` are obviously linear in `y`. The natural, clamped and
not-a-knot cubic splines solve a tridiagonal system whose matrix depends on `x` only and whose
right-hand side is linear in `y`. `Quadratic`'s knot slopes are cumulative sums of divided
differences, linear in `y`. `Mixed` is two such maps side by side.

Consequence: `y(x) = Σᵢ yᵢ · φᵢ(x)` for basis functions `φᵢ` that depend on the grid and scheme
only. In a `ZeroCurve` the node values are linear in the zero rates, so `ln DF` is affine in the
parameters, which is the engine contract of [01](../01_discount_curve.md) and the reason the
second derivative of `ln DF` with respect to the parameters is zero
([06](../06_risk_jacobian_hessian.md)).

Schemes that are **not** linear in `y`, and therefore are not in the library: Hagan–West monotone
convex (the forward construction has data-dependent clamps), tension splines with
data-dependent tension, anything with a positivity floor. Adding one would require the curve to
export second-order rows, not a change to `Line1d`.

## Rebinding

`Line1d.with_y(y)` keeps the grid and scheme, skips the grid checks, and refits. This is the hot
path of calibration and bump-and-reprice: the grid is validated once, the values change many
times.

Next: [maths/02 · Extrapolation](02_extrapolation.md).
