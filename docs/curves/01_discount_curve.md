# 01 · `DiscountCurve`: the protocol

`finance.markets.curves.DiscountCurve` is a `runtime_checkable` `Protocol`. It is the contract
`YieldCurve`, `MarketContext` and the pricing engines consume. Nothing downstream of a curve
knows whether it is a single interpolated line or a sum of several, so a composite prices and
risks through exactly the same path as an atom.

Two implementations exist:

| Class | Doc | What it is |
|---|---|---|
| `ZeroCurve` | [02](02_zero_curve.md) | the atom: one `Line1d` in log-DF or zero-rate space |
| `CompositeCurve` | [03](03_composite_curve.md) | `ln DF = Σ wᵢ ln DFᵢ` over any discount curves: spreads, bases, turns |

## Setup used by every document

Day offsets from the origin, Act/365 continuously compounded zero rates at the pillars, and the
discount factors they imply. Everything that follows reuses `x`, `z`, `dfs`, `sofr` and `xq`.

```python
import numpy as np
from finance.markets.curves import CompositeCurve, CurveSpace, DiscountCurve, ZeroCurve

x = np.array([0.0, 90.0, 365.0, 730.0, 1825.0, 3650.0])          # 3M 1Y 2Y 5Y 10Y, in days
z = np.array([0.0, 0.0410, 0.0400, 0.0390, 0.0410, 0.0430])      # Act/365 cc zero rates at the pillars
dfs = np.exp(-z * x / 365.0)
dfs[0] = 1.0
sofr = ZeroCurve(x, dfs)

isinstance(sofr, DiscountCurve)
# True
sofr
# ZeroCurve(nodes=6, max_x=3650, space=LogDF, interpolator=Linear(), extrapolation=FlatForward)
```

## The contract, member by member

### Geometry

```python
sofr.x
# [   0.   90.  365.  730. 1825. 3650.]
sofr.dfs
# [1.       0.989941 0.960789 0.924964 0.814647 0.650509]
sofr.max_x
# 3650.0
```

`x` is origin-inclusive and strictly increasing; `x[0] == 0` is the origin and `dfs[0] == 1`.
A composite reports the union of its components' pillars. `max_x` is where extrapolation
begins (see [maths/02](maths/02_extrapolation.md)).

### Parameters

```python
sofr.n_params, sofr.params
# (5, array([0.041, 0.04 , 0.039, 0.041, 0.043]))
```

`params` is the flat float64 vector the engines differentiate with respect to. For a
`ZeroCurve` it is the zero rate at each non-origin pillar; for a composite it is the components'
parameters concatenated. `with_params` rebinds it and returns a new curve of the same type with
the same geometry and scheme:

```python
bumped = sofr.with_params(sofr.params + 1e-4)
xq = np.array([30.0, 500.0, 5000.0])     # inside the first interval, between 1Y and 2Y, beyond 10Y
bumped.zero_rate(xq) - sofr.zero_rate(xq)
# [0.0001 0.0001 0.0001]
bumped.scheme == sofr.scheme and np.array_equal(bumped.x, sofr.x)
# True
```

### Scheme

```python
sofr.scheme
# (<CurveSpace.LogDF: 1>, Linear())
```

`scheme` is a hashable identity of everything *except* the parameter values. Two curves with
equal `x` and equal `scheme` have identical affine maps (below), so a prepared pricing program can
reuse its compiled query table across them. This is the compatibility key of the reprice and
risk hot path: a scenario that only moves parameters hits the cache, a scenario that changes
geometry or interpolation rebuilds. `is_log_linear` is the one special case the JAX test oracle
models.

### Queries

All take and return float64 arrays of day offsets. Queries before the origin are rejected.

```python
sofr.log_discount_factor(xq)
# [-0.00337  -0.054055 -0.596438]
sofr.discount_factor(xq)
# [0.996636 0.94738  0.55077 ]
sofr.zero_rate(xq)                     # -ln DF / (x/365); at x = 0 the instantaneous forward
# [0.041   0.03946 0.04354]
sofr.instantaneous_forward(xq)         # -d ln DF / dt
# [0.041 0.038 0.045]
```

The maths behind each is in [maths/03](maths/03_discount_factors.md) to
[maths/06](maths/06_forward_rates.md).

### The engine hand-off: `log_df_affine`

This is the member that makes the whole stack differentiable without the engines ever seeing an
interpolation scheme.

```python
offset, jac = sofr.log_df_affine(xq)
offset
# [0. 0. 0.]
jac
# [[ -0.082192   0.         0.         0.         0.      ]
#  [  0.        -0.630137  -0.739726   0.         0.      ]
#  [  0.         0.         0.         3.69863  -17.39726 ]]
np.allclose(offset + jac @ sofr.params, sofr.log_discount_factor(xq), atol=1e-14)
# True
```

The contract is that `ln DF(x)` is **exactly affine** in `params` for every query:

```
ln DF(x) = offset(x) + jacobian(x) · params
```

Reading the rows: the 30-day query sits in the first interval, so it depends on the 3M pillar
only, with weight `-30/365`. The 500-day query is log-linearly between 1Y and 2Y and depends on
both. The 5000-day query is beyond the last pillar; flat-forward extrapolation continues the
10Y segment's slope, so it depends on the 5Y and 10Y pillars with large opposite weights (the
terminal forward is a difference quotient of the last two nodes).

Why it holds for everything in the library: every scheme in `common.math.interpolation` is
linear in its node values (see [maths/01](maths/01_interpolation.md)), node values are linear in
the zero rates in either space (`ln DF_j = -z_j x_j / 365`, or `r_j = z_j`), both extrapolation
rules are linear, and a weighted sum of affine maps is affine. `ZeroCurve.log_df_affine` probes
its map one pillar at a time and checks the reconstruction, so a scheme that ever broke the
assumption would be rejected loudly rather than mis-differentiated.

Two consequences that shape the rest of the library:

- **Engines are scheme-agnostic.** At prepare time each curve hands over `(offset, jacobian)`
  on the program's static queries; the Numba kernel evaluates `exp(offset + jacobian @ params)`
  and scatters adjoints back through `jacobian`. Cubic, mixed, zero-rate-space: no kernel
  change.
- **`ln DF` has zero second-order sensitivity to the parameters.** All curvature of a PV lives
  downstream, in `exp`, in compounding, in par-rate quotients. [05](05_risk_jacobian_hessian.md)
  builds exact Hessians from this fact.

## Why a protocol and not a base class

Composition over inheritance. `CompositeCurve` *holds* its components and does not inherit from
`ZeroCurve`. The protocol states what a curve must do, `isinstance` checks it at runtime
(`CompositeCurve.__init__` rejects a component that does not satisfy it), and each class remains
free to implement storage and rebinding its own way.

Next: [02 · `ZeroCurve`](02_zero_curve.md).
