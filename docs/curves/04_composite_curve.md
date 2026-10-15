# 04 · `CompositeCurve`: weighted sums in log-DF space

`CompositeCurve(components, weights)` is the general form of which `SpreadCurve` is the
two-component special case:

```
ln DF(x) = Σᵢ wᵢ · ln DFᵢ(x)        ⇔        DF = Πᵢ DFᵢ(x)^{wᵢ}
```

Zero rates and instantaneous forwards add with the same weights. Components are any
`DiscountCurve`, so composites nest. Parameters are the components' parameters concatenated;
`param_offsets` and `component_slice(i)` recover the blocks.

Continues from [01](01_discount_curve.md) and [03](03_spread_curve.md) (`sofr`, `fedfund`,
`funding`, `xq`).

## Why only weighted sums

A weighted sum in log-DF is the **only** composition that keeps `ln DF` affine in the parameters,
which is the engine contract ([01](01_discount_curve.md)). Multiplying zero rates, taking a
minimum, clamping: each would break the affine map and the engines would have to know the scheme
again. Every production need fits the sum:

| Need | Composite |
|---|---|
| base + spread | `(base, spread)`, weights `(1, 1)` → `SpreadCurve` |
| basis between two indices | `(a, b)`, weights `(1, -1)` |
| turn-of-year on a smooth curve | `(smooth, turn)` |
| scenario overlay | `(curve, shock)` |
| a curve "in another curve's units" | weights other than ±1 |

## A basis curve

```python
basis = CompositeCurve((fedfund, sofr), weights=(1.0, -1.0))
basis.zero_rate(xq)
# [-0.0005 -0.0005 -0.0005]
basis.n_params, basis.param_offsets
# (11, array([ 0,  6, 11]))
```

`fedfund` is itself a spread over `sofr`, so SOFR appears twice in this composite's parameter
vector: once inside `fedfund`'s base block (columns 0–4) and once as the second component
(columns 6–10). With weights `(1, -1)` the two SOFR blocks of the affine Jacobian cancel exactly:

```python
_, jb = basis.log_df_affine(xq)
np.allclose(jb[:, :5] + jb[:, 6:], 0.0)
# True
```

That duplication is correct arithmetic but it is also the one thing to be aware of when
composites share an atom: today each *registered* curve's parameters form their own block in the
engine, so a SOFR atom reachable through two registered curves is reported under both. Risk is
not wrong, it is not yet consolidated; the shared-atom parameter layout (`ParamLayout` in
[risk_and_engine_design.md](../risk_and_engine_design.md) §4.1) is the piece that consolidates it.

## A turn

A turn is an atom whose forward is zero everywhere except inside one window. Its log-linear
pillars bracket the window; the parameters are the zero rates that put the desired forward
inside it.

```python
turn_x = np.array([0.0, 200.0, 210.0, 230.0])                      # a 10-day window ending on day 210
turn_z = np.array([0.0, 0.0, 0.0030 * 10.0 / 210.0, 0.0030 * 10.0 / 230.0])   # +30bp of forward in the window
turn_dfs = np.exp(-turn_z * turn_x / 365.0)
turn_dfs[0] = 1.0
turn = ZeroCurve(turn_x, turn_dfs)
turn.instantaneous_forward(np.array([100.0, 205.0, 300.0, 3000.0]))
# [-0.     0.003 -0.    -0.   ]

with_turn = CompositeCurve((sofr, turn))
probe = np.array([100.0, 205.0, 300.0])
with_turn.instantaneous_forward(probe) - sofr.instantaneous_forward(probe)
# [0.    0.003 0.   ]
with_turn.x
# [   0.   90.  200.  210.  230.  365.  730. 1825. 3650.]
```

The turn's parameters are their own block, so "turn risk" is a column of the ladder, separate
from the smooth curve's pillars, with no change to the smooth curve's calibration.

## Nesting

A composite of composites equals the flat composite with product weights; the affine maps agree
column for column.

```python
nested = CompositeCurve((funding, turn))                    # funding is itself SpreadCurve(sofr, term spread)
flat_equivalent = CompositeCurve((sofr, funding.spread, turn))
np.allclose(nested.log_discount_factor(xq), flat_equivalent.log_discount_factor(xq))
# True
nested.params
# [ 0.041     0.04      0.039     0.041     0.043     0.0008    0.0012    0.0015   -0.        0.000143  0.00013 ]
```

## Scenarios

`with_params` moves every block from one flat vector; `with_component(i, curve)` swaps one
component for a curve of any geometry and keeps the weights.

```python
scenario = with_turn.with_component(1, turn.with_params(turn.params * 2.0))
scenario.instantaneous_forward(np.array([205.0])) - sofr.instantaneous_forward(np.array([205.0]))
# [0.006]
with_turn.scheme == scenario.scheme                 # parameters moved, geometry did not
# True
with_turn.scheme == CompositeCurve((sofr, ZeroCurve(np.array([0.0, 400.0]), np.array([1.0, 0.99])))).scheme
# False                                              # a different turn geometry is a different scheme
```

`scheme` includes each component's pillar positions and scheme, not just the union `x`: the
union alone would not say which component owns a pillar, and the affine rows depend on that.

## Extrapolation and validation

Each component extrapolates by its own rule, so a two-node flat-forward spread continues forever
while a base with `NotAllowed` still rejects. Queries before the origin are rejected by every
component. Construction rejects an empty component list, a weight count that does not match, and
zero or non-finite weights (a zero weight would create parameters with no effect):

```python
CompositeCurve((sofr, turn), weights=(1.0, 0.0))
# ValueError: weights must be finite and non-zero; got (1.0, 0.0).
```

## As a `DiscountCurve`

| Member | `CompositeCurve` |
|---|---|
| `x` | `np.union1d` over components |
| `dfs` | `discount_factor(x)` |
| `params` / `n_params` | concatenation; `param_offsets`, `component_slice(i)` |
| `scheme` | `("composite", weights, ((component.x, component.scheme), ...))` |
| `with_params` | split by offsets, each component's `with_params`, rebuilt without re-validation |
| `log_df_affine` | `Σ wᵢ offsetᵢ`, `hstack(wᵢ jacobianᵢ)`; exact by construction, no probing |
| `is_log_linear` | `False` |

Next: [05 · `YieldCurve`](05_yield_curve.md), where a discount curve meets the calendar.
