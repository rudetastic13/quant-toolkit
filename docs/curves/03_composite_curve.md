# 03 · `CompositeCurve`: weighted sums in log-DF space

`CompositeCurve(components, weights)` is the one way curves combine:

```
ln DF(x) = Σᵢ wᵢ · ln DFᵢ(x)        ⇔        DF = Πᵢ DFᵢ(x)^{wᵢ}
```

Zero rates and instantaneous forwards add with the same weights. Components are any
`DiscountCurve`, so composites nest. Parameters are the components' parameters concatenated;
`param_offsets` and `component_slice` recover the blocks. Passing a mapping labels the
components, and labels are how a spread curve says which part is the base and which the spread.

Continues from [01](01_discount_curve.md) (`sofr`, `xq`).

## Why only weighted sums

A weighted sum in log-DF is the **only** composition that keeps `ln DF` affine in the parameters,
which is the engine contract ([01](01_discount_curve.md)). Multiplying zero rates, taking a
minimum, clamping: each would break the affine map and the engines would have to know the scheme
again. Every production need fits the sum:

| Need | Composite |
|---|---|
| base + spread | `{"base": a, "spread": b}` |
| basis between two indices | `(a, b)`, weights `(1, -1)` |
| turn-of-year on a smooth curve | `{"smooth": a, "turn": t}` |
| scenario overlay | `(curve, shock)` |
| a curve "in another curve's units" | weights other than ±1 |

There is deliberately no separate spread class. A spread *is* arithmetic on a base curve, so it
is a labelled two-component composite, and everything a desk wants from it is one label away.

## Arithmetic

Because a spread is arithmetic, the arithmetic is the API. Every operator builds a
`CompositeCurve`, so nothing outside the weighted-sum family can be written.

```python
from finance.markets.curves import bp          # bp = 1e-4

fedfund = sofr - 5 * bp                        # Fed Funds is SOFR minus 5
fedfund
# CompositeCurve(base=+1*ZeroCurve(nodes=6, ...), spread=+1*ZeroCurve(nodes=2, ...))
fedfund.labels
# ('base', 'spread')
termsofr_3m = sofr + s3m                       # base plus a term structure of spread; left operand is the base
basis = fedfund - sofr                         # a basis has no base: unlabelled, weights (1, -1)
basis.weights, basis.labels, basis.zero_rate(xq)
# ((1.0, -1.0), None, array([-0.0005, -0.0005, -0.0005]))
(0.5 * sofr).weights
# (0.5,)
sofr * fedfund
# TypeError: curves multiply only by scalars: a product of curves is not affine in the parameters.
```

| expression | builds |
|---|---|
| `a + b` | `CompositeCurve({"base": a, "spread": b})` |
| `a + s`, `s + a` | `CompositeCurve({"base": a, "spread": ZeroCurve.constant(s, a.max_x)})` |
| `a - b` | `CompositeCurve((a, b), (1, -1))` |
| `a - s` | base plus `ZeroCurve.constant(-s, a.max_x)` |
| `w * a`, `a * w`, `-a` | `CompositeCurve((a,), (w,))` |
| `a * b`, `a / w` | `TypeError` |

**Operators define curves; they do not move them.** `sofr + 10 * bp` is a new curve with one
more parameter, a 10bp spread. `sofr.with_params(sofr.params + 10 * bp)` is the same five
parameters at new values. Both price identically today; they report different risk and the
first is a different `scheme`:

```python
defined = sofr + 10 * bp
moved = sofr.with_params(sofr.params + 10 * bp)
np.allclose(defined.discount_factor(xq), moved.discount_factor(xq)), defined.n_params, moved.n_params
# (True, 6, 5)
```

A scenario is `with_params` (or, once the market knows how to propagate, its `bump`); a
definition is an operator.

## Leaves and replacement

Two structural methods make the sharing between composites usable. `leaves()` lists the atoms a
curve is built from, with the weight accumulated along the path and the columns they occupy in
the curve's own parameters and Jacobian. `replace(old, new)` rebuilds a composite with one
object substituted wherever it appears, by identity, leaving everything else shared.

```python
term6m = (sofr + ZeroCurve.constant(8 * bp, sofr.max_x)) + ZeroCurve.constant(4 * bp, sofr.max_x)   # 6M over 1M over SOFR
[(type(lf.curve).__name__, lf.weight, lf.columns) for lf in term6m.leaves()]
# [('ZeroCurve', 1.0, slice(0, 5)), ('ZeroCurve', 1.0, slice(5, 6)), ('ZeroCurve', 1.0, slice(6, 7))]
[lf.columns for lf in (term6m - sofr).leaves()]                   # sofr reached twice: once inside, once directly
# [slice(0, 5), slice(5, 6), slice(6, 7), slice(7, 12)]

new6m = term6m.replace(sofr, moved)
new6m["base"]["base"] is moved, new6m["spread"] is term6m["spread"], new6m.zero_rate(xq) - term6m.zero_rate(xq)
# (True, True, array([0.001, 0.001, 0.001]))
```

These are what a market uses to move SOFR through every curve built on it (`market.bump`,
`market.with_curve`), and what the engine's `ParamLayout` uses to give a shared atom one parameter
block instead of one per registered curve.

## A constant spread

The common case: an index that trades at a roughly fixed margin to another. The spread
component is `ZeroCurve.constant(rate, horizon)`, a two-node log-linear atom whose single
parameter is the rate; flat-forward extrapolation continues it unchanged for every `x`, so the
horizon only names the pillar.

```python
fedfund = CompositeCurve({"base": sofr, "spread": ZeroCurve.constant(-0.0005, sofr.max_x)})   # Fed Funds ~5bp under SOFR
fedfund
# CompositeCurve(base=+1*ZeroCurve(nodes=6, max_x=3650, space=LogDF, interpolator=Linear(), extrapolation=FlatForward),
#                spread=+1*ZeroCurve(nodes=2, max_x=3650, space=LogDF, interpolator=Linear(), extrapolation=FlatForward))
fedfund.params
# [ 0.041   0.04    0.039   0.041   0.043  -0.0005]
fedfund.component_slice("base"), fedfund.component_slice("spread")
# (slice(0, 5, None), slice(5, 6, None))
np.allclose(fedfund.discount_factor(xq), sofr.discount_factor(xq) * fedfund["spread"].discount_factor(xq))
# True
```

### Seeing the spread

The spread component is a curve, so every query is available on it, and the difference against
the base recovers the same number by construction:

```python
fedfund["spread"].zero_rate(xq)                         # the continuously compounded spread to x
# [-0.0005 -0.0005 -0.0005]
fedfund.zero_rate(xq) - fedfund["base"].zero_rate(xq)   # the same, as a difference
# [-0.0005 -0.0005 -0.0005]
fedfund["spread"].instantaneous_forward(xq)             # the forward spread
# [-0.0005 -0.0005 -0.0005]
```

## A term structure of spread

Any `DiscountCurve` can be the spread. A funding spread that widens with tenor:

```python
sx = np.array([0.0, 365.0, 1825.0, 3650.0])
sz = np.array([0.0, 0.0008, 0.0012, 0.0015])            # 8bp at 1Y, 12bp at 5Y, 15bp at 10Y
sdfs = np.exp(-sz * sx / 365.0)
sdfs[0] = 1.0
funding = CompositeCurve({"base": sofr, "spread": ZeroCurve(sx, sdfs)})

funding.x                                               # union of base and spread pillars
# [   0.   90.  365.  730. 1825. 3650.]
funding.n_params                                        # 5 base + 3 spread
# 8
funding["spread"].zero_rate(xq)
# [0.0008   0.000935 0.001581]
funding.instantaneous_forward(xq) - sofr.instantaneous_forward(xq)     # forward spread, piecewise constant
# [0.0008 0.0013 0.0018]
```

## Risk splits by block

The affine map is the components' maps side by side. The base block is the atom's own Jacobian,
unchanged; the spread block is the spread atom's.

```python
offset_s, jac_s = funding.log_df_affine(xq)
jac_s.shape
# (3, 8)
np.allclose(jac_s[:, funding.component_slice("base")], sofr.log_df_affine(xq)[1])
# True
jac_s[:, funding.component_slice("spread")]
# [[ -0.082192   0.         0.      ]
#  [ -0.907534  -0.462329   0.      ]
#  [  0.         3.69863  -17.39726 ]]
```

So a book priced on `funding` reports a ladder over SOFR pillars (index or funding risk, depending
on the role) and a separate ladder over the spread pillars (basis risk). A spread composite
registered as its own index (`USD.FEDFUND.1D` over the SOFR atom) does this through the engine with
no extra code; [05](05_risk_jacobian_hessian.md) shows the output.

## A basis curve

```python
basis = CompositeCurve((fedfund, sofr), weights=(1.0, -1.0))
basis.zero_rate(xq)
# [-0.0005 -0.0005 -0.0005]
basis.n_params, basis.param_offsets
# (11, array([ 0,  6, 11]))
```

`fedfund` is itself built on `sofr`, so SOFR appears twice in this composite's parameter vector:
once inside `fedfund`'s base block (columns 0–4) and once as the second component (columns 6–10).
With weights `(1, -1)` the two SOFR blocks of the affine Jacobian cancel exactly:

```python
_, jb = basis.log_df_affine(xq)
np.allclose(jb[:, :5] + jb[:, 6:], 0.0)
# True
```

That duplication is correct arithmetic but it is also the one thing to be aware of when
composites share an atom: today each *registered* curve's parameters form their own block in the
engine, so a SOFR atom reachable through two registered curves is reported under both. Risk is
not wrong, it is not yet consolidated; the identity-based parameter layout in
[curves_and_market_plan.md](../curves_and_market_plan.md) (PR E) is the piece that consolidates it.

## A turn

A turn is an atom whose forward is zero everywhere except inside one window. Its log-linear
pillars bracket the window; the parameters are the zero rates that put the desired forward
inside it.

```python
turn_x = np.array([0.0, 200.0, 210.0, 230.0])                                   # a 10-day window ending on day 210
turn_z = np.array([0.0, 0.0, 0.0030 * 10.0 / 210.0, 0.0030 * 10.0 / 230.0])     # +30bp of forward in the window
turn_dfs = np.exp(-turn_z * turn_x / 365.0)
turn_dfs[0] = 1.0
turn = ZeroCurve(turn_x, turn_dfs)
turn.instantaneous_forward(np.array([100.0, 205.0, 300.0, 3000.0]))
# [-0.     0.003 -0.    -0.   ]

with_turn = CompositeCurve({"smooth": sofr, "turn": turn})
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
nested = CompositeCurve((funding, turn))                          # funding is itself base + term spread
flat_equivalent = CompositeCurve((sofr, funding["spread"], turn))
np.allclose(nested.log_discount_factor(xq), flat_equivalent.log_discount_factor(xq))
# True
nested.params
# [ 0.041     0.04      0.039     0.041     0.043     0.0008    0.0012    0.0015   -0.        0.000143  0.00013 ]
```

## Scenarios

`with_params` moves every block from one flat vector; `with_component(key, curve)` swaps one
component, by label or index, for a curve of any geometry and keeps the weights and labels. The
untouched components are shared by reference, not copied.

```python
wider = funding.with_component("spread", funding["spread"].with_params(funding["spread"].params + 0.0005))
wider["spread"].zero_rate(xq) - funding["spread"].zero_rate(xq)
# [0.0005 0.0005 0.0005]
wider["base"] is sofr
# True

scenario = with_turn.with_component("turn", turn.with_params(turn.params * 2.0))
scenario.instantaneous_forward(np.array([205.0])) - sofr.instantaneous_forward(np.array([205.0]))
# [0.006]
with_turn.scheme == scenario.scheme                 # parameters moved, geometry did not
# True
with_turn.scheme == CompositeCurve((sofr, ZeroCurve(np.array([0.0, 400.0]), np.array([1.0, 0.99])))).scheme
# False                                              # a different turn geometry is a different scheme
```

`scheme` includes each component's pillar positions and scheme, not just the union `x`: the
union alone would not say which component owns a pillar, and the affine rows depend on that.
Labels are *not* part of `scheme`, since they do not change the arithmetic.

## Extrapolation and validation

Each component extrapolates by its own rule, so a two-node flat-forward spread continues forever
while a base with `NotAllowed` still rejects. Queries before the origin are rejected by every
component. Construction rejects an empty component list, a weight count that does not match,
duplicate labels, and zero or non-finite weights (a zero weight would create parameters with no
effect):

```python
CompositeCurve((sofr, turn), weights=(1.0, 0.0))
# ValueError: weights must be finite and non-zero; got (1.0, 0.0).
```

## When is a spread a curve, and when is it a convention?

Two different things look like "index A plus a margin":

- **A risk factor.** Fed Funds over SOFR, a funding or CSA spread, a z-spread on a bond. The
  spread moves, the desk wants to see and bump it, hedges depend on it. That is a spread
  composite: a curve with parameters.
- **A contractual redefinition.** An index that is *defined* as another index plus a fixed
  number (a fallback-rate index, or a legacy index quoted as a benchmark plus a fixed spread
  adjustment). The spread has no risk because it cannot change. That belongs on the leg or the
  index family's conventions, in the rate generator, not in a curve, or a parameter would be reported
  with zero risk forever.

The library does not decide for you; both are expressible. Prefer the convention when the
spread is fixed by contract, the curve when it is a market quote.

## As a `DiscountCurve`

| Member | `CompositeCurve` |
|---|---|
| `x` | `np.union1d` over components |
| `dfs` | `discount_factor(x)` |
| `params` / `n_params` | concatenation; `param_offsets`, `component_slice(key)` |
| `labels`, `curve[key]`, `len(curve)` | component access by label or index |
| `scheme` | `("composite", weights, ((component.x, component.scheme), ...))`; labels excluded |
| `with_params` | split by offsets, each component's `with_params`, rebuilt without re-validation |
| `with_component(key, curve)` | one component replaced, weights and labels kept |
| `log_df_affine` | `Σ wᵢ offsetᵢ`, `hstack(wᵢ jacobianᵢ)`; exact by construction, no probing |
| `is_log_linear` | `False` |

Next: [04 · `YieldCurve`](04_yield_curve.md), where a discount curve meets the calendar.
