# 06 · Risk: first and second order through the affine map

A present value is a function of the curve parameters, `PV(z)`. Risk is its Taylor expansion:

```
PV(z + Δz) ≈ PV(z) + gᵀ Δz + ½ Δzᵀ H Δz

g = ∂PV/∂z        the gradient (first order): one number per parameter, shape (P,)
H = ∂²PV/∂z∂zᵀ    the Hessian  (second order): shape (P, P), symmetric
```

For a book of `N` instruments the gradients stack into the **Jacobian** `G` of shape `(N, P)`,
one row per instrument. For a set of `K` quotes (par rates, futures prices) the same object is
the **template Jacobian** `J_T` of shape `(K, P)`. The whole risk layer of the library is linear
algebra on these two matrices and on `H`; this document shows where they come from.

Continues from [01](01_discount_curve.md) (`sofr`, `x`) and [04](04_yield_curve.md) (`as_of`,
`sofr_yc`, `fedfund_yc`).

## 1. The affine map is the whole derivative of the curve

From [01](01_discount_curve.md): for any set of query offsets,

```
ln DF = offset + J z          J = ∂ ln DF / ∂z,   shape (n_queries, P),   constant in z
```

Everything the pricer needs from the curve follows by the chain rule, in closed form:

```
DF          = exp(offset + J z)
∂DF/∂z      = DF · J                       (row q: DF_q · J[q, :])
∂²DF/∂z∂zᵀ  = DF · J[q,:]ᵀ J[q,:]          (rank one per query, because ∂²ln DF/∂z² = 0)
```

That last line is the key second-order fact: **the curve contributes no curvature of its own**.
`Line1d` is linear in its node values, node values are linear in `z`, so the Hessian of every
discount factor is the outer product of its own Jacobian row, scaled by the DF. No interpolation
scheme in the library needs a second-order extension for this to hold.

## 2. A fixed leg by hand

Four percent semi-annual on 100mm for five years, bullet redemption. Query the affine map once on
the payment offsets; everything else is numpy.

```python
pay_x = np.arange(182.5, 1825.1, 182.5)
cash = np.full(pay_x.size, 100e6 * 0.04 * 0.5)
cash[-1] += 100e6
off, J = sofr.log_df_affine(pay_x)                 # ln DF = off + J z
D = np.exp(off + J @ sofr.params)
pv = cash @ D
pv
# 99395465.50211665

grad = (cash * D) @ J                              # ∂PV/∂z_j = Σ_i c_i DF_i J_ij
hess = (J * (cash * D)[:, None]).T @ J             # ∂²PV/∂z_j∂z_k = Σ_i c_i DF_i J_ij J_ik
grad
# [-3.207398e+05 -3.523584e+06 -1.439224e+07 -4.368749e+08  0.000000e+00]
hess
# [[5.248470e+04 1.078852e+05 0.000000e+00 0.000000e+00 0.000000e+00]
#  [1.078852e+05 2.614696e+06 9.427068e+05 0.000000e+00 0.000000e+00]
#  [0.000000e+00 9.427068e+05 2.014510e+07 1.688490e+07 0.000000e+00]
#  [0.000000e+00 0.000000e+00 1.688490e+07 2.142162e+09 0.000000e+00]
#  [0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00]]
```

The 10Y column is zero because no payment lies beyond 5Y and log-linear interpolation is local;
the Hessian is banded for the same reason. A cubic scheme would fill both in (every node moves
the whole curve), still exactly.

Check against finite differences of the same function:

```python
def pv_at(params):
    return cash @ np.exp(off + J @ params)

eps = 1e-6
E = np.eye(5)
fd_grad = np.array([(pv_at(sofr.params + eps * e) - pv_at(sofr.params - eps * e)) / (2 * eps) for e in E])
fd_hess = np.array([[(pv_at(sofr.params + eps * (ei + ej)) - pv_at(sofr.params + eps * (ei - ej))
                      - pv_at(sofr.params - eps * (ei - ej)) + pv_at(sofr.params - eps * (ei + ej))) / (4 * eps**2)
                     for ej in E] for ei in E])
np.max(np.abs(grad - fd_grad)) / np.max(np.abs(grad)), np.max(np.abs(hess - fd_hess)) / np.max(np.abs(hess))
# (1.6e-11, 2.3e-06)      the analytic forms are exact; the differences are the FD error
```

## 3. From derivatives to fixed-income analytics

Every trader-facing number is a contraction of `g` or `H` with a chosen move `Δz`. The library
fixes the units once: gradients are per unit of zero rate, and `.dv01`-style numbers multiply by
`bp = 1e-4`.

```python
bp = 1e-4
ones = np.ones(5)
grad * bp                                          # key-rate ladder: PV change per +1bp at each pillar
# [   -32.07398    -352.358413  -1439.223595 -43687.492649      0.      ]
grad @ ones * bp                                   # zero-parallel DV01
# -45511.148637720915
0.5 * ones @ hess @ ones * bp**2                   # parallel gamma: ½ · 1ᵀH1 · bp²
# 11.004228223760405

taylor = grad @ ones * 10 * bp + 0.5 * ones @ hess @ ones * (10 * bp) ** 2
taylor, pv_at(sofr.params + 10 * bp) - pv
# (-454011.06, -454012.86)                         second order explains a +10bp move to a few dollars in 100mm
```

| Analytic | Expression | Shape of the move |
|---|---|---|
| Key-rate DV01 (pillar `j`) | `g_j · bp` | `Δz = bp · e_j` |
| Zero-parallel DV01 | `gᵀ1 · bp` | `Δz = bp · 1` |
| Key-rate gamma | `½ H_jj · bp²` | second order of the same |
| Parallel gamma / convexity | `½ 1ᵀH1 · bp²` | second order of the same |
| Cross gamma | `H_jk · bp²` | two pillars together |
| Scenario P&L to second order | `gᵀΔz + ½ ΔzᵀHΔz` | any `Δz` |

Modified duration and convexity of a bond are the same contractions divided by price, with
`Δz = 1` parallel. The engine's `NumbaRisk.gamma` returns `½ H bp²` directly.

## 4. Quotes: the template Jacobian

A par swap rate is a quotient of two PVs, both linear in discount factors. Its gradient is the
quotient rule applied to the gradients above, and it is one row of the template Jacobian `J_T`
that [risk_and_engine_design.md](../risk_and_engine_design.md) §2 uses to re-express pillar risk
as risk to tradeable quotes.

```python
accrual = np.full(pay_x.size, 0.5)
annuity = accrual @ D                              # Σ τ_i DF_i
float_pv = 1.0 - D[-1]                             # Σ (DF_{i-1} - DF_i) telescopes
par = float_pv / annuity
par
# 0.04134859956668942

grad_annuity = (accrual * D) @ J
grad_float = -D[-1] * J[-1]
grad_par = (grad_float * annuity - float_pv * grad_annuity) / annuity**2
grad_par
# [ 0.00074   0.008125  0.033189  0.976806 -0.      ]
grad_par @ ones
# 1.0188599218024366                               a parallel +1 zero move lifts the 5Y par rate by ≈1.02
```

That last number is why the two "parallel DV01s" differ by a couple of percent: a +1bp parallel
*zero* move is not a +1bp parallel *par* move. The `RiskTemplate` makes the basis explicit and
reports both ([risk_and_engine_design.md](../risk_and_engine_design.md) §2.2).

Second order on quotes: `∂²par/∂z²` is the quotient rule once more on closed-form pieces, and
gamma in quote space is the base Hessian sandwiched by the transfer matrix plus a term from the
curvature of the quote-to-pillar map. The library keeps gamma on the base curve, where it is
exact, and leaves quote-space gamma to the template layer.

## 5. The engine does the same thing over real schedules

Same mathematics, real conventions (Act/360 floating accrual, compounding, payment lags, fixings),
one fused Numba kernel for primal and adjoint. The adjoint is the hand-written reverse of
section 2 generalised to compounded legs, and it consumes exactly the `(offset, J)` rows above.

```python
from finance.instruments.resolution import Swap
from finance.pricing.pricers import SwapPricer
from finance.pricing.risk import NumbaRisk
from finance.pricing.risk.sensitivities import Sensitivities
from finance.pricing.types import Backend

sofr_only = MarketContext(as_of, CurveNamespace())
sofr_only.curves.bind(sofr_yc)
template = Swap.fixed_float_swap(notional=100e6, rate_index="SOFR", fixed_rate=0.0, tenor="5Y", as_of=as_of)
par_engine = RateGenerator(sofr_only).par_swap_rate(template)
par_engine
# 0.041209832892398726                             (differs from §4: real dates, Act/360 floating leg, compounding)

swap = Swap.fixed_float_swap(notional=100e6, rate_index="SOFR", fixed_rate=par_engine, tenor="5Y", as_of=as_of)
program = SwapPricer().compile([swap])
program.reprice(sofr_only, backend=Backend.Numba).instrument_pv
# [0.]

risk = NumbaRisk(program, sofr_only)
ladder = risk.zero_ladder("USD.SOFR.ON")              # (n_instruments, P) · bp, adjoint
ladder
# [[    54.782212   -399.156099  -1503.399885 -44164.759033   -139.458284]]
ladder.sum()
# -46151.99108965049
Sensitivities(program, sofr_only).key_rate_durations("USD.SOFR.ON").krd     # bump-and-reprice oracle
# [[    54.782212   -399.1561    -1503.399891 -44164.760817   -139.458284]]
risk.gamma("USD.SOFR.ON")                             # ½ H bp², by central difference of the exact gradient
# [[-0.000015 -0.        0.        0.        0.      ]
#  [-0.        0.019848  0.000219  0.        0.      ]
#  [ 0.        0.000219  0.11738   0.081307 -0.      ]
#  [ 0.        0.        0.081307 10.820518  0.034807]
#  [ 0.        0.       -0.        0.034807  0.000115]]
```

The small positive 3M risk and the non-zero 10Y risk that the hand-built leg did not have are the
floating leg: its compounded coupons reach into the first interval, and its last payment date
falls past the 5Y pillar, into the segment that the 5Y and 10Y nodes share.

## 6. Composite curves split the ladder by block

A Fed Funds swap projected off the spread composite registered as `USD.FEDFUND.ON`
([04](04_yield_curve.md)) and discounted off `USD.STDCSA`, the SOFR curve under its discount key.
The index ladder on `USD.FEDFUND.ON` has six columns: the five SOFR pillars of its base block,
then the spread parameter. The spread
column is the parallel index risk, because a flat spread *is* a parallel move of the projection
curve.

```python
ff_market = MarketContext(as_of, CurveNamespace())
ff_market.curves.bind(sofr_yc)
ff_market.curves.bind(sofr_yc, "USD.STDCSA")
ff_market.curves.bind(fedfund_yc)
ff_swap = Swap.fixed_float_swap(notional=100e6, rate_index="FEDFUND", fixed_rate=0.04, tenor="5Y", as_of=as_of)
ff_risk = NumbaRisk(SwapPricer().compile([ff_swap]), ff_market)
ff_risk.index_ladder("USD.FEDFUND.ON")                # [ base block (SOFR pillars) | spread ]
# [[    54.754828   -389.556882  -1472.34591  -44265.337577   -139.586974 -46212.072515]]
ff_risk.funding_ladder("USD.STDCSA")               # discounting risk; the SOFR block under its discount key
# [[  0.        -2.695521  -4.811094 181.990268   0.292894]]
```

Basis risk (the spread column), index risk (the base block) and funding risk (the SOFR ladder)
come out of one adjoint sweep with no product-specific code, because the composite's affine map
already carried the block structure.

## Where this goes next

- **`PillarRisk`** (design §2.7) is the typed carrier of `G` per curve block.
- **`RiskTemplate`** (design §2) computes `J_T` for a set of tradeable instruments on the base
  curve and solves `R = G B (J_T B)⁻¹` so the book's risk is reported against those quotes, with
  the unexplained remainder shown rather than hidden.
- **Shared atoms.** Today a SOFR atom reachable through two registered curves is two parameter
  blocks. The `ParamLayout` of design §4.1 keys parameters by atom so it is reported once.
