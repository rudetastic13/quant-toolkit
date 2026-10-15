# Curves

The curve stack, one object per document, in reading order. Every document carries runnable
snippets; they share one setup (defined in [01](01_discount_curve.md)) and build on each
other, so read them in order the first time. Outputs shown as comments were produced by running
the snippets against the repository at the time of writing.

| # | Document | What it covers |
|---|---|---|
| 01 | [`DiscountCurve`](01_discount_curve.md) | The protocol every curve satisfies: geometry, parameters, queries, and the affine engine contract |
| 02 | [`ZeroCurve`](02_zero_curve.md) | The atom: a `Line1d` in log-DF or zero-rate space with any interpolation scheme |
| 03 | [`SpreadCurve`](03_spread_curve.md) | Base plus spread, flat or term-structured; risk splits into base and spread blocks |
| 04 | [`CompositeCurve`](04_composite_curve.md) | Weighted sums in log-DF space: basis curves, turns, overlays, nesting |
| 05 | [`YieldCurve`](05_yield_curve.md) | The landing place: dates, index conventions, fixings, the market, rate generation |
| 06 | [Risk: Jacobian and Hessian](06_risk_jacobian_hessian.md) | First- and second-order risk, the affine map, and how it becomes fixed-income analytics |

The `maths/` folder is the quantitative background the objects rest on:

| # | Document |
|---|---|
| 01 | [Interpolation](maths/01_interpolation.md) |
| 02 | [Extrapolation](maths/02_extrapolation.md) |
| 03 | [Discount factors](maths/03_discount_factors.md) |
| 04 | [Log discount factors](maths/04_log_discount_factors.md) |
| 05 | [Continuously compounded rates](maths/05_continuous_rates.md) |
| 06 | [Forward rates](maths/06_forward_rates.md) |

Related: [curve_architecture.md](../curve_architecture.md) (responsibilities and boundaries),
[curves_line1d.md](../curves_line1d.md) (`Line1d` usage with figures),
[risk_and_engine_design.md](../risk_and_engine_design.md) (the `RiskTemplate` that consumes
the Jacobians described in 06).

## Running the snippets

```bash
source bin/use_env.sh pricing          # or: PYTHONPATH=src python ...
python -c "import numpy as np; np.set_printoptions(precision=6, suppress=True)"
```

Every snippet assumes `numpy` is imported as `np` with that print configuration.
