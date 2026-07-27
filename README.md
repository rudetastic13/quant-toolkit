# quant-toolkit

Dabbling in interest rate analytics

## Documentation

- [Pricing Architecture](docs/pricing_architecture.md) — how a trade flows from a data row
  through an instrument into a compiled `PricingProgram` and is repriced against market data,
  with diagrams, design decisions, and the patterns used.
- [Curve and Rate Architecture](docs/curve_architecture.md) — the `ZeroCurve` → `YieldCurve`
  → `MarketContext` → `RateGenerator` layering, explicit calibration registration, and
  Numba pricing/risk example.
- [Engine Selection](docs/engine_selection.md) — policy for which numeric backend
  (`Numpy` / `Numba` / `Jax`) a pricer binds, and why greeks must come from the same engine.
- [Numba Engine](docs/numba_engine.md) — fused pricing and analytic-adjoint engine for linear
  rates and European options, with FD-of-gradient second order.
- [SOFR Futures](docs/sofr_futures.md) — implemented SR1/SR3 contracts, contract-month dating,
  pricing/risk, and convexity-adjusted calibration.
- [SOFR calibration and aggregate risk sample](research/sofr_curve_calibration.py) — calibrates
  deposits/swaps, registers a SOFR yield curve, reports a 7Y swap, then nets N randomized
  swaps by zero and par-quote buckets with index/funding DV01 and gamma.
- [Excel Add-in](src/quant_toolkit_xl/README.md) — the `quant_toolkit_xl` xlwings add-in:
  worksheet functions for curves, swaps, and compiled pricing programs.

# Setup

Source the ~/bin/use_env.sh and the conda environment for using the right conda yml parameters.

```bash
source bin/use_env.sh pricing
```

## Folder Structure

The project is laid out into sections

`common` are core software engineering tools, patterns and principles that are used in other packages, all other packages
are allowed to import it, but it can't import other packages

`finance` is the core module for interest rate analytics: dates, calendars, terms, instruments,
markets/curves, and the pricing layer (compile-once/reprice-many programs, calibration, risk).
Its test suite lives at `finance/tests`, mirroring the source structure.

`quant_toolkit` is the public API facade — it re-exports the `finance` module for external users.

`quant_toolkit_xl` is the Excel add-in (xlwings): worksheet functions (`qBuildCurve`, `qSwap`,
`qPrice`, ...) over the same pricing layer. Install with the `[excel]` extra.

`research` is for testing out new ideas without needing to land in a specific place.

### Build

Eventually we will have a single `pyproject.toml` that lays out the ruff rules to adhere to globally.

Each of these core folders will then have their own pyproject.toml

This is intentionally a monorepo.

### Docstrings

Functions are explicitly typed and have NumPy style docstrings. This is to ensure that the code is self-documenting and that the types are clear to users and developers alike.
