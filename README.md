# quant-toolkit

Dabbling in interest rate analytics

## Documentation

- [Pricing Architecture](docs/pricing_architecture.md) — how a trade flows from a data row
  through an instrument into a compiled `PricingProgram` and is repriced against market data,
  with diagrams, design decisions, and the patterns used.
- [Engine Selection](docs/engine_selection.md) — policy for which numeric backend
  (`Numpy` / `Numba` / `Jax`) a pricer binds, and why greeks must come from the same engine.
- [Numba Engine](docs/numba_engine.md) — implementation plan for the Numba engine: analytic
  first-order risk for linear rates and European options, FD-of-gradient for second order.
- [SOFR Futures](docs/sofr_futures.md) — design spec for adding SOFR futures (SR1/SR3):
  instrument layer, contract-month dating, convexity-adjusted calibration.
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
