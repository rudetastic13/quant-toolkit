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
- [Swaptions](docs/swaptions.md) — building, pricing and risking European swaptions, with worked
  numbers for curve DV01 and node vega.
- [SOFR calibration and aggregate risk sample](research/sofr_curve_calibration.py) — calibrates
  deposits/swaps, registers a SOFR yield curve, reports a 7Y swap, then nets N randomized
  swaps by zero and par-quote buckets with index/funding DV01 and gamma.
- [Excel Add-in](apps/excel/README.md) — the `quant_toolkit.xl` xlOil add-in:
  `QuantToolkit.*` worksheet functions for dates, swaps, curve calibration and pricing.

# Setup

Conda only supplies the Python interpreter (`conda/pricing/*.yml`); the libraries come from the `pyproject.toml` files.
Sourcing the script creates/activates the env and editable-installs every portion plus the dev extras.
Every script in `bin/` comes in two flavours with the same behaviour: `bin/bash/*.sh` (Linux, macOS, Git Bash)
and `bin/powershell/*.ps1` (Windows). conda must be hooked into the shell first (`conda init bash` / `conda init powershell`).

```bash
source bin/bash/use_env.sh pricing           # bash / Git Bash
```

```powershell
. bin/powershell/use_env.ps1 pricing          # PowerShell (dot-source so the activation sticks)
```

Installing the library elsewhere:

```bash
pip install quant-toolkit                      # common + finance
pip install "quant-toolkit[accel,oracle]"      # optional: Numba engine, JAX validation oracle
pip install "quant-toolkit[test]"              # what the test suite needs; also: dev, notebook, cpp
```

## Folder Structure

Everything imports under one namespace, `quant_toolkit` (PEP 420: no `__init__.py` at `quant_toolkit/` in any portion).
Each portion is its own distribution with its own `pyproject.toml`; versions move in lockstep (`VERSION`,
`bin/bump_version.py`).

```
packages/common/   quant-toolkit-common   quant_toolkit.common   core tools and patterns; imports no other portion
packages/finance/  quant-toolkit-finance  quant_toolkit.finance  dates, calendars, curves, instruments, pricing, risk
apps/excel/        quant-toolkit-xl       quant_toolkit.xl       Excel add-in (xlOil); standalone: `pip install -e apps/excel`
pyproject.toml     quant-toolkit          meta-package (installs common + finance) and all tool config
research/          scratch for trying out ideas; not a distribution
```

Tests live inside each portion (`.../quant_toolkit/<portion>/tests`, mirroring the source structure). This is
intentionally a monorepo.

### Docstrings

Functions are explicitly typed and have NumPy style docstrings. This is to ensure that the code is self-documenting and that the types are clear to users and developers alike.
