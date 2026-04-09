# quant-toolkit
Dabbling in interest rate analytics

# Setup

Source the ~/bin/use_env.sh and the conda environment for using the right conda yml parameters.

## Folder Structure

The project is laid out into sections

`common` are core software engineering tools, patterns and principles that are used in other packages, all other packages
are allowed to import it, but it can't import other packages

`finance` is the core module for interest rate analytics.

`quant_toolkit` is the core module for working with excel tools, exposes finance module.

`research` are for testing out new ideas without needing to land in a specific place.

`tests` this needs to be refatored.

### Build

Eventually we will have a single `pyproject.toml` that lays out the ruff rules to adhere to globally.

Each of these core folders will then have their own pyproject.toml

This is intentionally a monorepo.

### Docstrings

Functions are explicitly typed and have NumPy style docstrings. This is to ensure that the code is self-documenting and that the types are clear to users and developers alike.
