"""JAX autodiff backend for the pricing engine.

Optional and lazily imported: nothing in the core path imports this package, so the
jax/scipy/numpy pin friction never touches normal pricing.  Import it explicitly (or via the
autodiff risk layer) when you want gradient-based risk:

    from finance.pricing.engines.jax import JaxProgram

It consumes the *same* ``KernelInputs`` the numpy engine does — see ``JaxProgram``.

Importing this package enables JAX's 64-bit mode: curve parameters and discount factors are
float64 everywhere else in the library, and JAX defaults to float32, which would silently
truncate them.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from finance.pricing.engines.jax.curves import CurveGeometry, curve_geometry, make_df, make_logdf  # noqa: E402
from finance.pricing.engines.jax.program import JaxProgram  # noqa: E402

__all__ = ["JaxProgram", "CurveGeometry", "curve_geometry", "make_df", "make_logdf"]
