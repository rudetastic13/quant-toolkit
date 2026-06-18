"""JAX autodiff backend for the pricing engine.

Optional and lazily imported: nothing in the core path imports this package, so the
jax/scipy/numpy pin friction never touches normal pricing.  Import it explicitly (or via the
autodiff risk layer) when you want gradient-based risk:

    from finance.pricing.engines.jax import JaxProgram

It consumes the *same* ``KernelInputs`` the numpy engine does — see ``JaxProgram``.
"""
from __future__ import annotations

from finance.pricing.engines.jax.curves import CurveGeometry, curve_geometry, make_df, make_logdf
from finance.pricing.engines.jax.program import JaxProgram

__all__ = ["JaxProgram", "CurveGeometry", "curve_geometry", "make_df", "make_logdf"]
