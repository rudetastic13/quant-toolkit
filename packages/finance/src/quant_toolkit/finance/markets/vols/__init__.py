"""Volatility market data.

A :class:`VolSurface` is numbers over (expiry, tenor); a :class:`ModelVolCube` pairs one with
its quoting units and an optional :class:`SmileModel` and is what option pricers resolve by
name from a :class:`VolNamespace`.  With no model the cube is plain 2D interpolation of quoted
vols.  The surface parameters are the vol risk factors (``cube.params`` / ``with_params`` /
``vol_gradient``).
"""
from quant_toolkit.finance.markets.vols.cube import ModelVolCube, SmileModel, VolUnits
from quant_toolkit.finance.markets.vols.namespace import VolNamespace
from quant_toolkit.finance.markets.vols.surface import FlatVolSurface, GridVolSurface, VolSurface

__all__ = [
    "FlatVolSurface",
    "GridVolSurface",
    "ModelVolCube",
    "SmileModel",
    "VolNamespace",
    "VolSurface",
    "VolUnits",
]
