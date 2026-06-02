"""Option pricing kernels — DESIGNED, not implemented.

Reserved seams so swaptions / options on treasury futures slot in alongside the linear
kernels with no architectural change.  When implemented, register under
``(KERNEL_BACHELIER, Backend.Numpy)`` / ``(KERNEL_BLACK, Backend.Numpy)`` exactly like the
rate kernels; the option pricer supplies forward, strike, expiry (year-fraction) and a vol
read from a ``VolSurface``, plus the annuity from the curve.

Kept pure and array-shaped (forward/strike/expiry/vol broadcast), so a numba/rust version is
a drop-in under the same key.
"""
from __future__ import annotations

import numpy as np

FloatArray = np.ndarray


def bachelier(forward: FloatArray, strike: FloatArray, expiry: FloatArray, vol: FloatArray,
              is_call: bool = True) -> FloatArray:
    """Normal-vol (Bachelier) undiscounted option value per unit annuity. NOT IMPLEMENTED."""
    raise NotImplementedError("Bachelier kernel is a designed seam; not implemented yet.")


def black(forward: FloatArray, strike: FloatArray, expiry: FloatArray, vol: FloatArray,
          is_call: bool = True) -> FloatArray:
    """Lognormal-vol (Black) undiscounted option value per unit annuity. NOT IMPLEMENTED."""
    raise NotImplementedError("Black kernel is a designed seam; not implemented yet.")


__all__ = ["bachelier", "black"]
