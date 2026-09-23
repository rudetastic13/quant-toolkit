"""Volatility surfaces — DESIGNED seam, minimal stub implementation.

Linear products (swaps/bonds/futures) do not need this; it exists so that
option pricers (swaptions, options on treasury futures) slot in without
reworking the market layer.  Only ``FlatVolSurface`` is implemented today;
calibrated surfaces are future work.

Mirrors ``CurveNamespace`` (versioned bind/resolve/snapshot) so the option
pricers resolve a surface by name exactly as linear pricers resolve a curve.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

import numpy as np

FloatArray = np.ndarray  # NDArray[float64]


class VolUnits(IntEnum):
    """Quoting convention of a surface: Normal (Bachelier) or Lognormal (Black)."""

    Normal = 1
    Lognormal = 2


@runtime_checkable
class VolSurface(Protocol):
    """Contract every vol surface must satisfy.

    ``expiry`` and ``tenor`` are year-fractions; ``strike`` and ``forward`` are in rate
    units.  ``units`` declares the quoting convention so a pricer can refuse a surface
    quoted for the wrong model.  ``forward`` lets a smile surface interpolate in
    moneyness; strike-only surfaces may ignore it.  All arguments broadcast as numpy
    arrays.
    """

    units: VolUnits

    def vol(
        self, expiry: FloatArray, tenor: FloatArray, strike: FloatArray, forward: FloatArray
    ) -> FloatArray: ...


@dataclass
class FlatVolSurface:
    """A single constant vol for all (expiry, tenor, strike). Stub only."""

    level: float
    units: VolUnits

    def vol(self, expiry, tenor, strike, forward) -> FloatArray:
        return np.full(np.broadcast(expiry, tenor, strike).shape, self.level, dtype=np.float64)


class VolNamespace:
    """Name -> VolSurface store with a version counter, mirroring CurveNamespace.

    Not a singleton; one per scenario is fine. See
    ``finance.markets.curves.CurveNamespace`` for the pattern.
    """

    def __init__(self) -> None:
        self._surfaces: dict[str, VolSurface] = {}
        self._versions: dict[str, int] = {}

    def bind(self, name: str, surface: VolSurface) -> None:
        if name in self._surfaces:
            raise KeyError(f"'{name}' is already bound. Use rebind() to overwrite.")
        self._surfaces[name] = surface
        self._versions[name] = 0

    def rebind(self, name: str, surface: VolSurface) -> None:
        self._surfaces[name] = surface
        self._versions[name] = self._versions.get(name, -1) + 1

    def resolve(self, name: str) -> VolSurface:
        try:
            return self._surfaces[name]
        except KeyError:
            raise KeyError(f"No vol surface bound under '{name}'") from None

    def version(self, name: str) -> int:
        try:
            return self._versions[name]
        except KeyError:
            raise KeyError(f"No vol surface bound under '{name}'") from None

    def __contains__(self, name: str) -> bool:
        return name in self._surfaces


__all__ = ["VolSurface", "VolUnits", "FlatVolSurface", "VolNamespace"]
