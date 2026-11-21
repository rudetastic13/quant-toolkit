"""``ModelVolCube``: a surface paired with what its numbers mean.

The cube is what a pricer resolves and asks for a vol.  With no model the surface holds quoted
vols and the cube is plain interpolation over (expiry, tenor); with a :class:`SmileModel` the
surface holds that model's parameters, which are interpolated first and then turned into a vol
at the option's strike.  Either way the surface parameters are the risk factors, and
``vol_gradient`` is the exact derivative of ``vol`` in them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Protocol, Self, runtime_checkable

import numpy as np

from quant_toolkit.finance.markets.vols.surface import FloatArray, IntArray, VolSurface


class VolUnits(IntEnum):
    """Quoting convention of a cube's vols: Normal (Bachelier) or Lognormal (Black)."""

    Normal = 1
    Lognormal = 2


@runtime_checkable
class SmileModel(Protocol):
    """Interpolated model parameters -> a vol at a strike, with its parameter derivative."""

    def vol(self, params: FloatArray, expiry: FloatArray, strike: FloatArray, forward: FloatArray) -> FloatArray:
        """Vol per option, shape (n,), from ``params`` of shape (n, width)."""
        ...

    def d_params(self, params: FloatArray, expiry: FloatArray, strike: FloatArray, forward: FloatArray) -> FloatArray:
        """``d vol / d params``, shape (n, width)."""
        ...


@dataclass(frozen=True)
class ModelVolCube:
    """A :class:`VolSurface` plus an optional :class:`SmileModel`.

    Parameters
    ----------
    surface : VolSurface
        Quoted vols (``width == 1``) when ``model`` is ``None``, else the model's parameters.
    units : VolUnits
        Quoting convention of the vols the cube returns; a pricer refuses a cube quoted for
        the wrong option model.
    model : SmileModel or None, optional

    Notes
    -----
    ``expiry`` and ``tenor`` are year fractions; ``strike`` and ``forward`` are in rate units.
    All four broadcast and are read flattened, so results have a leading axis of length ``n``.
    """

    surface: VolSurface
    units: VolUnits
    model: SmileModel | None = None

    def __post_init__(self) -> None:
        if self.model is None and self.surface.width != 1:
            raise ValueError(f"a cube with no model reads one vol per node; the surface carries {self.surface.width}")

    @property
    def params(self) -> FloatArray:
        """The surface parameters, shape (n_params,): the vol risk factors."""
        return self.surface.params

    def with_params(self, params: FloatArray) -> Self:
        return replace(self, surface=self.surface.with_params(params))

    def _interpolate(self, expiry: FloatArray, tenor: FloatArray) -> tuple[IntArray, FloatArray, FloatArray]:
        index, weight = self.surface.node_weights(expiry, tenor)
        nodes = self.surface.params.reshape(-1, self.surface.width)
        return index, weight, np.einsum("nk,nkp->np", weight, nodes[index])

    def vol(self, expiry: FloatArray, tenor: FloatArray, strike: FloatArray, forward: FloatArray) -> FloatArray:
        """Vol per option.

        Returns
        -------
        ndarray of float64, shape (n,)
        """
        expiry, tenor, strike, forward = (np.ravel(q) for q in np.broadcast_arrays(expiry, tenor, strike, forward))
        _, _, values = self._interpolate(expiry, tenor)
        if self.model is None:
            return values[:, 0]
        return self.model.vol(values, expiry, strike, forward)

    def vol_gradient(
        self, expiry: FloatArray, tenor: FloatArray, strike: FloatArray, forward: FloatArray
    ) -> FloatArray:
        """``d vol / d params`` per option, holding the forward fixed.

        Returns
        -------
        ndarray of float64, shape (n, n_params)
        """
        expiry, tenor, strike, forward = (np.ravel(q) for q in np.broadcast_arrays(expiry, tenor, strike, forward))
        index, weight, values = self._interpolate(expiry, tenor)
        width = self.surface.width
        d_values = np.ones_like(values) if self.model is None else self.model.d_params(values, expiry, strike, forward)
        gradient = np.zeros((expiry.size, self.params.size))
        rows = np.arange(expiry.size)[:, None, None]
        columns = index[:, :, None] * width + np.arange(width)
        np.add.at(gradient, (rows, columns), weight[:, :, None] * d_values[:, None, :])
        return gradient
