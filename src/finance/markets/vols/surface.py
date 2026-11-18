"""Vol surfaces: numbers over (expiry, tenor), with the weights that locate them.

A surface is strike-less geometry.  It holds ``width`` numbers at each node and says, for a
query ``(expiry, tenor)``, which nodes it reads and with what weights.  What those numbers
*mean* -- a quoted vol, or a smile model's parameters -- is decided by the
:class:`~finance.markets.vols.cube.ModelVolCube` that holds the surface, which is also where
the interpolated values and the gradient are formed: both follow from the weights alone, so
a surface never implements them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Self, runtime_checkable

import numpy as np

FloatArray = np.ndarray  # NDArray[float64]
IntArray = np.ndarray  # NDArray[int64]


@runtime_checkable
class VolSurface(Protocol):
    """Contract every surface satisfies.

    ``params`` is the flat parameter vector, node-major: ``params.reshape(-1, width)[j]`` are
    the ``width`` numbers at node ``j``.
    """

    @property
    def params(self) -> FloatArray: ...

    @property
    def width(self) -> int: ...

    def with_params(self, params: FloatArray) -> Self: ...

    def node_weights(self, expiry: FloatArray, tenor: FloatArray) -> tuple[IntArray, FloatArray]:
        """Nodes each query reads and their weights.

        Parameters
        ----------
        expiry, tenor : ndarray of float64, shape (n,)
            Year fractions.

        Returns
        -------
        index : ndarray of int64, shape (n, k)
            Node indices; a node may repeat within a row.
        weight : ndarray of float64, shape (n, k)
            Interpolation weights; each row sums to one.
        """
        ...


@dataclass(frozen=True)
class FlatVolSurface:
    """One node read everywhere: a constant vol, or a constant set of model parameters.

    Parameters
    ----------
    level : float or ndarray of float64, shape (width,)
        A single vol, or one value per model parameter.
    """

    level: float | FloatArray

    @property
    def params(self) -> FloatArray:
        return np.atleast_1d(np.asarray(self.level, dtype=np.float64))

    @property
    def width(self) -> int:
        return self.params.size

    def with_params(self, params: FloatArray) -> Self:
        return replace(self, level=params)

    def node_weights(self, expiry: FloatArray, tenor: FloatArray) -> tuple[IntArray, FloatArray]:
        n = np.broadcast(expiry, tenor).size
        return np.zeros((n, 1), dtype=np.int64), np.ones((n, 1))


def _bracket(x: FloatArray, xq: FloatArray) -> tuple[IntArray, IntArray, FloatArray]:
    """Bracketing nodes of ``xq`` on ``x`` and the weight of the upper one, clamped at the ends."""
    lo = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, max(x.size - 2, 0))
    hi = np.minimum(lo + 1, x.size - 1)
    span = x[hi] - x[lo]
    upper = np.zeros(xq.shape)
    np.divide(xq - x[lo], span, out=upper, where=span > 0.0)
    return lo, hi, np.clip(upper, 0.0, 1.0)


@dataclass(frozen=True)
class GridVolSurface:
    """An expiry x tenor grid, bilinear inside and flat beyond the edges.

    Parameters
    ----------
    expiries : ndarray of float64, shape (n_expiries,)
        Year fractions, strictly increasing.
    tenors : ndarray of float64, shape (n_tenors,)
        Year fractions, strictly increasing.
    values : ndarray of float64, shape (n_expiries, n_tenors) or (n_expiries, n_tenors, width)
        Node values; the two-dimensional form is ``width = 1``.
    """

    expiries: FloatArray
    tenors: FloatArray
    values: FloatArray

    def __post_init__(self) -> None:
        if self.values.shape[:2] != (self.expiries.size, self.tenors.size):
            raise ValueError(
                f"values has shape {self.values.shape}; expected ({self.expiries.size}, {self.tenors.size}[, width])"
            )
        if np.any(np.diff(self.expiries) <= 0.0) or np.any(np.diff(self.tenors) <= 0.0):
            raise ValueError("expiries and tenors must be strictly increasing")

    @property
    def params(self) -> FloatArray:
        return self.values.ravel()

    @property
    def width(self) -> int:
        return self.values.shape[2] if self.values.ndim == 3 else 1

    def with_params(self, params: FloatArray) -> Self:
        return replace(self, values=params.reshape(self.values.shape))

    def node_weights(self, expiry: FloatArray, tenor: FloatArray) -> tuple[IntArray, FloatArray]:
        expiry, tenor = (np.ravel(q) for q in np.broadcast_arrays(expiry, tenor))
        e_lo, e_hi, e_up = _bracket(self.expiries, expiry)
        t_lo, t_hi, t_up = _bracket(self.tenors, tenor)
        n_tenors = self.tenors.size
        index = np.stack(
            [e_lo * n_tenors + t_lo, e_lo * n_tenors + t_hi, e_hi * n_tenors + t_lo, e_hi * n_tenors + t_hi], axis=1
        )
        weight = np.stack(
            [(1.0 - e_up) * (1.0 - t_up), (1.0 - e_up) * t_up, e_up * (1.0 - t_up), e_up * t_up], axis=1
        )
        return index, weight
