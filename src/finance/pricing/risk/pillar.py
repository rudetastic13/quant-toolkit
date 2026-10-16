"""PillarRisk — ``∂PV/∂z`` for a book, one block per owning key: the contract between engines and templates.

Every pricing door in the library (a ``PricingProgram`` of legs, a ``FuturesProgram``, a
``SwaptionProgram``) can produce the same thing: a matrix of first-order sensitivities of each
position to every curve parameter the market holds, laid out by the engine's
:class:`~finance.pricing.engines.layout.ParamLayout` (one block per owning key, each atom once).
:func:`pillar_risk` is the adapter that produces it, and :class:`PillarRisk` is the only thing a
:class:`~finance.pricing.risk.template.RiskTemplate` consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finance.pricing.types import Backend

FloatArray = np.ndarray
DateArray = np.ndarray


@dataclass(frozen=True)
class PillarRisk:
    """``∂value/∂z`` for a set of positions, per curve block.

    Parameters
    ----------
    curves : tuple of str
        Owning keys, block order.
    param_offsets : ndarray of intp, shape (C + 1,)
        Block ``j`` is ``values[:, param_offsets[j]:param_offsets[j + 1]]``.
    pillar_dates : dict[str, ndarray of datetime64[D]]
        Per curve, the non-origin pillar dates of its atoms, column order.
    values : ndarray of float64, shape (N, P)
        Derivatives per unit (per 1.0 of zero rate), raw sign.
    pv : ndarray of float64, shape (N,)
        The positions' values at the same market (PV, or margined P&L for futures).
    labels : tuple of str or None
        Position ids.
    """

    curves: tuple[str, ...]
    param_offsets: np.ndarray
    pillar_dates: dict[str, DateArray]
    values: FloatArray
    pv: FloatArray
    labels: tuple[str, ...] | None = None

    @property
    def n_positions(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_params(self) -> int:
        return int(self.values.shape[1])

    def slice(self, curve: str) -> slice:
        """Columns of ``curve``'s block."""
        try:
            j = self.curves.index(curve)
        except ValueError:
            raise KeyError(f"no block for curve '{curve}'; blocks are {list(self.curves)}") from None
        return slice(int(self.param_offsets[j]), int(self.param_offsets[j + 1]))

    def block(self, curve: str) -> FloatArray:
        """``values`` restricted to ``curve``'s block, shape (N, P_c) (a view)."""
        return self.values[:, self.slice(curve)]

    @property
    def dv01(self) -> FloatArray:
        """``values`` per +1bp."""
        return self.values * 1e-4

    def aligned(self, curves: tuple[str, ...], pillar_dates: dict[str, DateArray]) -> PillarRisk:
        """This risk re-laid out over ``curves`` in that order.

        Blocks this risk has are copied (their pillars must equal ``pillar_dates[c]``); blocks it
        lacks are zero-filled; blocks it has but ``curves`` does not name are dropped (the
        caller decides whether that is "uncovered" risk worth reporting).

        Raises
        ------
        ValueError
            If a shared curve's pillars differ.
        """
        curves = tuple(curves)
        sizes = [int(np.asarray(pillar_dates[c]).size) for c in curves]
        offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.intp)
        values = np.zeros((self.n_positions, int(offsets[-1])), dtype=np.float64)
        for j, c in enumerate(curves):
            if c in self.curves:
                if not np.array_equal(self.pillar_dates[c], pillar_dates[c]):
                    raise ValueError(f"pillar dates of '{c}' differ between the risk and the requested layout")
                values[:, offsets[j] : offsets[j + 1]] = self.block(c)
        return PillarRisk(
            curves=curves,
            param_offsets=offsets,
            pillar_dates={c: np.asarray(pillar_dates[c]) for c in curves},
            values=values,
            pv=self.pv,
            labels=self.labels,
        )

    @classmethod
    def from_layout(cls, layout, values: FloatArray, pv: FloatArray, labels=None) -> PillarRisk:
        """Wrap a full-width gradient laid out by ``layout``."""
        offsets = np.concatenate([[0], np.cumsum([layout.block_slices[k].stop - layout.block_slices[k].start for k in layout.blocks])]).astype(np.intp)
        return cls(
            curves=tuple(layout.blocks),
            param_offsets=offsets,
            pillar_dates={k: layout.pillar_dates(k) for k in layout.blocks},
            values=np.ascontiguousarray(values, dtype=np.float64),
            pv=np.ascontiguousarray(pv, dtype=np.float64),
            labels=None if labels is None else tuple(labels),
        )


def pillar_risk(program, market, *, backend: Backend = Backend.Numba, labels=None) -> PillarRisk:
    """``∂value/∂z`` of every position in ``program`` at ``market``, from the analytic adjoint.

    Parameters
    ----------
    program : PricingProgram, FuturesProgram or SwaptionProgram
        Legs give total (projection + discounting) PV risk; a futures strip gives the margined
        P&L gradient (no discounting; ``pv`` is the P&L against the entry price, 0 without one);
        swaptions give curve delta at sticky strike.
    market : MarketContext
    backend : Backend
        Only ``Numba`` has the adjoint.
    labels : sequence of str, optional

    Returns
    -------
    PillarRisk
    """
    from finance.pricing.pricers.base import PricingProgram
    from finance.pricing.pricers.futures import FuturesProgram
    from finance.pricing.pricers.swaption import SwaptionProgram

    if Backend(backend) != Backend.Numba:
        raise NotImplementedError(f"pillar_risk needs the Numba adjoint; backend {Backend(backend).name} has none")
    if isinstance(program, PricingProgram):
        numba = program.prepare(market, backend=Backend.Numba)
        result = numba.value_and_grad(market)
        return PillarRisk.from_layout(numba.layout, result.instrument_total_gradient, result.primal.instrument_pv, labels)
    if isinstance(program, FuturesProgram):
        numba = program.program.prepare(market, backend=Backend.Numba)
        risk = program.risk(market)
        pnl = program.reprice(market).pnl
        return PillarRisk.from_layout(numba.layout, risk.pnl_gradient, np.where(np.isnan(pnl), 0.0, pnl), labels)
    if isinstance(program, SwaptionProgram):
        numba = program.program.prepare(market, backend=Backend.Numba)
        risk = program.risk(market)
        return PillarRisk.from_layout(numba.layout, risk.curve_gradient, program.reprice(market).pv, labels)
    raise TypeError(f"pillar_risk has no adapter for {type(program).__name__}")
