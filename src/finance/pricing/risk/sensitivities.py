"""Bump-and-reprice sensitivities: the universal, engine-free risk path.

Every bump is ``market.bump(key, ...)``: a frozen copy of the market in which the parameters
``key`` owns are shifted and every curve built on them follows.  The compiled program is never
rebuilt; a scenario is one ``reprice``.  Pillars are those of the block ``key`` reports under
(:class:`~finance.pricing.engines.layout.ParamLayout`): an alias such as ``USD.STDCSA`` bumps
the SOFR atom it holds, and ``USD.FEDFUND.1D`` built as ``sofr - 5 * bp`` bumps its spread
alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finance.markets.context import MarketContext
from finance.pricing.engines.layout import ParamLayout
from finance.pricing.pricers.base import PricingProgram

FloatArray = np.ndarray


@dataclass
class KeyRateLadder:
    """Per-pillar key-rate durations for a set of instruments."""

    pillar_dates: np.ndarray  # (P,) datetime64[D]
    pillar_years: FloatArray  # (P,)
    krd: FloatArray  # (n_instruments, P) — PV change per +1bp at each pillar

    @property
    def total(self) -> FloatArray:
        """Sum across pillars ~ parallel DV01 (key-rate additivity)."""
        return self.krd.sum(axis=1)


@dataclass
class Sensitivities:
    """Bump-and-reprice sensitivities for a compiled program against a market.

    ``bp`` is the bump size (default 1bp); results are central differences expressed as the
    PV change per +1bp move.
    """

    program: PricingProgram
    market: MarketContext
    bp: float = 1e-4
    _layouts: dict[str, ParamLayout] = field(default_factory=dict, init=False, repr=False, compare=False)

    def _block(self, key: str) -> tuple[ParamLayout, str]:
        layout = self._layouts.get(key)
        if layout is None:
            layout = self._layouts[key] = ParamLayout.build(self.market, [key])
        return layout, layout.block_of(key)

    def _pv(self, market: MarketContext) -> FloatArray:
        return self.program.reprice(market).instrument_pv

    def dv01(self, curve_name: str) -> FloatArray:
        """Parallel DV01 per instrument (PV change for a +1bp move of every parameter the key's block holds)."""
        _, block = self._block(curve_name)
        up = self._pv(self.market.bump(block, self.bp))
        dn = self._pv(self.market.bump(block, -self.bp))
        return (up - dn) / 2.0

    def key_rate_durations(self, curve_name: str) -> KeyRateLadder:
        """Key-rate durations per instrument across the block's pillars (one column per parameter)."""
        layout, block = self._block(curve_name)
        n_params = layout.block_slice(block).stop - layout.block_slice(block).start
        n_inst = self.program.inputs.n_instruments
        krd = np.zeros((n_inst, n_params), dtype=np.float64)
        for j in range(n_params):
            shift = np.zeros(n_params)
            shift[j] = self.bp
            up = self._pv(self.market.bump(block, shift))
            dn = self._pv(self.market.bump(block, -shift))
            krd[:, j] = (up - dn) / 2.0
        return KeyRateLadder(pillar_dates=layout.pillar_dates(block), pillar_years=layout.pillar_years(block), krd=krd)


__all__ = ["Sensitivities", "KeyRateLadder"]
