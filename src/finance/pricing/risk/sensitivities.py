"""Sensitivities — DV01 and key-rate durations by bump-and-reprice over a PricingProgram.

This is the clean inversion of the legacy KRD that rebuilt the whole trade per shocked
curve.  Here the program is compiled ONCE; a bump is a new curve bound into a fresh market
and a single ``reprice`` over the same columnar arrays.  Sensitivities live in their own
layer — they are not methods on the instrument or the pricer.

First cut: bumps are on the curve's own zero-rate pillars (a "zero-rate key rate").  A
par-instrument / calibration-instrument key rate would sit on top of a calibration layer
that does not exist yet; the structure here (bump -> reprice) is identical when it does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finance.markets.curves import ZeroCurve
from finance.markets.context import MarketContext
from finance.pricing.pricers.base import PricingProgram

FloatArray = np.ndarray


def _zero_rates(curve: ZeroCurve) -> tuple[FloatArray, FloatArray]:
    """Return (year-fractions, continuously-compounded zero rates) at the curve's pillars."""
    t = curve.x / 365.0
    dfs = curve.dfs
    z = np.zeros_like(dfs)
    m = t > 0
    z[m] = -np.log(dfs[m]) / t[m]
    return t, z


def bumped_curve(curve: ZeroCurve, bp: float, pillar: int | None = None) -> ZeroCurve:
    """Curve with zero rates shifted by ``bp`` — all pillars (parallel) or just ``pillar``.

    The origin (t=0, DF=1) is never bumped. Interpolation method is preserved.
    """
    t, z = _zero_rates(curve)
    m = t > 0
    z2 = z.copy()
    if pillar is None:
        z2[m] += bp
    elif t[pillar] > 0:
        z2[pillar] += bp
    new_dfs = curve.dfs.copy()
    new_dfs[m] = np.exp(-z2[m] * t[m])
    return curve.with_dfs(new_dfs)


@dataclass
class KeyRateLadder:
    """Per-pillar key-rate durations for a set of instruments."""

    pillar_dates: np.ndarray   # (P,) datetime64[D]
    pillar_years: FloatArray   # (P,)
    krd: FloatArray            # (n_instruments, P) — PV change per +1bp at each pillar

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

    def _reprice_with(self, curve_name: str, curve: ZeroCurve) -> FloatArray:
        yield_curve = self.market.yield_curve(curve_name).with_zero_curve(curve)
        return self.program.reprice(self.market.with_curve(yield_curve)).instrument_pv

    def dv01(self, curve_name: str) -> FloatArray:
        """Parallel DV01 per instrument (PV change for a +1bp parallel zero-rate move)."""
        base = self.market.zero_curve(curve_name)
        up = self._reprice_with(curve_name, bumped_curve(base, self.bp))
        dn = self._reprice_with(curve_name, bumped_curve(base, -self.bp))
        return (up - dn) / 2.0

    def key_rate_durations(self, curve_name: str) -> KeyRateLadder:
        """Key-rate durations per instrument across the curve's zero-rate pillars."""
        base = self.market.zero_curve(curve_name)
        t, _ = _zero_rates(base)
        pillars = np.nonzero(t > 0)[0]
        n_inst = self.program.inputs.n_instruments
        krd = np.zeros((n_inst, pillars.size), dtype=np.float64)
        for j, i in enumerate(pillars):
            up = self._reprice_with(curve_name, bumped_curve(base, self.bp, pillar=int(i)))
            dn = self._reprice_with(curve_name, bumped_curve(base, -self.bp, pillar=int(i)))
            krd[:, j] = (up - dn) / 2.0
        pillar_dates = self.market.yield_curve(curve_name).node_dates[pillars]
        return KeyRateLadder(pillar_dates=pillar_dates, pillar_years=t[pillars], krd=krd)


__all__ = ["Sensitivities", "KeyRateLadder", "bumped_curve"]
