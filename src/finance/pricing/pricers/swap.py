"""SwapPricer — the swap-shaped name for :class:`LegPricer`.

A swap is two signed legs; nothing about pricing it is swap-specific, so the work lives in
``pricers/legs.py`` and is shared with deposits, FRAs and futures underlyings.  This class
exists so call sites and docs can keep saying ``SwapPricer``.
"""
from __future__ import annotations

from finance.pricing.pricers.legs import LegPricer


class SwapPricer(LegPricer):
    """Compiles Swaps (or any leg-bearing instrument) and prices them against a MarketContext."""


__all__ = ["SwapPricer"]
