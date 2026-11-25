"""SwapPricer — the swap-shaped name for :class:`LegPricer`.

A swap is two signed legs; nothing about pricing it is swap-specific, so the work lives in
``pricers/legs.py`` and is shared with deposits, FRAs and futures underlyings.  The alias
exists so call sites and docs can keep saying ``SwapPricer``.
"""
from __future__ import annotations

from quant_toolkit.finance.pricing.pricers.legs import LegPricer

SwapPricer = LegPricer
