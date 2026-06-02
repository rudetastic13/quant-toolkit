"""Discounted-cashflow PV kernel — pure, portfolio-wide, no per-instrument loop.

Given per-flow cashflows for a whole portfolio concatenated into one flat array, plus
``row_offsets`` marking where each instrument's flows begin, PV-per-instrument is a
single ``np.add.reduceat``.  Pricing 10k instruments is one reduction, not 10k loops.
"""
from __future__ import annotations

import numpy as np

from common.registry import register_with
from finance.pricing.engines import engine_registry
from finance.pricing.types import KERNEL_DCF, Backend

FloatArray = np.ndarray


@register_with(engine_registry, (KERNEL_DCF, Backend.Numpy))
def dcf(
    cash: FloatArray,
    df: FloatArray,
    sign: FloatArray,
    row_offsets: np.ndarray,
) -> FloatArray:
    """Present value per instrument.

    Parameters
    ----------
    cash : (F,) float
        Per-flow cashflow amount (``notional * rate * period_frac`` plus any principal),
        all instruments' flows concatenated.
    df : (F,) float
        Discount factor at each flow's payment date.
    sign : (F,) float
        +1 / -1 per flow (receive / pay).
    row_offsets : (I,) int
        Start index of each instrument's flows in the flat arrays (strictly increasing,
        ``row_offsets[0] == 0``).

    Returns
    -------
    (I,) float
        PV for each instrument.
    """
    cash = np.asarray(cash, dtype=np.float64)
    df = np.asarray(df, dtype=np.float64)
    sign = np.asarray(sign, dtype=np.float64)
    row_offsets = np.asarray(row_offsets, dtype=np.intp)

    if not (cash.shape == df.shape == sign.shape):
        raise ValueError("cash, df, sign must share shape")
    if row_offsets.size and row_offsets[0] != 0:
        raise ValueError("row_offsets[0] must be 0")

    flow_pv = cash * df * sign
    return np.add.reduceat(flow_pv, row_offsets)


__all__ = ["dcf"]
