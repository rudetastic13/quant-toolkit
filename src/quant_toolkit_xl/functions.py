"""The ``q*`` Excel UDFs — the public surface traders see in a worksheet.

Deliberately thin: each function coerces nothing itself, delegates to ``api`` (the real
logic), and converts any exception into a readable ``#QERR: ...`` string so a bad input
shows an explanation in the cell instead of Excel's opaque ``#VALUE!``.

Names are camelCase on purpose — xlwings exposes the Python function name verbatim as the
worksheet function, and ``qBuildCurve`` reads better in a formula than ``q_build_curve``.
Treat these signatures as a stable contract: saved workbooks bind to them.
"""
from __future__ import annotations

from quant_toolkit_xl import api
from quant_toolkit_xl._xw import xw

_ERR = "#QERR: {}"


@xw.func
def qBuildCurve(dates, dfs, interpolation="LogLinearDF"):  # noqa: N802
    """Build a discount curve from pillar dates + discount factors. Returns a curve handle.

    interpolation: one of LogLinearDF, LogCubicDF, RateLinear, RateQuadratic, RateCubic.
    """
    try:
        return api.build_curve(dates, dfs, interpolation)
    except Exception as e:  # noqa: BLE001 - surface to the cell
        return _ERR.format(e)


@xw.func
def qDiscountFactor(curve, dates):  # noqa: N802
    """Discount factors off a curve handle for one or more dates."""
    try:
        return api.discount_factor(curve, dates)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)


@xw.func
def qZeroRate(curve, dates):  # noqa: N802
    """Continuously-compounded zero rates off a curve handle for one or more dates."""
    try:
        return api.zero_rate(curve, dates)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)


@xw.func
def qSwap(notional, rate_index, fixed_rate, tenor, as_of):  # noqa: N802
    """Build a vanilla swap. Returns a swap handle.

    Sign rides on notional: positive = receive fixed, negative = pay fixed.
    """
    try:
        return api.make_swap(notional, rate_index, fixed_rate, tenor, as_of)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)


@xw.func
def qCompileProgram(swaps):  # noqa: N802
    """Compile a range of swap handles into a pricing program. Returns a program handle."""
    try:
        return api.compile_program(swaps)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)


@xw.func
def qMarket(as_of, currency, rate_index, curve):  # noqa: N802
    """Bind a curve handle into a market under (currency, index). Returns a market handle."""
    try:
        return api.make_market(as_of, currency, rate_index, curve)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)


@xw.func
def qPrice(program, market):  # noqa: N802
    """Reprice a program handle against a market handle. Returns per-instrument PVs."""
    try:
        return api.price(program, market)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)
