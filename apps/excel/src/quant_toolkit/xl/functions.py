"""The ``q*`` Excel worksheet functions (xlOil) — the public surface traders see in a worksheet.

Loaded in-process by the xlOil add-in when listed in ``[xlOil_Python] LoadModules`` of
``%APPDATA%\\xlOil\\xlOil.ini`` (``quant_toolkit_xl_installer install`` adds it); every
``@xlo.func`` here registers as a worksheet function on load.

Deliberately thin: each function coerces nothing itself, delegates to ``api`` (the real
logic), and converts any exception into a readable ``#QERR: ...`` string so a bad input
shows an explanation in the cell instead of Excel's opaque ``#VALUE!``.

Names are camelCase on purpose: xlOil registers the Python function name verbatim as the
worksheet function. Treat these signatures as a stable contract — saved workbooks bind to them.
"""
from __future__ import annotations

import xloil as xlo

from quant_toolkit.xl import api

_ERR = "#QERR: {}"


@xlo.func
def qVersion() -> str:  # noqa: N802
    """quant-toolkit-xl version; a smoke test that the add-in loaded."""
    from importlib.metadata import version

    return f"quant-toolkit-xl {version('quant-toolkit-xl')}"


@xlo.func
def qBuildCurve(dates, dfs, interpolation: str = "LogLinearDF"):  # noqa: N802
    """Build a discount curve from pillar dates + discount factors. Returns a curve handle.

    interpolation: one of LogLinearDF, LogCubicDF, RateLinear, RateQuadratic, RateCubic.
    """
    try:
        return api.build_curve(dates, dfs, interpolation)
    except Exception as e:  # noqa: BLE001 - surface to the cell
        return _ERR.format(e)


@xlo.func
def qDiscountFactor(curve: str, dates):  # noqa: N802
    """Discount factors off a curve handle for one or more dates."""
    try:
        return api.discount_factor(curve, dates)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)


@xlo.func
def qZeroRate(curve: str, dates):  # noqa: N802
    """Continuously-compounded zero rates off a curve handle for one or more dates."""
    try:
        return api.zero_rate(curve, dates)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)


@xlo.func
def qSwap(notional: float, rate_index: str, fixed_rate: float, tenor: str, as_of):  # noqa: N802
    """Build a vanilla swap. Returns a swap handle.

    Sign rides on notional: positive = receive fixed, negative = pay fixed.
    """
    try:
        return api.make_swap(notional, rate_index, fixed_rate, tenor, as_of)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)


@xlo.func
def qCompileProgram(swaps):  # noqa: N802
    """Compile a range of swap handles into a pricing program. Returns a program handle."""
    try:
        return api.compile_program(swaps)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)


@xlo.func
def qMarket(as_of, ccy: str, rate_index: str, curve: str, funding_id: str = "STDCSA"):  # noqa: N802
    """Bind a curve handle into a market under its projection key and the funding key. Returns a market handle."""
    try:
        return api.make_market(as_of, ccy, rate_index, curve, funding_id)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)


@xlo.func
def qPrice(program: str, market: str):  # noqa: N802
    """Reprice a program handle against a market handle. Returns per-instrument PVs as a column."""
    try:
        return api.price(program, market).reshape(-1, 1)
    except Exception as e:  # noqa: BLE001
        return _ERR.format(e)
