"""Pure logic for the Excel adapter — no xlwings dependency, fully unit-testable.

Each function takes already-loosely-typed Excel inputs, marshals them into core-library
types, calls into ``quant_toolkit``, and (for object-producing calls) stashes the result in
the handle ``cache``. The camelCase ``q*`` UDFs in ``functions`` are thin wrappers over
these.
"""
from __future__ import annotations

import numpy as np

from finance.instruments.resolution import Swap, curve_name
from finance.markets.context import MarketContext
from finance.markets.curves import CurveInterpolator, CurveNamespace, YieldCurve, ZeroCurve, dates_to_x
from finance.pricing.pricers import SwapPricer

from quant_toolkit_xl import cache, marshalling


def _interp(interpolation: object) -> CurveInterpolator:
    if isinstance(interpolation, CurveInterpolator):
        return interpolation
    return CurveInterpolator[str(interpolation)]


def build_curve(dates: object, dfs: object, interpolation: object = "LogLinearDF") -> str:
    """Build a ``ZeroCurve`` from pillar dates + discount factors; return a Curve handle.

    The handle holds ``(origin, ZeroCurve)``: the first pillar date is the origin and the
    curve itself is on day offsets.
    """
    node_dates = marshalling.to_datetime64_array(dates)
    node_dfs = marshalling.to_float_array(dfs)
    interp = _interp(interpolation)
    space, interpolator = interp.resolve()
    origin = node_dates[0]
    curve = ZeroCurve(dates_to_x(origin, node_dates), node_dfs, space=space, interpolator=interpolator)
    return cache.store("Curve", (origin, curve), node_dates, node_dfs, interp.name)


def _load_curve(curve_handle: object) -> tuple[np.datetime64, ZeroCurve]:
    return cache.load(curve_handle, "Curve")  # type: ignore[return-value]


def discount_factor(curve_handle: object, dates: object) -> np.ndarray:
    """Discount factors off a cached curve for an array of dates."""
    origin, curve = _load_curve(curve_handle)
    return curve.discount_factor(dates_to_x(origin, marshalling.to_datetime64_array(dates)))


def zero_rate(curve_handle: object, dates: object) -> np.ndarray:
    """Continuously-compounded zero rates off a cached curve for an array of dates."""
    origin, curve = _load_curve(curve_handle)
    return curve.zero_rate(dates_to_x(origin, marshalling.to_datetime64_array(dates)))


def make_swap(
    notional: object,
    rate_index: object,
    fixed_rate: object,
    tenor: object,
    as_of: object,
) -> str:
    """Resolve a vanilla swap from the trader-facing builder; return a Swap handle.

    Sign rides on the notional: positive = receive fixed, negative = pay fixed.
    """
    as_of_date = marshalling.to_date(as_of)
    swap = Swap.fixed_float_swap(
        notional=float(notional),
        rate_index=str(rate_index),
        fixed_rate=float(fixed_rate),
        tenor=str(tenor),
        as_of=as_of_date,
    )
    return cache.store(
        "Swap", swap, float(notional), str(rate_index), float(fixed_rate),
        str(tenor), str(as_of_date),
    )


def compile_program(swap_handles: object) -> str:
    """Compile one or more cached swaps into a ``PricingProgram``; return a Program handle."""
    hs = marshalling.handles(swap_handles)
    if not hs:
        raise ValueError("no swap handles supplied to compile")
    swaps = [cache.load(h, "Swap") for h in hs]
    program = SwapPricer().compile(swaps)
    return cache.store("Program", program, tuple(hs))


def make_market(as_of: object, currency: object, rate_index: object, curve_handle: object) -> str:
    """Bind a cached curve into a ``MarketContext`` under (ccy, index); return a Market handle."""
    as_of_date = marshalling.to_date(as_of)
    origin, curve = _load_curve(curve_handle)
    currency_name = str(currency)
    index_name = str(rate_index)
    name = curve_name(currency_name, index_name)
    yield_curve = YieldCurve.from_registry(
        origin,
        curve,
        currency=currency_name,
        index_name=index_name,
    )
    namespace = CurveNamespace()
    namespace.bind(yield_curve)
    market = MarketContext(as_of_date=as_of_date, curves=namespace)
    return cache.store("Market", market, str(as_of_date), name, str(curve_handle))


def price(program_handle: object, market_handle: object) -> np.ndarray:
    """Reprice a cached program against a cached market; return per-instrument PVs."""
    program = cache.load(program_handle, "Program")
    market = cache.load(market_handle, "Market")
    result = program.price(market)
    return np.asarray(result.instrument_pv, dtype=np.float64)
