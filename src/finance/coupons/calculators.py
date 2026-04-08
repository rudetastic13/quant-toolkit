"""Define common calculator functions for coupons."""
import numpy as np
from finance.markets import Market


def calculate_fixed(rate: float, out: np.ndarray) -> np.ndarray:
    """Calculate fixed rate, which is just the same rate for all dates.

    Parameters
    ----------
    rate: float
        rate to fill array
    out: np.ndarray
        array to fill with rate, should be pre-allocated with correct shape and dtype

    Returns
    -------
    out: np.ndarray
        array filled with fixed rates, same as input ``out`` parameter
    """
    out[:] = rate
    return out


def calculate_floating(
    rate_index: str,
    market: Market,
    spread: float,
    index_floor: float,
    cap: float,
    floor: float,
    fixing_dates: np.ndarray,
    out: np.ndarray,
    ) -> np.ndarray:
    """Calculate the floating rate applying rate transformations in correct order

    Parameters
    ----------
    rate_index: str
    market: Market
    spread: float
    index_floor: float
    cap: float
    floor: float
    fixing_dates: np.ndarray
    out: np.ndarray

    Returns
    -------
    out: np.ndarray
        array filled with calculated rates, same as input ``out`` parameter
    """
    index_rates = market.get_rates(rate_index, fixing_dates)
    np.maximum(index_rates, index_floor, out=out)
    np.add(out, spread, out=out)
    np.maximum(out, floor, out=out)
    np.minimum(out, cap, out=out)
    return out
