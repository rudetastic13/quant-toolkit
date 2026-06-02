"""Define common calculator functions for coupons."""
import numpy as np
from finance.markets import Market
from finance.instruments.enums import MarginTreatment, CouponType
from finance.instruments.schedules.coupon_schedule import CouponSchedule


def calculate_fixed(coupon_rate: float, out: np.ndarray) -> np.ndarray:
    """Calculate fixed rate, which is just the same rate for all dates.

    Parameters
    ----------
    coupon_rate: float
        rate to fill array
    out: np.ndarray
        array to fill with rate, should be pre-allocated with correct shape and dtype

    Returns
    -------
    out: np.ndarray
        array filled with fixed rates, same as input ``out`` parameter
    """
    out[:] = coupon_rate
    return out


def calculate_floating(
    rate_index: str,
    spread: float,
    index_floor: float,
    cap: float,
    floor: float,
    market: Market,
    reset_dates: np.ndarray,
    out: np.ndarray,
    ) -> np.ndarray:
    """Calculate the floating rate applying rate transformations in correct order

    Parameters
    ----------
    rate_index: str
    spread: float
    index_floor: float
    cap: float
    floor: float
    market: Market
    reset_dates: np.ndarray
    out: np.ndarray

    Returns
    -------
    out: np.ndarray
        array filled with calculated rates, same as input ``out`` parameter
    """
    out[:] = market.get_rates(rate_index, reset_dates)
    if index_floor is not None:
        np.maximum(out, index_floor, out=out)
    np.add(out, spread, out=out)
    if floor is not None:
        np.maximum(out, floor, out=out)
    if cap is not None:
        np.minimum(out, cap, out=out)
    return out

def calculate_geometric_average(
    rate_index: str,
    spread: float,
    index_floor: float,
    cap: float,
    floor: float,
    margin_treatment: MarginTreatment,
    market: Market,
    fixing_dates: np.ndarray,
    rate_weights: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Calculate the geometric average of the index rates for the given fixing dates.

    Parameters
    ----------
    rate_index: str
    spread: float
    index_floor: float
    cap: float
    floor: float
    margin_treatment: MarginTreatment
    market: Market
    fixing_dates: np.ndarray
    rate_weights: np.ndarray
    out: np.ndarray

    Returns
    -------
    out: np.ndarray
        array filled with calculated geometric average rates, same as input ``out`` parameter
    """
    for idx in range(out.shape[0]):
        index_rates = market.get_rates(rate_index, fixing_dates)
        if index_floor:
            np.maximum(index_rates, index_floor, out=index_rates)
        if margin_treatment == MarginTreatment.Inclusive:
            np.add(index_rates, spread, out=index_rates)
        np.multiply(index_rates, rate_weights, out=index_rates)
        np.add(index_rates, 1, out=index_rates)
        np.cumprod(index_rates, out=index_rates)
        np.divide(index_rates, rate_weights.cumsum(), out=index_rates)
        np.subtract(index_rates, 1, out=index_rates)
        if margin_treatment == MarginTreatment.Exclusive:
            np.add(index_rates, spread, out=index_rates)
        if floor:
            np.maximum(index_rates, floor, out=index_rates)
        if cap:
            np.minimum(index_rates, cap, out=index_rates)
        out[idx] = index_rates[idx]
    return out


def calculate_arithmetic_average(
    rate_index: str,
    spread: float,
    index_floor: float,
    cap: float,
    floor: float,
    margin_treatment: MarginTreatment,
    market: Market,
    fixing_dates: np.ndarray,
    rate_weights: np.ndarray,
    out: np.ndarray,
) -> np.ndarray:
    """Calculate the arithmetic average of the index rates for the given fixing dates.

    Parameters
    ----------
    rate_index: str
    spread: float
    index_floor: float
    cap: float
    floor: float
    margin_treatment: MarginTreatment
    market: Market
    fixing_dates: np.ndarray
    rate_weights: np.ndarray
    out: np.ndarray

    Returns
    -------
    out: np.ndarray
        array filled with calculated geometric average rates, same as input ``out`` parameter
    """
    for idx in range(out.shape[0]):
        index_rates = market.get_rates(rate_index, fixing_dates)
        if index_floor:
            np.maximum(index_rates, index_floor, out=index_rates)
        if margin_treatment == MarginTreatment.Inclusive:
            np.add(index_rates, spread, out=index_rates)
        np.multiply(index_rates, rate_weights, out=index_rates)
        np.cumsum(index_rates, out=index_rates)
        np.divide(index_rates, rate_weights.cumsum(), out=index_rates)
        if margin_treatment == MarginTreatment.Exclusive:
            np.add(index_rates, spread, out=index_rates)
        if floor:
            np.maximum(index_rates, floor, out=index_rates)
        if cap:
            np.minimum(index_rates, cap, out=index_rates)
        out[idx] = index_rates[idx]
    return out

def calculate_custom(
    schedule: CouponSchedule,
    accrual_grid: np.ndarray,
    out: np.ndarray,
    market: Market | None = None,
    reset_dates: np.ndarray | None = None,
    fixing_dates: np.ndarray | None = None,
    rate_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate custom rate based on user-defined logic.

    Parameters
    ----------
    schedule: CouponSchedule
        schedule of coupon events defining the logic for rate calculations, should be aligned with accrual grid
    accrual_grid: np.ndarray
        array of accrual dates for which rates need to be calculated, should be aligned with schedule
    out: np.ndarray
        pre-allocated array to fill with calculated rates, should have correct shape and dtype
    market: Market, optional
        market data provider, can be used to fetch rates or other market data for calculations
    reset_dates: np.ndarray, optional
        array of reset dates relevant for the calculation, can be used to fetch time series data from
    fixing_dates: np.ndarray, optional
        array of fixing dates relevant for the calculation, can be used to fetch time series data from market
    rate_weights: np.ndarray, optional
        array of weights for rate calculations, can be used in averaging or other weighted calculations

    Returns
    -------
    out: np.ndarray
        array filled with calculated custom rates, same as input ``out`` parameter
    """
    # Placeholder implementation, replace with actual custom logic as needed
    schedule_arr = schedule.schedule_from_accrual_grid(accrual_grid)
    bool_mask = np.full_like(schedule_arr, False, dtype=bool)
    for idx in np.unique(schedule_arr):
        if idx == 0:
            bool_mask[:] = True
        else:
            bool_mask[:] = schedule_arr == idx
        event = schedule.events[idx]
        params = {"out" : out[bool_mask]}
        if event.coupon_type == CouponType.Fixed:
            pass # nothing to do
        elif event.coupon_type == CouponType.Floating:
            params.update(
                {"market" : market,
                  "reset_dates" : reset_dates,})
        else:
            params = {"market" : market,
                      "fixing_dates" : fixing_dates,
                      "rate_weights" : rate_weights,}

        out[bool_mask] = event.calculate(**params)
        bool_mask[:] = False
    return out
