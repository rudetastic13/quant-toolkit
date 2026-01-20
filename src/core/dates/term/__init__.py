"""Term offsets, numpy style"""

from __future__ import annotations
from functools import wraps
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from core.dates.enums.term_type import TermType
from typing import Callable, Any

# region, numpy term
_NP_FUNC_MAP = {}


def _register_np_term_math(term_type: TermType) -> Callable:
    def _decorator(func: Callable) -> Callable:
        _NP_FUNC_MAP[term_type] = func
        return func

    return _decorator


@_register_np_term_math(TermType.Days)
def _add_np_day(dt, offset):
    return dt + offset.astype("timedelta64[D]")


@_register_np_term_math(TermType.Weeks)
def _add_np_weeks(dt, offset):
    return _add_np_day(dt, offset * 7)


@_register_np_term_math(TermType.Months)
def _add_np_months(dt, offset):
    start = dt.astype("datetime64[M]")
    target = start + offset.astype("timedelta64[M]")
    days = (dt - start) + 1
    days_in_target = (target + np.timedelta64(1, "M")).astype("datetime64[D]") - target
    days_clamped = np.minimum(days, days_in_target)
    return target.astype("datetime64[D]") + days_clamped - 1


@_register_np_term_math(TermType.Quarters)
def _add_np_quarters(dt, offset):
    return _add_np_months(dt, 3 * offset)


@_register_np_term_math(TermType.Years)
def _add_np_years(dt, offset):
    return _add_np_day(dt, 12 * offset)


@_register_np_term_math(TermType.BusinessDays)
def _add_no_business_days(dt, offset):
    raise ArithmeticError("No BD math supported at term level!")


def _validate_ops(func: Callable) -> Any:
    @wraps(func)
    def wrapper(self, np_func, method, *inputs, **kwargs):
        if np_func not in {np.add, np.subtract}:
            raise NotImplementedError(f"Operation {np_func}, not supported - only allow add and subtract")
        dates = inputs[0]
        if dates.dtype != "datetime64[D]":
            raise TypeError(f"Only support math operations with datetime64[D], given {dates.dtype}")
        value = self.term_length
        if value.ndim and dates.shape != value.shape:
            raise ValueError(f"Shape mismatch, dates {dates.shape} vs term {value.shape}")
        return func(self, np_func, method, *inputs, **kwargs)

    return wrapper


class Term(NDArrayOperatorsMixin):
    """Numpy-based term offsets for calendar dates"""

    def __init__(self, term_length: int | np.int64 | np.ndarray, term_type: TermType):
        if isinstance(term_length, (int)):
            term_length = np.int64(term_length)
        assert isinstance(term_length, (np.ndarray, np.int64)), (
            f"Term length must be int or np.ndarray, got {term_length.__class__.__name__}"
        )
        if type(term_length) is np.ndarray:
            assert term_length.dtype == np.int64, (
                f"The term_length must be np.int64 based, got dtype {term_length.dtype}"
            )
        assert term_type in _NP_FUNC_MAP, f"Term Type {term_type} is not supported in math operations"
        self.term_info = (term_length, term_type)
        self._math = _NP_FUNC_MAP[term_type]

    @property
    def term_length(self):
        return self.term_info[0]

    @property
    def term_type(self):
        return self.term_info[1]

    @property
    def size(self):
        return self.term_length.size

    @property
    def shape(self):
        return self.term_length.shape

    @property
    def dtype(self):
        return Term

    @_validate_ops
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> np.ndarray:
        dates = inputs[0]
        offsets = self.term_length.copy()
        if ufunc == np.subtract:
            offsets *= -1
        return self._math(dates, offsets)


# endregion
