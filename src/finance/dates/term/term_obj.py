from __future__ import annotations
import re
from typing import Callable, Any, Generic, overload
from functools import wraps
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from finance.dates.date import Date
from finance.dates.enums.frequencies import Frequency
from .term_math_registry import (
    date_term_math,
    np_term_math,
    IntT,
    DateT,
    DateScalar,
    DateNpT,
    IntNpt
)
from .term_type import TermType

# constant mapping of frequencies to term
_FREQ_MAP = {
    Frequency.Daily: (1, TermType.Days),
    Frequency.Weekly: (1, TermType.Weeks),
    Frequency.BiWeekly: (2, TermType.Weeks),
    Frequency.Monthly: (1, TermType.Months),
    Frequency.BiMonthly: (2, TermType.Months),
    Frequency.Quarterly: (1, TermType.Quarters),
    Frequency.SemiAnnually: (6, TermType.Months),
    Frequency.Annually: (1, TermType.Years),
    Frequency.TwoYearly: (2, TermType.Years),
    Frequency.ThreeYearly: (3, TermType.Years),
    Frequency.FiveYearly: (5, TermType.Years),
    Frequency.SevenYearly: (7, TermType.Years),
    Frequency.TenYearly: (10, TermType.Years),
    Frequency.FifteenYearly: (15, TermType.Years),
    Frequency.TwentyYearly: (20, TermType.Years),
    Frequency.ThirtyYearly: (30, TermType.Years),
}

# region, decorators
def _validate_term_input(func: Callable) -> Any:
    @wraps(func)
    def wrapper(self, value: Any) -> Any:
        if not isinstance(value, Term):
            raise TypeError(f"Input value must be of type Term, got {type(value)}")
        return func(self, value)
    return wrapper
# endregion

class Term(Generic[IntT], NDArrayOperatorsMixin):

    def __init__(self, term_length: IntT, term_type: TermType):
        self.term_info = (term_length, term_type)

    @property
    def term_length(self):
        return self.term_info[0]

    @property
    def term_type(self):
        return self.term_info[1]

    @overload
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: tuple[DateT, "Term[int]"], **kwargs) -> DateScalar:
        ...

    @overload
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: tuple[DateNpT, "Term[IntNpt]"], **kwargs) -> DateNpT:
        ...

    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs, **kwargs) -> DateT:
        dates, _ = inputs
        if type(dates) is Date:
            if ufunc == np.subtract:
                return date_term_math(self.term_type, dates, -self.term_length)
            elif ufunc == np.add:
                return date_term_math(self.term_type, dates, self.term_length)
            else:
                raise NotImplementedError(f"Operation {ufunc} not supported for Term and Date")

        offsets = np.int64(self.term_length) if type(self.term_length) is int else self.term_length
        if ufunc == np.subtract:
            return np_term_math(self.term_type, dates, -1 * offsets)
        elif ufunc == np.add:
            return np_term_math(self.term_type, dates, offsets)
        else:
            raise NotImplementedError(f"Operation {ufunc} not supported for Term and Date array")

    @classmethod
    def from_str(cls, value: str) -> Term:
        if match:= re.match(r"(-?\d+)\s*([a-zA-Z]+)", value):
            term_length = int(match.group(1))
            term_type = TermType(match.group(2))
            return cls(term_length, term_type)

        raise SyntaxError(f"Could not parse Term {value}")

    @classmethod
    def from_frequency(cls, frequency: Frequency) -> Term:
        return Term(*_FREQ_MAP[frequency])

    def __neg__(self) -> Term:
        return Term(-self.term_length, self.term_type)

    def __pos__(self) -> Term:
        return self

    def __abs__(self) -> Term:
        return Term(abs(self.term_length), self.term_type)

    @_validate_term_input
    def __eq__(self, other: Term) -> bool:
        return self.term_info == other.term_info

    @_validate_term_input
    def __ne__(self, other: Term) -> bool:
        return self.term_info != other.term_info

