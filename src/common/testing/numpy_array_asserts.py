"""Extend numpy array assertions"""
import operator
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_array_less,
    assert_array_compare
)

__all__ = [
    "assert_array_equal",
    "assert_array_almost_equal",
    "assert_array_less",
    "assert_array_less_equal",
    "assert_array_greater",
    "assert_array_greater_equal",
]

def assert_array_less_equal(x, y, err_msg='', verbose=True, strict=False):
    """Assert that x is less than or equal to y elementwise."""
    __tracebackhide__ = True
    assert_array_compare(operator.__le__, x, y, err_msg=err_msg, verbose=verbose, strict=strict)

def assert_array_greater(x, y, err_msg='', verbose=True, strict=False):
    """Assert that x is less than or equal to y elementwise."""
    __tracebackhide__ = True
    assert_array_compare(operator.__gt__, x, y, err_msg=err_msg, verbose=verbose, strict=strict)

def assert_array_greater_equal(x, y, err_msg='', verbose=True, strict=False):
    """Assert that x is less than or equal to y elementwise."""
    __tracebackhide__ = True
    assert_array_compare(operator.__ge__, x, y, err_msg=err_msg, verbose=verbose, strict=strict)