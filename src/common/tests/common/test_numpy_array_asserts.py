from __future__ import annotations

import numpy as np
import pytest

from common.testing import UnitTest
from common.testing.numpy_array_asserts import (
    assert_array_greater,
    assert_array_greater_equal,
    assert_array_less_equal,
)


class TestNumpyArrayAsserts(UnitTest):
    COVERAGE = ["common.testing.numpy_array_asserts"]

    def test_assert_array_less_equal_passes(self):
        assert_array_less_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert_array_less_equal(np.array([0, 1, 2]), np.array([1, 2, 3]))

    def test_assert_array_less_equal_fails(self):
        with pytest.raises(AssertionError):
            assert_array_less_equal(np.array([2, 2, 2]), np.array([1, 2, 3]))

    def test_assert_array_greater_passes(self):
        assert_array_greater(np.array([2, 3, 4]), np.array([1, 2, 3]))

    def test_assert_array_greater_fails(self):
        with pytest.raises(AssertionError):
            assert_array_greater(np.array([1, 2, 3]), np.array([1, 2, 3]))

    def test_assert_array_greater_equal_passes(self):
        assert_array_greater_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert_array_greater_equal(np.array([2, 3, 4]), np.array([1, 2, 3]))

    def test_assert_array_greater_equal_fails(self):
        with pytest.raises(AssertionError):
            assert_array_greater_equal(np.array([0, 2, 3]), np.array([1, 2, 3]))
