"""Unit tests for Term class."""

import unittest
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from dates import Term, TermType


@pytest.mark.unittest
@pytest.mark.dates
class TestTerm(unittest.TestCase):
    """Testing Term"""

    def test_term_creation(self):
        """Test creation"""
        term = Term(1, TermType.Days)
        self.assertEqual(term.term_length, 1)
        self.assertEqual(term.term_type, TermType.Days)
        self.assertEqual(term.size, 1)
        self.assertEqual(term.shape, ())

    def test_ctor_raises(self):
        """Test raises in constructor"""
        cases = (
            [1, "invalid"],
            [1, None],
            [1, 3.14],
            [np.float64(1), TermType.Days],
            [np.array([1, 2], dtype=np.int32), TermType.Days],
            [1.0, TermType.Months],
        )
        for case in cases:
            with self.subTest(case=f"Testing cases : {case}"):
                with self.assertRaises(AssertionError):
                    _ = Term(*case)

    def test_validate_ops_raises(self):
        """Test raises in validate ops"""
        dts = np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[D]")
        term = Term(1, TermType.Days)

        with self.assertRaises(TypeError):
            _ = dts.astype("datetime[M]") + term
        with self.assertRaises(NotImplementedError):
            _ = dts * term
        with self.assertRaises(ValueError):
            _ = dts + Term(np.array([1, 2, 3], dtype=np.int64), TermType.Days)

    class TestOffsets(unittest.TestCase):
        """Base test class for offsets"""

        __test__ = False
        maxDiff = None
        dts = np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[D]")
        term_length = None
        term_type = None
        expects = []

        def test_cases(self):
            for expect in self.expects:
                with self.subTest(f"Testing Term Offset {self.term_type}"):
                    term = Term(self.term_length, self.term_type)
                    assert_array_equal(self.dts + term, expect)

    class TestDayOffset(TestOffsets):
        __test__ = True
        term_length = 1
        term_type = TermType.Days
        expects = [np.array(["2023-01-02", "2023-01-03"], dtype="datetime64[D]")]

    class TestWeekOffsets(TestOffsets):
        __test__ = True
        term_length = 1
        term_type = TermType.Weeks
        expects = [np.array(["2023-01-08", "2023-01-09"], dtype="datetime64[D]")]

    class TestMonthOffset(TestOffsets):
        __test__ = True
        term_length = 1
        term_type = TermType.Months
        expects = [np.array(["2023-02-01", "2023-02-01"], dtype="datetime64[D]")]

    class TestQuarterOffset(TestOffsets):
        __test__ = True
        term_length = 1
        term_type = TermType.Quarters
        expects = [np.array(["2023-04-01", "2023-04-02"], dtype="datetime64[D]")]

    class TestYearOffset(TestOffsets):
        __test__ = True
        term_length = 1
        term_type = TermType.Years
        expects = [np.array(["2024-01-01", "2024-01-01"], dtype="datetime64[D]")]


if __name__ == "__main__":
    unittest.main()
