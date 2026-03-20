"""Unit tests for Term class."""
from operator import add, sub
from typing import Callable
from common.testing import UnitTest
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from finance.dates import Term, TermType, Date, Frequency


@pytest.mark.dates
class TestTerm(UnitTest):
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
            [np.array([1, 2], dtype=np.float64), TermType.Days],
            [1.0, TermType.Months],
        )
        for case in cases:
            with self.subTest(case=f"Testing cases : {case}"):
                with self.assertRaises(TypeError):
                    _ = Term(*case)

    def test_validate_ops_raises(self):
        """Test raises in validate ops"""
        dts = np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[D]")
        term = Term(1, TermType.Days)

        with self.assertRaises(TypeError):
            _ = dts.astype("datetime[M]") + term
        with self.assertRaises(NotImplementedError):
            _ = dts * term

    def test_shape(self):
        term = Term(1, TermType.Days)
        self.assertEqual(term.shape, ())
        term = Term(np.array([1,2,3], dtype=np.int32), TermType.Days.Days)
        self.assertEqual(term.shape, (3,))

    def test_size(self):
        term = Term(1, TermType.Days)
        self.assertEqual(term.size, 1)
        term = Term(np.array([1,2,3], dtype=np.int32), TermType.Days.Days)
        self.assertEqual(term.size, 3)

    def test_pos(self):
        term = Term(1, TermType.Days)
        self.assertEqual(+term, term)

    def test_neg(self):
        term = Term(1, TermType.Days)
        self.assertEqual(-term, Term(-1, TermType.Days))

    def test_abs(self):
        term = Term(1, TermType.Days)
        neg_term = Term(-1, TermType.Days.Days)
        self.assertEqual(abs(neg_term), term)
        self.assertEqual(abs(term), term)

    def test_equality(self):
        term = Term(1, TermType.Days)
        neg_term = Term(-1, TermType.Days.Days)
        self.assertNotEqual(term, neg_term)
        self.assertEqual(term, Term(1, TermType.Days))

    def test_from_frequency(self):
        undefined = {Frequency.Once}
        for freq in Frequency.__members__.values():
            if freq in undefined:
                with self.assertRaises(KeyError):
                    _ = Term.from_frequency(freq)
            else:
                self.assertIsInstance(Term.from_frequency(freq), Term)

    def test_invalid_compare_type(self):
        term = Term(1, TermType.Days)
        with self.assertRaises(TypeError):
            _ = term != 1

    def test_from_str(self):
        self.assertIsInstance(Term.from_str("1D"), Term)
        self.assertIsInstance(Term.from_str("1d"), Term)
        with self.assertRaises(SyntaxError):
            _ = Term.from_str("days0")

class TestNpOffset(UnitTest):
    """Base test class for offsets"""

    __test__ = False
    maxDiff = None
    dts = np.array(["2023-01-01", "2023-01-02"], dtype="datetime64[D]")
    term_length = None
    term_type = None
    add_expects = []
    sub_expects = []

    def _test_expects(self, expects: list, op: Callable):
        for expect in expects:
            with self.subTest(f"Testing Term Offset {self.term_type}"):
                term = Term(self.term_length, self.term_type)
                assert_array_equal(op(self.dts, term), expect)

    def test_add(self):
        self._test_expects(self.add_expects, add)

    def test_subtact(self):
        self._test_expects(self.sub_expects, sub)


class TestDayNpOffset(TestNpOffset):
    __test__ = True
    term_length = 1
    term_type = TermType.Days
    add_expects = [np.array(["2023-01-02", "2023-01-03"], dtype="datetime64[D]")]
    sub_expects = [np.array(["2022-12-31", "2023-01-01"], dtype="datetime64[D]")]

class TestWeekNpOffset(TestNpOffset):
    __test__ = True
    term_length = 1
    term_type = TermType.Weeks
    add_expects = [np.array(["2023-01-08", "2023-01-09"], dtype="datetime64[D]")]
    sub_expects = [np.array(["2022-12-25", "2022-12-26"], dtype="datetime64[D]")]

class TestMonthNpOffset(TestNpOffset):
    __test__ = True
    term_length = 1
    term_type = TermType.Months
    add_expects = [np.array(["2023-02-01", "2023-02-02"], dtype="datetime64[D]")]
    sub_expects = [np.array(["2022-12-01", "2022-12-02"], dtype="datetime64[D]")]

class TestQuarterNpOffset(TestNpOffset):
    __test__ = True
    term_length = 1
    term_type = TermType.Quarters
    add_expects = [np.array(["2023-04-01", "2023-04-02"], dtype="datetime64[D]")]
    sub_expects = [np.array(["2022-10-01", "2022-10-02"], dtype="datetime64[D]")]

class TestYearNpOffset(TestNpOffset):
    __test__ = True
    term_length = 1
    term_type = TermType.Years
    add_expects = [np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")]
    sub_expects = [np.array(["2022-01-01", "2022-01-02"], dtype="datetime64[D]")]



class TestDtOffset(UnitTest):
    """Base test class for offsets"""

    __test__ = False
    maxDiff = None
    dts = [Date(2025, 1, 1), Date(2025, 1, 2)]
    term_length = None
    term_type = None
    expects = []
    add_expects = []
    sub_expects = []

    def _test_expects(self, expects: list, op: Callable):
        for idx, expect in enumerate(expects):
            with self.subTest(f"Testing Term Offset {self.term_type}"):
                term = Term(self.term_length, self.term_type)
                self.assertEqual(op(self.dts[idx], term), expect)

    def test_add(self):
        self._test_expects(self.add_expects, add)

    def test_subtact(self):
        self._test_expects(self.sub_expects, sub)


class TestDayDtOffset(TestDtOffset):
    __test__ = True
    term_length = 1
    term_type = TermType.Days
    add_expects = [Date(2025, 1, 2), Date(2025, 1, 3)]
    sub_expects = [Date(2024, 12, 31), Date(2025, 1, 1)]


class TestWeekDtOffset(TestDtOffset):
    __test__ = True
    term_length = 1
    term_type = TermType.Weeks
    add_expects = [Date(2025, 1, 8), Date(2025, 1, 9)]
    sub_expects = [Date(2024, 12, 25), Date(2024, 12, 26)]

class TestMonthDtOffset(TestDtOffset):
    __test__ = True
    term_length = 1
    term_type = TermType.Months
    add_expects = [Date(2025, 2, 1), Date(2025, 2, 2)]
    sub_expects = [Date(2024, 12, 1), Date(2024, 12, 2)]

class TestQuarterDtOffset(TestDtOffset):
    __test__ = True
    term_length = 1
    term_type = TermType.Quarters
    add_expects = [Date(2025, 4, 1), Date(2025, 4, 2)]
    sub_expects = [Date(2024, 10, 1), Date(2024, 10, 2)]

class TestYearDtOffset(TestDtOffset):
    __test__ = True
    term_length = 1
    term_type = TermType.Years
    add_expects = [Date(2026, 1, 1), Date(2026, 1, 2)]
    sub_expects = [Date(2024, 1, 1), Date(2024, 1, 2)]

