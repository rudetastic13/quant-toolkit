"""Tests for Excel <-> domain marshalling."""
import datetime as dt

import numpy as np

from common.testing import UnitTest
from finance.dates import Date
from quant_toolkit_xl import marshalling as m


class TestFlatten(UnitTest):
    COVERAGE = ["quant_toolkit_xl.marshalling"]

    def test_scalar(self):
        self.assertEqual(m.flatten(5), [5])

    def test_none_is_empty(self):
        self.assertEqual(m.flatten(None), [])

    def test_1d(self):
        self.assertEqual(m.flatten([1, 2, 3]), [1, 2, 3])

    def test_2d_range(self):
        # Excel hands a column/area as nested lists
        self.assertEqual(m.flatten([[1, 2], [3]]), [1, 2, 3])

    def test_ndarray(self):
        self.assertEqual(m.flatten(np.array([[1.0], [2.0]])), [1.0, 2.0])


class TestToDate(UnitTest):
    COVERAGE = ["quant_toolkit_xl.marshalling"]

    target = Date(2026, 6, 1)

    def test_dispatch_over_types(self):
        cases = [
            dt.datetime(2026, 6, 1, 13, 30),   # datetime -> day
            dt.date(2026, 6, 1),
            np.datetime64("2026-06-01"),
            "2026-06-01",
            self.target,                       # Date passthrough
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertEqual(m.to_date(value), self.target)

    def test_excel_serial(self):
        # numeric input is treated as an Excel serial via Date.from_excel
        self.assertEqual(m.to_date(44_197), Date(2021, 1, 1))

    def test_first_of_range(self):
        self.assertEqual(m.to_date([["2026-06-01", "2026-07-01"]]), self.target)

    def test_empty_range_raises(self):
        with self.assertRaises(ValueError):
            m.to_date([])

    def test_unhandled_type_raises(self):
        with self.assertRaises(TypeError):
            m.to_date(object())


class TestArrayHelpers(UnitTest):
    COVERAGE = ["quant_toolkit_xl.marshalling"]

    def test_to_datetime64_array(self):
        out = m.to_datetime64_array(["2026-06-01", "2026-07-01"])
        self.assertEqual(out.dtype, np.dtype("datetime64[D]"))
        self.assertEqual(list(out.astype(str)), ["2026-06-01", "2026-07-01"])

    def test_to_datetime64_array_mixed_inputs(self):
        out = m.to_datetime64_array([dt.date(2026, 6, 1), 44_197])
        self.assertEqual(list(out.astype(str)), ["2026-06-01", "2021-01-01"])

    def test_to_float_array(self):
        out = m.to_float_array([[1], [2], [3]])
        self.assertEqual(out.dtype, np.float64)
        self.assertEqual(list(out), [1.0, 2.0, 3.0])

    def test_handles_drops_blanks(self):
        self.assertEqual(
            m.handles([["Curve::aa"], [""], [None], ["Swap::bb"]]),
            ["Curve::aa", "Swap::bb"],
        )
