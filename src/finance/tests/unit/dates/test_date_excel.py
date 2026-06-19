"""Testing Date <-> Excel 1900-system serial conversions."""
import pytest

from common.testing import UnitTest
from finance.dates.date import Date


@pytest.mark.datemath
class TestDateExcel(UnitTest):
    COVERAGE = ["finance.dates.date"]

    # well-known Excel serials (1900 date system; day 0 is 1899-12-30)
    anchors = [
        (Date(1970, 1, 1), 25_569),   # numpy/unix epoch
        (Date(2021, 1, 1), 44_197),   # commonly cited reference serial
        (Date(1900, 1, 1), 2),        # 1900-01-01 is serial 2 in the offset convention used
    ]

    def test_to_excel_anchors(self):
        for date, serial in self.anchors:
            with self.subTest(date=date):
                self.assertEqual(date.to_excel(), serial)

    def test_from_excel_anchors(self):
        for date, serial in self.anchors:
            with self.subTest(serial=serial):
                self.assertEqual(Date.from_excel(serial), date)

    def test_roundtrip(self):
        for date, _ in self.anchors + [(Date(1999, 12, 31), 0), (Date(2050, 7, 15), 0)]:
            with self.subTest(date=date):
                self.assertEqual(Date.from_excel(date.to_excel()), date)

    def test_to_excel_is_int(self):
        self.assertIsInstance(Date(2026, 6, 1).to_excel(), int)

    def test_fractional_serial_truncates_to_day(self):
        # intraday time component is dropped — same day as the integer part
        self.assertEqual(Date.from_excel(44_197.99), Date(2021, 1, 1))
        self.assertEqual(Date.from_excel(44_197.0), Date(2021, 1, 1))

    def test_monotonic(self):
        # later dates have strictly larger serials
        earlier = Date(2026, 6, 1).to_excel()
        later = Date(2026, 6, 2).to_excel()
        self.assertEqual(later - earlier, 1)
