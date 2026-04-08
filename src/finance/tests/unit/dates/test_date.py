"""Testing the Date constructor"""
import datetime
import pandas as pd
import numpy as np
from common.testing import UnitTest
from finance.dates.date import Date

class TestDate(UnitTest):
    COVERAGE = ["finance.dates.date"]

    # simple cases to use in multiple tests
    cases = [
            (2025, 1, 1),
            (2024, 2, 29),
            (2025, 12, 31),
            (2025, 4, 30),

        ]
    def test_simple(self):

        # test constructor
        date = Date(2025, 2, 13)

        # test boundaries
        cases = [
            (0, 1, 1), # year 0 not allowed
            (10_000, 1, 1), # year 10_000 outside of boundaries
            (2025, 1, 32), # only 31 days in january
            (2025, 0, 31), # months must be between 1 and 12, inclusive
            (2025, 13, 1), # month must be between 1 and 12, inclusive
            (2025, 2, 29), # not a leap year
            (2025, 4, 31), # only 30 days in April
            (2025, 5, 32), # only 31 days in May
            (2025, 5, 0), # days must be between 0 and boundary
            (2025, 5, -1), # days can't be negative
        ]
        for case in cases:
            with self.subTest(msg=f"Testing Date constructor with year, month, day {case}"):
                with self.assertRaises(ValueError):
                    _ = Date(*case)

    def test_weekday(self):
        cases = [
            (0, Date(2026, 3, 22)), # Sunday
            (1, Date(2026, 3, 23)), # Monday
            (2, Date(2026, 3, 24)), # Tuesday
            (3, Date(2026, 3, 25)), # Wednesday
            (4, Date(2026, 3, 26)), # Thursday
            (5, Date(2026, 3, 27)), # Friday
            (6, Date(2026, 3, 28)), # Saturday
            (4, Date(1970, 1, 1)), # Thursday
            (3, Date(1969, 12, 31)), # Wednesday
            (2, Date(1969, 12, 30)), # Tuesday
            (1, Date(1969, 12, 29)), # Monday
            (0, Date(1969, 12, 28)), # Sunday
        ]
        for case in cases:
            weekday = case[0]
            dt = case[1]
            with self.subTest(case=f"Testing date weekday for {dt}"):
                self.assertEqual(dt.weekday(), weekday)

    def test_isoweekday(self):
        cases = [
            (6, Date(2026, 3, 22)), # Sunday
            (0, Date(2026, 3, 23)), # Monday
            (1, Date(2026, 3, 24)), # Tuesday
            (2, Date(2026, 3, 25)), # Wednesday
            (3, Date(2026, 3, 26)), # Thursday
            (4, Date(2026, 3, 27)), # Friday
            (5, Date(2026, 3, 28)), # Saturday
            (3, Date(1970, 1, 1)), # Thursday
            (2, Date(1969, 12, 31)), # Wednesday
            (1, Date(1969, 12, 30)), # Tuesday
            (0, Date(1969, 12, 29)), # Monday
            (6, Date(1969, 12, 28)), # Sunday
        ]
        for case in cases:
            isoweekday = case[0]
            dt = case[1]
            with self.subTest(case=f"Testing date isoweekday for {dt}"):
                self.assertEqual(dt.isoweekday(), isoweekday)

    def test_to_ymd(self):
        """Testing to ymd"""
        for case in self.cases:
            with self.subTest(msg=f"Testing Date to_ymd with year, month, day {case}"):
                date = Date(*case)
                ymd = date.to_ymd()
                self.assertEqual(ymd, case)

    def test_to_str(self):
        """Testing to string"""
        for case in self.cases:
            with self.subTest(msg=f"Testing Date to_str with year, month, day {case}"):
                date = Date(*case)
                str_date = date.to_str()
                expected_str = f"{case[0]:04d}-{case[1]:02d}-{case[2]:02d}"
                self.assertEqual(str_date, expected_str)

    def test_to_int(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date to_int with year, month, day {case}"):
                date = Date(*case)
                int_date = date.to_int()
                expected_int = case[0] * 10_000 + case[1] * 100 + case[2]
                self.assertEqual(int_date, expected_int)

    def test_to_numpy(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date to_numpy with year, month, day {case}"):
                date = Date(*case)
                date_str = f"{case[0]:04d}-{case[1]:02d}-{case[2]:02d}"
                np_date = date.to_numpy()
                expected_np = np.datetime64(date_str, "D")
                self.assertEqual(np_date, expected_np)

    def test_to_pandas(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date to_pandas with year, month, day {case}"):
                date = Date(*case)
                date_str = f"{case[0]:04d}-{case[1]:02d}-{case[2]:02d}"
                pd_date = date.to_pandas()
                expected_pd = pd.Timestamp(date_str)
                self.assertEqual(pd_date, expected_pd)

    def test_to_ordinal(self):
        EPOCH = datetime.date(1970, 1, 1).toordinal()
        for case in self.cases:
            with self.subTest(msg=f"Testing Date to_pandas with year, month, day {case}"):
                my_date = Date(*case)
                dt = datetime.date(*case)
                self.assertEqual(my_date.toordinal(), dt.toordinal() - EPOCH)

    def test_to_date(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date to_date with year, month, day {case}"):
                date = Date(*case)
                dt = date.to_date()
                expected_dt = datetime.date(*case)
                self.assertEqual(dt, expected_dt)

    def test_isoformat(self):
        self.assertEqual(Date(2025, 1, 1).isoformat(), "2025-01-01")

    def test_to_datetime(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date to_datetime with year, month, day {case}"):
                date = Date(*case)
                dt = date.to_datetime()
                expected_dt = datetime.datetime(*case)
                self.assertEqual(dt, expected_dt)

    def test_from_ymd(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date from_ymd with year, month, day {case}"):
                date = Date.from_ymd(case)
                self.assertEqual(date.to_ymd(), case)

    def test_from_str(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date from_str with year, month, day {case}"):
                date_str = f"{case[0]:04d}-{case[1]:02d}-{case[2]:02d}"
                date = Date.from_str(date_str)
                self.assertEqual(date.to_ymd(), case)

    def test_from_int(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date from_int with year, month, day {case}"):
                int_date = case[0] * 10_000 + case[1] * 100 + case[2]
                date = Date.from_int(int_date)
                self.assertEqual(date.to_ymd(), case)

    def test_from_numpy(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date from_numpy with year, month, day {case}"):
                date_str = f"{case[0]:04d}-{case[1]:02d}-{case[2]:02d}"
                np_date = np.datetime64(date_str, "D")
                date = Date.from_numpy(np_date)
                self.assertEqual(date.to_ymd(), case)

    def test_from_pandas(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date from_pandas with year, month, day {case}"):
                date_str = f"{case[0]:04d}-{case[1]:02d}-{case[2]:02d}"
                pd_date = pd.Timestamp(date_str)
                date = Date.from_pandas(pd_date)
                self.assertEqual(date.to_ymd(), case)

    def test_from_ordinal(self):
        EPOCH = datetime.date(1970, 1, 1).toordinal()
        for case in self.cases:
            with self.subTest(msg=f"Testing Date fromordinal with year, month, day {case}"):
                dt = datetime.date(*case)
                ordinal = dt.toordinal() - EPOCH
                date = Date.fromordinal(ordinal)
                self.assertEqual(date.to_ymd(), case)

    def test_from_date(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date from_date with year, month, day {case}"):
                dt = datetime.date(*case)
                date = Date.from_date(dt)
                self.assertEqual(date.to_ymd(), case)

    def test_from_datetime(self):
        for case in self.cases:
            with self.subTest(msg=f"Testing Date from_date with year, month, day {case}"):
                dt = datetime.datetime(*case)
                date = Date.from_datetime(dt)
                self.assertEqual(date.to_ymd(), case)

    def test_today(self):
        today = datetime.date.today()
        date_today = Date.today()
        self.assertEqual(date_today.to_ymd(), (today.year, today.month, today.day))

    def test_replace(self):
        date = Date(2025, 2, 13)
        date.replace(year=2024, month=2, day=29)
        self.assertEqual(date.to_ymd(), (2024, 2, 29))

    def test_operators(self):
        self.assertTrue(Date(2025, 2, 13) == Date(2025, 2, 13))
        self.assertFalse(Date(2025, 2, 13) == Date(2025, 2, 14))
        self.assertTrue(Date(2025, 2, 13) != Date(2025, 2, 14))
        self.assertFalse(Date(2025, 2, 13) != Date(2025, 2, 13))
        self.assertTrue(Date(2025, 2, 13) < Date(2025, 2, 14))
        self.assertFalse(Date(2025, 2, 13) < Date(2025, 2, 13))
        self.assertTrue(Date(2025, 2, 13) <= Date(2025, 2, 13))
        self.assertFalse(Date(2025, 2, 13) <= Date(2025, 2, 12))
        self.assertTrue(Date(2025, 2, 13) > Date(2025, 2, 12))
        self.assertFalse(Date(2025, 2, 13) > Date(2025, 2, 13))
        self.assertTrue(Date(2025, 2, 13) >= Date(2025, 2, 13))
        self.assertFalse(Date(2025, 2, 13) >= Date(2025, 2, 14))

        from operator import lt, gt, eq
        for case in [lt, gt, eq]:
            with self.subTest(msg=f"Testing Date operators with year, month, day {case}"):
                with self.assertRaises(TypeError):
                    _ = case(Date(2025, 2, 13), "2025-02-14")

    def test_repr(self):
        result = repr(Date(2025, 2, 13))
        expected = "Date(2025, 2, 13)"
        self.assertEqual(result, expected)

    def test_str(self):
        result = str(Date(2025, 2, 13))
        expected = "2025-02-13"
        self.assertEqual(result, expected)

    def test_hash(self):
        date1 = Date(2025, 2, 13)
        date2 = Date(2025, 2, 13)
        date3 = Date(2025, 2, 14)
        self.assertEqual(hash(date1), hash(date2))
        self.assertNotEqual(hash(date1), hash(date3))
