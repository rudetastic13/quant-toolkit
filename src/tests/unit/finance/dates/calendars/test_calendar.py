"""Test the calendar utility"""
import numpy as np
from common.testing import UnitTest
from finance.dates import Date
from finance.dates.calendars.calendar import StaticCalendar


class TestCalendar(UnitTest):
    COVERAGE = ["finance.dates.calendars.calendar"]

    @classmethod
    def setUpClass(cls):
        cls.testing_calendar = StaticCalendar(
            name="test_calendar",
            week_mask="1111100",
            holidays={
                Date(2026, 1, 1),
                Date(2026, 1, 17),
                Date(2026, 2, 13),
                Date(2026, 12, 25),
                Date(2026, 11, 24)
            }
        )

    def test_ctor(self):
        cal = StaticCalendar(
            name="a_calendar",
            week_mask="1111100",
            holidays={Date(2025, 1, 1)}
        )
        self.assertIsInstance(cal, StaticCalendar)

        with self.assertRaises(ValueError):
            _ = StaticCalendar(name="foo", week_mask="11100")

    def test_is_holiday(self):
        self.assertFalse(self.testing_calendar.is_holiday(Date(2026, 1, 2)))
        self.assertTrue(self.testing_calendar.is_holiday(Date(2026, 1, 1)))
        self.assertTrue(self.testing_calendar.is_holiday(Date(2026, 12, 25)))

    def test_is_weekday(self):
        self.assertTrue(self.testing_calendar.is_weekday(Date(2026, 3, 24)))
        self.assertTrue(self.testing_calendar.is_weekday(Date(2026, 3, 23)))
        self.assertFalse(self.testing_calendar.is_weekday(Date(2026, 3, 22)))

    def test_is_business_day(self):
        self.assertTrue(self.testing_calendar.is_business_day(Date(2026, 3, 24)))
        self.assertTrue(self.testing_calendar.is_business_day(Date(2026, 3, 23)))
        self.assertFalse(self.testing_calendar.is_business_day(Date(2026, 3, 22)))
        self.assertFalse(self.testing_calendar.is_business_day(Date(2026, 12, 25)))

    def test_np_calendar(self):
        self.assertIsInstance(self.testing_calendar.np_calendar, np.busdaycalendar)

    def test_add_holidays(self):
        cal = StaticCalendar(
            name="foo",
            week_mask="1111100",
        )
        cal.add_holidays(Date(2026, 1, 1))
        cal.add_holidays(Date(2026, 1, 2))
        self.assertSetEqual(set(cal.holidays), {Date(2026, 1, 1), Date(2026, 1, 2)})
        cal.add_holidays(Date(2026, 1, 3), Date(2026, 1, 4))
        self.assertSetEqual(set(cal.holidays), {Date(2026, 1, 1), Date(2026, 1, 2), Date(2026, 1, 3), Date(2026, 1, 4)})
        cal.remove_holidays(Date(2026, 1, 4), Date(2026, 1, 3))
        self.assertSetEqual(set(cal.holidays), {Date(2026, 1, 1), Date(2026, 1, 2)})

    def test_add_calendars(self):
        cal = StaticCalendar(
            name="foo",
            week_mask="0111000",
        )
        base_cal = self.testing_calendar
        combo_cal = base_cal + cal
        self.assertSetEqual(combo_cal.holidays, self.testing_calendar.holidays)
        self.assertEqual(combo_cal.name, "test_calendar+foo")
        self.assertEqual(combo_cal.week_mask, "0111000")
        with self.assertRaises(TypeError):
            _ = base_cal + 1
