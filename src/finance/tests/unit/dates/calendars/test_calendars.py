"""Test calendars singleton"""
from common.testing import UnitTest
from finance.dates.calendars import Calendars
from finance.dates.calendars.calendar import StaticCalendar

class TestCalendars(UnitTest):
    COVERAGE = ["finance.dates.calendars"]

    def test_calendars_ctor(self):
        """Test building of calendars and show the singleton pattern is respected"""
        self.assertEqual(Calendars._calendars, {})
        calendars = Calendars()
        self.assertListEqual(list(Calendars._calendars.keys()), ["no_holidays"])
        self.assertListEqual(list(calendars._calendars.keys()), ["no_holidays"])
        other_calendars = Calendars()
        self.assertEqual(id(calendars), id(other_calendars))
        Calendars.clear()

    def test_list_calendars(self):
        """Test listing of calendars registered"""
        calendars = Calendars()
        self.assertListEqual(calendars.list_calendars(), ["no_holidays"])
        Calendars.clear()

    def test_get(self):
        """Test getting of 1 calendar"""
        calendars = Calendars()
        self.assertEqual(calendars.get("no_holidays").name, "no_holidays")
        Calendars.clear()

    def test_register(self):
        """Test the registry works as expected"""
        calendars = Calendars()
        other_cal = StaticCalendar(name="carlos", week_mask="1111000")
        calendars.register(other_cal)
        self.assertListEqual(calendars.list_calendars(), ["no_holidays", "carlos"])
        Calendars.clear()

    def test_get_combination(self):
        """Testing the getter but with combo calendars"""
        calendars = Calendars()
        other_cal = StaticCalendar(name="carlos", week_mask="1111000")
        calendars.register(other_cal)
        combo_cal = calendars.get("carlos+no_holidays")
        self.assertEqual(combo_cal.name, "carlos+no_holidays")
        combo_cal = calendars.get("no_holidays+carlos")
        self.assertEqual(combo_cal.name, "carlos+no_holidays")
        Calendars.clear()



