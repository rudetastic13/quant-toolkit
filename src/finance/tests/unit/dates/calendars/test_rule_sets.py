"""Test the holiday rules, the US rule sets and the calendars built from them"""
from datetime import date
import pytest
from common.testing import UnitTest
from finance.dates import Date
from finance.dates.calendars import Calendars
from finance.dates.calendars.calendar import StaticCalendar
from finance.dates.calendars.rules import HolidayRule, HolidayRuleSet
from finance.dates.calendars.rules.impl import US_CME_TREASURY_FUTURES, US_FED, US_NYSE, US_SIFMA

REAGAN = Date(2004, 6, 11)
BUSH = Date(2018, 12, 5)
GOOD_FRIDAY_NFP = Date(2026, 4, 3)  # Good Friday on the first Friday of the month
GOOD_FRIDAY = Date(2025, 4, 18)  # an ordinary Good Friday

@pytest.mark.datemath
class TestHolidayRule(UnitTest):
    COVERAGE = ["finance.dates.calendars.rules.rule"]

    def test_fixed(self):
        """A fixed rule generates exactly its dates, and only inside the requested years"""
        rule = HolidayRule.fixed("one offs", date(2004, 6, 11), date(2018, 12, 5))
        self.assertSetEqual(set(rule.generate(2000, 2020)), {REAGAN, BUSH})
        self.assertSetEqual(set(rule.generate(2005, 2020)), {BUSH})
        self.assertSetEqual(set(rule.generate(2019, 2020)), set())

        with self.assertRaises(ValueError):
            _ = HolidayRule.fixed("nothing")

    def test_generate(self):
        """Generation validates the year range and reuses the cached result"""
        rule = HolidayRule.fixed("one offs", date(2004, 6, 11))
        with self.assertRaises(ValueError):
            _ = rule.generate(2020, 2019)
        self.assertIs(rule.generate(2000, 2020), rule.generate(2000, 2020))

@pytest.mark.datemath
class TestHolidayRuleSet(UnitTest):
    COVERAGE = ["finance.dates.calendars.rules.rule_set", "finance.dates.calendars.rules.impl.us_rule_sets"]

    def test_generate(self):
        """A rule set is the union of its rules"""
        rule_set = HolidayRuleSet(
            name="test_set",
            rules=[
                HolidayRule.fixed("reagan", date(2004, 6, 11)),
                HolidayRule.fixed("bush", date(2018, 12, 5)),
            ]
        )
        self.assertEqual(len(rule_set), 2)
        self.assertSetEqual(set(rule_set.generate(2000, 2020)), {REAGAN, BUSH})

        with self.assertRaises(ValueError):
            _ = rule_set.generate(2020, 2019)

    def test_add_remove(self):
        rule_set = HolidayRuleSet(name="test_set")
        rule_set.add(HolidayRule.fixed("reagan", date(2004, 6, 11)))
        self.assertEqual(len(rule_set), 1)
        rule_set.remove("reagan")
        self.assertEqual(len(rule_set), 0)

        with self.assertRaises(KeyError):
            rule_set.remove("reagan")

    def test_national_mourning(self):
        """Reagan and Bush days close the exchanges and the bond market, not the Fed"""
        for rule_set in (US_NYSE, US_SIFMA, US_CME_TREASURY_FUTURES):
            with self.subTest(rule_set=rule_set.name):
                holidays = rule_set.generate(2004, 2018)
                self.assertIn(REAGAN, holidays)
                self.assertIn(BUSH, holidays)

        holidays = US_FED.generate(2004, 2018)
        self.assertNotIn(REAGAN, holidays)
        self.assertNotIn(BUSH, holidays)

    def test_good_friday(self):
        """NYSE always closes; SIFMA and CME treasury futures trade when the jobs report lands on Good Friday"""
        holidays = US_NYSE.generate(2025, 2026)
        self.assertIn(GOOD_FRIDAY, holidays)
        self.assertIn(GOOD_FRIDAY_NFP, holidays)

        for rule_set in (US_SIFMA, US_CME_TREASURY_FUTURES):
            with self.subTest(rule_set=rule_set.name):
                holidays = rule_set.generate(2025, 2026)
                self.assertIn(GOOD_FRIDAY, holidays)
                self.assertNotIn(GOOD_FRIDAY_NFP, holidays)

        holidays = US_FED.generate(2025, 2026)
        self.assertNotIn(GOOD_FRIDAY, holidays)
        self.assertNotIn(GOOD_FRIDAY_NFP, holidays)

    def test_mlk(self):
        """The Fed observes MLK from 1986, NYSE only from 1998"""
        self.assertIn(Date(1997, 1, 20), US_FED.generate(1997, 1998))
        self.assertNotIn(Date(1997, 1, 20), US_NYSE.generate(1997, 1998))
        self.assertIn(Date(1998, 1, 19), US_NYSE.generate(1997, 1998))

@pytest.mark.datemath
class TestRuleSetCalendars(UnitTest):
    COVERAGE = ["finance.dates.calendars", "finance.dates.calendars.calendar"]

    def test_from_rule_set(self):
        """A calendar built from a rule set takes its name and its holidays"""
        cal = StaticCalendar.from_rule_set(US_NYSE, start_year=2018, end_year=2018)
        self.assertIsInstance(cal, StaticCalendar)
        self.assertEqual(cal.name, "us_nyse")
        self.assertTrue(cal.is_holiday(BUSH))
        self.assertFalse(cal.is_holiday(REAGAN))  # outside the generated years

        cal = StaticCalendar.from_rule_set(US_NYSE, name="other_name")
        self.assertEqual(cal.name, "other_name")

    def test_registered(self):
        """The US rule sets are registered when the singleton is built"""
        calendars = Calendars()
        for name in ("us_fed", "us_nyse", "us_sifma", "us_cme_treasury_futures"):
            with self.subTest(name=name):
                self.assertIn(name, calendars.list_calendars())
                self.assertEqual(calendars.get(name).name, name)
        Calendars.clear()

    def test_national_mourning(self):
        calendars = Calendars()
        for name in ("us_nyse", "us_sifma", "us_cme_treasury_futures"):
            with self.subTest(name=name):
                self.assertFalse(calendars.get(name).is_business_day(REAGAN))
                self.assertFalse(calendars.get(name).is_business_day(BUSH))
        self.assertTrue(calendars.get("us_fed").is_business_day(REAGAN))
        self.assertTrue(calendars.get("us_fed").is_business_day(BUSH))
        Calendars.clear()

    def test_good_friday(self):
        calendars = Calendars()
        self.assertFalse(calendars.get("us_nyse").is_business_day(GOOD_FRIDAY_NFP))
        for name in ("us_sifma", "us_cme_treasury_futures"):
            with self.subTest(name=name):
                self.assertTrue(calendars.get(name).is_business_day(GOOD_FRIDAY_NFP))
                self.assertFalse(calendars.get(name).is_business_day(GOOD_FRIDAY))
        Calendars.clear()

    def test_get_combination(self):
        """Combining two rule set calendars unions their holidays"""
        calendars = Calendars()
        combo_cal = calendars.get("us_sifma+us_nyse")
        self.assertEqual(combo_cal.name, "us_nyse+us_sifma")
        self.assertTrue(combo_cal.is_holiday(GOOD_FRIDAY_NFP))
        Calendars.clear()
