"""Common us market rules, I ignore early close for now"""
from datetime import date, datetime
from dateutil.rrule import MO, TH, YEARLY, rrule
from ...rules import HolidayRule, nearest_weekday, not_first_week_of_month, sunday_to_monday

# region, constants for starting rules - I start from 1970
_DT_START = {"dtstart" : datetime(1970, 1 , 1)}
_JUNTEENTH_DT_START = {"dtstart" : datetime(2022, 1 , 1)}
_MLK_DT_START = {"dtstart" : datetime(1986, 1 , 1)}
_MLK_NYSE_DT_START = {"dtstart" : datetime(1998, 1 , 1)}  # NYSE first closed for MLK in 1998
# endregion

# exclusion set
_exclude_sat = lambda dt: dt.isoweekday() != 6


# region, static holidays

NEW_YEARS_DAY = HolidayRule(
    name="New Year's Day",
    rule=rrule(YEARLY, bymonth=1, bymonthday=1, **_DT_START),
    observance=sunday_to_monday,
    dt_filter=_exclude_sat,
)

NEW_YEARS_DAY_WD = HolidayRule(
    name="New Year's Day",
    rule=rrule(YEARLY, bymonth=1, bymonthday=1, **_DT_START),
    observance=nearest_weekday,
)

JUNTEENTH_DAY = HolidayRule(
    name="Juneteenth",
    rule=rrule(YEARLY, bymonth=6, bymonthday=19, **_JUNTEENTH_DT_START),
    observance=sunday_to_monday,
)

JUNTEENTH_DAY_WD = HolidayRule(
    name="Juneteenth",
    rule=rrule(YEARLY, bymonth=6, bymonthday=19, **_JUNTEENTH_DT_START),
    observance=nearest_weekday,
)

INDEPENDENCE_DAY = HolidayRule(
    name="July 4th",
    rule=rrule(YEARLY, bymonth=7, bymonthday=4, **_DT_START),
    observance=sunday_to_monday,
)

INDEPENDENCE_DAY_WD = HolidayRule(
    name="July 4th",
    rule=rrule(YEARLY, bymonth=7, bymonthday=4, **_DT_START),
    observance=nearest_weekday,
)

VETERANS_DAY = HolidayRule(
    name="Veteran's Day",
    rule=rrule(YEARLY, bymonth=11, bymonthday=11, **_DT_START),
    observance=sunday_to_monday,
    dt_filter=_exclude_sat,
)

VETERANS_DAY_WD = HolidayRule(
    name="Veteran's Day",
    rule=rrule(YEARLY, bymonth=11, bymonthday=11, **_DT_START),
    observance=nearest_weekday,
)

XMAS_DAY = HolidayRule(
    name="Christmas",
    rule=rrule(YEARLY, bymonth=12, bymonthday=25, **_DT_START),
    observance=sunday_to_monday,
)

XMAS_DAY_WD = HolidayRule(
    name="Christmas",
    rule=rrule(YEARLY, bymonth=12, bymonthday=25, **_DT_START),
    observance=nearest_weekday,
)

# endregion

# region, floating holidays
MLK_DAY = HolidayRule(
    name="MLK",
    rule=rrule(YEARLY, bymonth=1, byweekday=MO(3), **_MLK_DT_START),
)

MLK_DAY_NYSE = HolidayRule(
    name="MLK",
    rule=rrule(YEARLY, bymonth=1, byweekday=MO(3), **_MLK_NYSE_DT_START),
)

PRESIDENTS_DAY = HolidayRule(
    name="President's Day",
    rule=rrule(YEARLY, bymonth=2, byweekday=MO(3), **_DT_START),
)

MEMORIAL_DAY = HolidayRule(
    name="Memorial Day",
    rule=rrule(YEARLY, bymonth=5, byweekday=MO(-1), **_DT_START),
)

LABOR_DAY = HolidayRule(
    name="Labor Day",
    rule=rrule(YEARLY, bymonth=9, byweekday=MO(1), **_DT_START),
)

COLUMBUS_DAY = HolidayRule(
    name="Columbus Day",
    rule=rrule(YEARLY, bymonth=10, byweekday=MO(2), **_DT_START),
)

THANKSGIVING_DAY = HolidayRule(
    name="Thanksgiving Day",
    rule=rrule(YEARLY, bymonth=11, byweekday=TH(4), **_DT_START),
)

GOOD_FRIDAY = HolidayRule(
    name="Good Friday",
    rule=rrule(YEARLY, byeaster=-2, **_DT_START),
)

GOOD_FRIDAY_EXCLUDE_NFP = HolidayRule(
    name="Good Friday",
    rule=rrule(YEARLY, byeaster=-2, **_DT_START),
    dt_filter=not_first_week_of_month
)
# endregion

# region, one-off closures
NATIONAL_MOURNING = HolidayRule.fixed(
    "National Day of Mourning",
    date(2004, 6, 11),  # Ronald Reagan
    date(2018, 12, 5),  # George H. W. Bush
)
# endregion
