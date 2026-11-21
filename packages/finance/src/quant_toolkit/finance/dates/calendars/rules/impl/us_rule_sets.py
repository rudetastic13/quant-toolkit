from ..rule_set import HolidayRuleSet
from . import us_rules

US_FED = HolidayRuleSet(
    name="us_fed",
    rules=[
        us_rules.NEW_YEARS_DAY,
        us_rules.JUNTEENTH_DAY,
        us_rules.INDEPENDENCE_DAY,
        us_rules.VETERANS_DAY,
        us_rules.XMAS_DAY,
        us_rules.MLK_DAY,
        us_rules.PRESIDENTS_DAY,
        us_rules.MEMORIAL_DAY,
        us_rules.LABOR_DAY,
        us_rules.COLUMBUS_DAY,
        us_rules.THANKSGIVING_DAY,
    ]
)

US_NYSE = HolidayRuleSet(
    name="us_nyse",
    rules=[
        us_rules.NEW_YEARS_DAY,
        us_rules.JUNTEENTH_DAY_WD,
        us_rules.INDEPENDENCE_DAY_WD,
        us_rules.XMAS_DAY_WD,
        us_rules.MLK_DAY_NYSE,
        us_rules.PRESIDENTS_DAY,
        us_rules.MEMORIAL_DAY,
        us_rules.LABOR_DAY,
        us_rules.THANKSGIVING_DAY,
        us_rules.GOOD_FRIDAY,
        us_rules.NATIONAL_MOURNING,
    ]
)

US_SIFMA = HolidayRuleSet(
    name="us_sifma",
    rules=[
        us_rules.NEW_YEARS_DAY,
        us_rules.JUNTEENTH_DAY_WD,
        us_rules.INDEPENDENCE_DAY_WD,
        us_rules.VETERANS_DAY,
        us_rules.XMAS_DAY_WD,
        us_rules.MLK_DAY,
        us_rules.PRESIDENTS_DAY,
        us_rules.MEMORIAL_DAY,
        us_rules.LABOR_DAY,
        us_rules.COLUMBUS_DAY,
        us_rules.THANKSGIVING_DAY,
        us_rules.GOOD_FRIDAY_EXCLUDE_NFP,
        us_rules.NATIONAL_MOURNING,
    ]
)

US_CME_TREASURY_FUTURES = HolidayRuleSet(
    name="us_cme_treasury_futures",
    rules=[
        us_rules.NEW_YEARS_DAY,
        us_rules.MLK_DAY,
        us_rules.PRESIDENTS_DAY,
        us_rules.MEMORIAL_DAY,
        us_rules.JUNTEENTH_DAY_WD,
        us_rules.INDEPENDENCE_DAY_WD,
        us_rules.LABOR_DAY,
        us_rules.THANKSGIVING_DAY,
        us_rules.XMAS_DAY_WD,
        us_rules.GOOD_FRIDAY_EXCLUDE_NFP,
        us_rules.NATIONAL_MOURNING,
    ]
)

