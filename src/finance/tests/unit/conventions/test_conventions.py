"""Tests for the MarketConventions model and ConventionRegistry."""
from dataclasses import FrozenInstanceError, replace

from common.testing import UnitTest
from finance.dates import Term
from finance.dates.term.term_type import TermType
from finance.dates.enums import BDC, DayCountMethod, Frequency, Roll
from finance.instruments.enums import CouponType
from finance.conventions import (
    MATCH_TENOR,
    ConventionRegistry,
    CurveKey,
    DepositConventions,
    FixedLegConventions,
    FloatLegConventions,
    FraConventions,
    IndexFamily,
    MarketConventions,
    SwapConventions,
    frequency_for_tenor,
)


def _make_market(**index_overrides) -> MarketConventions:
    family = IndexFamily(
        currency="USD",
        name="TEST",
        day_count_method=DayCountMethod.Actual360,
        fixing_calendar="NYC",
        **index_overrides,
    )
    spot_lag = Term(2, TermType.BusinessDays)
    return MarketConventions(
        family=family,
        swap=SwapConventions(
            fixed_leg=FixedLegConventions(
                payment_frequency=Frequency.SemiAnnually,
                day_count_method=DayCountMethod.Thirty360,
                business_day_convention=BDC.ModifiedFollowing,
                roll_convention=Roll.EOM,
                pay_calendar="NYC",
            ),
            float_leg=FloatLegConventions(
                payment_frequency=Frequency.Quarterly,
                coupon_type=CouponType.GeometricAveraged,
                reset_frequency=Frequency.Daily,
                business_day_convention=BDC.ModifiedFollowing,
                roll_convention=Roll.EOM,
                pay_calendar="NYC",
            ),
            spot_lag=spot_lag,
            spot_calendar="NYC",
        ),
        deposit=DepositConventions(
            spot_lag=spot_lag, business_day_convention=BDC.ModifiedFollowing, calendar="NYC",
        ),
        fra=FraConventions(
            spot_lag=spot_lag, business_day_convention=BDC.ModifiedFollowing, calendar="NYC",
        ),
    )


class TestMarketConventions(UnitTest):
    COVERAGE = ["finance.conventions.market_conventions"]

    def test_per_leg_conventions_differ(self):
        m = _make_market()
        self.assertEqual(m.swap.fixed_leg.payment_frequency, Frequency.SemiAnnually)
        self.assertEqual(m.swap.float_leg.payment_frequency, Frequency.Quarterly)
        self.assertEqual(m.swap.fixed_leg.day_count_method, DayCountMethod.Thirty360)

    def test_float_day_count_defaults_from_index(self):
        m = _make_market()
        self.assertIsNone(m.swap.float_leg.day_count_method)
        self.assertEqual(m.float_leg_day_count, DayCountMethod.Actual360)

    def test_float_day_count_explicit_wins_over_index(self):
        m = _make_market()
        float_leg = replace(m.swap.float_leg, day_count_method=DayCountMethod.ActualActual)
        m2 = replace(m, swap=replace(m.swap, float_leg=float_leg))
        self.assertEqual(m2.float_leg_day_count, DayCountMethod.ActualActual)

    def test_deposit_and_fra_day_counts_default_from_index(self):
        m = _make_market()
        self.assertEqual(m.deposit_day_count, DayCountMethod.Actual360)
        self.assertEqual(m.fra_day_count, DayCountMethod.Actual360)

    def test_coupon_type_is_a_convention(self):
        m = _make_market()
        self.assertEqual(m.swap.float_leg.coupon_type, CouponType.GeometricAveraged)

    def test_frozen(self):
        m = _make_market()
        with self.assertRaises(FrozenInstanceError):
            m.family = None
        with self.assertRaises(FrozenInstanceError):
            m.swap.fixed_leg.payment_frequency = Frequency.Annually

    def test_replace_returns_new_instance(self):
        m = _make_market()
        fixed = replace(m.swap.fixed_leg, payment_frequency=Frequency.Annually)
        self.assertEqual(fixed.payment_frequency, Frequency.Annually)
        self.assertEqual(m.swap.fixed_leg.payment_frequency, Frequency.SemiAnnually)

    def test_match_tenor_resolves_per_index(self):
        m = _make_market(tenors=("1M", "3M", "6M"))
        float_leg = replace(m.swap.float_leg, payment_frequency=MATCH_TENOR, reset_frequency=MATCH_TENOR)
        m2 = replace(m, swap=replace(m.swap, float_leg=float_leg))
        three = m2.index("3M")
        self.assertEqual(m2.float_leg_payment_frequency(three), Frequency.Quarterly)
        self.assertEqual(m2.float_leg_reset_frequency(three), Frequency.Quarterly)
        self.assertEqual(m2.float_leg_payment_frequency(m2.index("6M")), Frequency.SemiAnnually)
        # a fixed frequency is left alone
        self.assertEqual(m.float_leg_payment_frequency(m.index("3M")), Frequency.Quarterly)
        self.assertEqual(m.float_leg_reset_frequency(m.index("1M")), Frequency.Daily)

    def test_frequency_for_tenor(self):
        self.assertEqual(frequency_for_tenor(Term.from_str("12M")), Frequency.Annually)
        self.assertEqual(frequency_for_tenor(Term.from_str("1Y")), Frequency.Annually)
        with self.assertRaises(ValueError):
            frequency_for_tenor(Term.from_str("5M"))


class TestConventionRegistry(UnitTest):
    COVERAGE = ["finance.conventions.convention_registry"]

    def _fresh_registry(self) -> ConventionRegistry:
        from common.registry import Registry
        return ConventionRegistry(registry=Registry("TestConventionRegistry"))

    def test_register_and_get(self):
        reg = self._fresh_registry()
        conv = _make_market()
        reg.register("USD", "TEST", conv)
        self.assertEqual(reg.get("USD", "TEST"), conv)

    def test_key_is_case_insensitive(self):
        reg = self._fresh_registry()
        conv = _make_market()
        reg.register("usd", "test", conv)
        self.assertEqual(reg.get("USD", "TEST"), conv)
        self.assertEqual(reg.get("Usd", "Test"), conv)

    def test_missing_key_raises(self):
        reg = self._fresh_registry()
        with self.assertRaises(KeyError):
            reg.get("EUR", "EURIBOR")

    def test_duplicate_raises_without_overwrite(self):
        reg = self._fresh_registry()
        conv = _make_market()
        reg.register("USD", "TEST", conv)
        with self.assertRaises(ValueError):
            reg.register("USD", "TEST", conv)

    def test_overwrite_succeeds(self):
        reg = self._fresh_registry()
        c1 = _make_market()
        c2 = replace(c1, deposit=replace(c1.deposit, calendar="LON"))
        reg.register("USD", "TEST", c1)
        reg.register("USD", "TEST", c2, overwrite=True)
        self.assertEqual(reg.get("USD", "TEST").deposit.calendar, "LON")

    def test_has(self):
        reg = self._fresh_registry()
        reg.register("USD", "TEST", _make_market())
        self.assertTrue(reg.has("USD", "TEST"))
        self.assertFalse(reg.has("EUR", "EURIBOR"))

    def test_bundled_usd_sofr_definition(self):
        from finance.conventions import default_registry
        m = default_registry.get("USD", "SOFR")
        self.assertEqual(m.index().label, "USD.SOFR.ON")
        self.assertTrue(m.family.is_overnight)
        self.assertEqual(m.float_leg_day_count, DayCountMethod.Actual360)
        self.assertEqual(m.swap.float_leg.coupon_type, CouponType.GeometricAveraged)

    def test_index_lookup_by_name_and_tenor(self):
        from finance.conventions import default_registry as reg
        self.assertEqual(reg.index("USD", "SOFR").label, "USD.SOFR.ON")
        self.assertEqual(reg.index("usd", "sofr.on").label, "USD.SOFR.ON")
        self.assertEqual(reg.key("USD", "CME-TERM-SOFR.3M"), CurveKey.parse("USD.CME-TERM-SOFR.3M"))
        self.assertEqual(reg.key("USD", "CME-TERM-SOFR", "6M"), CurveKey.parse("USD.CME-TERM-SOFR.6M"))
        self.assertEqual(reg.get("USD", "CME-TERM-SOFR.3M").family.name, "CME-TERM-SOFR")
        with self.assertRaises(KeyError):
            reg.index("USD", "CME-TERM-SOFR")  # tenor required for a term family
        with self.assertRaises(KeyError):
            reg.index("USD", "SOFR.3M")  # overnight family publishes ON only
        with self.assertRaises(ValueError):
            reg.index("USD", "CME-TERM-SOFR.3M", "6M")  # tenor twice

    def test_validate_key(self):
        from finance.conventions import default_registry as reg
        self.assertEqual(reg.validate("USD.STDCSA"), CurveKey.parse("USD.STDCSA"))  # no conventions needed
        self.assertEqual(reg.validate("USD.SOFR.ON"), CurveKey.parse("USD.SOFR.ON"))
        with self.assertRaises(KeyError):
            reg.validate("USD.SOFR.3M")
        with self.assertRaises(KeyError):
            reg.validate("USD.NOTANINDEX.ON")

    def test_bundled_cme_term_sofr(self):
        from finance.conventions import default_registry
        m = default_registry.get("USD", "CME-TERM-SOFR")
        self.assertEqual(m.family.tenor_labels, ("1M", "3M", "6M", "12M"))
        self.assertIs(m.swap.float_leg.payment_frequency, MATCH_TENOR)
        self.assertEqual(m.swap.float_leg.coupon_type, CouponType.Floating)
        self.assertEqual(m.float_leg_payment_frequency(m.index("3M")), Frequency.Quarterly)
