"""Unit tests for finance.dates.utils dispatcher and its np/dt backends."""
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from common.testing import UnitTest
from finance.dates import (
    Date,
    Term,
    TermType,
    BDC,
    Frequency,
    Calendars,
)
from finance.dates.calendars.calendar import StaticCalendar
from finance.dates.utils import (
    is_good_bd,
    adjust_date,
    add_business_days,
    add_term,
    subtract_term,
    add_frequency,
    subtract_frequency,
    next_imm_date,
    prior_imm_date,
    date_based_utils_module,
    np_date_based_utils_module,
)

TEST_CAL_NAME = "test_utils_cal"


def _make_test_calendar() -> StaticCalendar:
    """Weekday calendar with a handful of known 2025 holidays."""
    return StaticCalendar(
        name=TEST_CAL_NAME,
        week_mask="1111100",
        holidays={
            Date(2025, 1, 1),    # Wed
            Date(2025, 7, 4),    # Fri
            Date(2025, 12, 25),  # Thu
            Date(2024, 12, 31),  # Tue
        },
    )


# ---------------------------------------------------------------------------
# base harness: one set of scenarios, two backends (np + dt) via subclasses
# ---------------------------------------------------------------------------

class _BaseUtilsTest(UnitTest):
    __test__ = False
    COVERAGE = ["finance.dates.utils"]

    def setUp(self):
        Calendars()
        Calendars.register(_make_test_calendar())

    def _dt(self, d: Date):
        raise NotImplementedError

    def _dts(self, ds: list[Date]):
        raise NotImplementedError

    def _assert_dt_equal(self, actual, expected: Date):
        raise NotImplementedError

    def _assert_dts_equal(self, actual, expected: list[Date]):
        raise NotImplementedError

    def _assert_bool(self, actual, expected: bool):
        self.assertEqual(bool(actual), expected)


class _NpBase(_BaseUtilsTest):
    """Backend: np.datetime64 scalars and ndarrays."""

    def _dt(self, d):
        return d.to_numpy()

    def _dts(self, ds):
        return np.array([d.to_str() for d in ds], dtype="datetime64[D]")

    def _assert_dt_equal(self, actual, expected):
        self.assertEqual(actual, np.datetime64(expected.to_str()))

    def _assert_dts_equal(self, actual, expected):
        assert_array_equal(
            actual,
            np.array([d.to_str() for d in expected], dtype="datetime64[D]"),
        )


class _DtBase(_BaseUtilsTest):
    """Backend: finance.dates.Date scalars."""

    def _dt(self, d):
        return d

    def _dts(self, ds):
        return ds

    def _assert_dt_equal(self, actual, expected):
        self.assertEqual(actual, expected)

    def _assert_dts_equal(self, actual, expected):
        self.assertEqual(list(actual), list(expected))


# ---------------------------------------------------------------------------
# is_good_bd
# ---------------------------------------------------------------------------

@pytest.mark.datemath
class _TestIsGoodBd(_BaseUtilsTest):
    __test__ = False

    def test_weekday_is_good(self):
        self._assert_bool(is_good_bd(self._dt(Date(2025, 1, 6)), TEST_CAL_NAME), True)

    def test_saturday_not_good(self):
        self._assert_bool(is_good_bd(self._dt(Date(2025, 1, 4)), TEST_CAL_NAME), False)

    def test_sunday_not_good(self):
        self._assert_bool(is_good_bd(self._dt(Date(2025, 1, 5)), TEST_CAL_NAME), False)

    def test_holiday_not_good(self):
        self._assert_bool(is_good_bd(self._dt(Date(2025, 1, 1)), TEST_CAL_NAME), False)
        self._assert_bool(is_good_bd(self._dt(Date(2025, 7, 4)), TEST_CAL_NAME), False)


class TestIsGoodBdNp(_NpBase, _TestIsGoodBd):
    __test__ = True

    def test_array_input(self):
        dates = self._dts([Date(2025, 1, 4), Date(2025, 1, 6), Date(2025, 1, 1)])
        result = is_good_bd(dates, TEST_CAL_NAME)
        assert_array_equal(result, np.array([False, True, False]))


class TestIsGoodBdDt(_DtBase, _TestIsGoodBd):
    __test__ = True

    def test_returns_python_bool(self):
        self.assertIsInstance(is_good_bd(Date(2025, 1, 6), TEST_CAL_NAME), bool)


# ---------------------------------------------------------------------------
# adjust_date
# ---------------------------------------------------------------------------

@pytest.mark.datemath
class _TestAdjustDate(_BaseUtilsTest):
    __test__ = False

    def test_no_adjustment_returns_input(self):
        result = adjust_date(self._dt(Date(2025, 1, 4)), BDC.NoAdjustment, TEST_CAL_NAME)
        self._assert_dt_equal(result, Date(2025, 1, 4))

    def test_following_from_saturday(self):
        result = adjust_date(self._dt(Date(2025, 1, 4)), BDC.Following, TEST_CAL_NAME)
        self._assert_dt_equal(result, Date(2025, 1, 6))

    def test_preceding_from_saturday(self):
        result = adjust_date(self._dt(Date(2025, 1, 4)), BDC.Preceding, TEST_CAL_NAME)
        self._assert_dt_equal(result, Date(2025, 1, 3))

    def test_business_day_unchanged(self):
        result = adjust_date(self._dt(Date(2025, 1, 6)), BDC.Following, TEST_CAL_NAME)
        self._assert_dt_equal(result, Date(2025, 1, 6))

    def test_following_from_holiday(self):
        # Wed Jan 1 is a holiday → next bd is Thu Jan 2
        result = adjust_date(self._dt(Date(2025, 1, 1)), BDC.Following, TEST_CAL_NAME)
        self._assert_dt_equal(result, Date(2025, 1, 2))

    def test_modified_following_rolls_back_across_month(self):
        # Sat May 31 2025; Following → Mon Jun 2 (different month) → roll back to Fri May 30
        result = adjust_date(
            self._dt(Date(2025, 5, 31)), BDC.ModifiedFollowing, TEST_CAL_NAME
        )
        self._assert_dt_equal(result, Date(2025, 5, 30))

    def test_modified_preceding_rolls_forward_across_month(self):
        # Sun Jun 1 2025; Preceding → Fri May 30 (different month) → roll forward to Mon Jun 2
        result = adjust_date(
            self._dt(Date(2025, 6, 1)), BDC.ModifiedPreceding, TEST_CAL_NAME
        )
        self._assert_dt_equal(result, Date(2025, 6, 2))

    def test_bdc_as_string(self):
        result = adjust_date(self._dt(Date(2025, 1, 4)), "Following", TEST_CAL_NAME)
        self._assert_dt_equal(result, Date(2025, 1, 6))


class TestAdjustDateNp(_NpBase, _TestAdjustDate):
    __test__ = True

    def test_array_input(self):
        dates = self._dts([Date(2025, 1, 4), Date(2025, 1, 5)])
        result = adjust_date(dates, BDC.Following, TEST_CAL_NAME)
        self._assert_dts_equal(result, [Date(2025, 1, 6), Date(2025, 1, 6)])


class TestAdjustDateDt(_DtBase, _TestAdjustDate):
    __test__ = True


# ---------------------------------------------------------------------------
# add_business_days
# ---------------------------------------------------------------------------

@pytest.mark.datemath
class _TestAddBusinessDays(_BaseUtilsTest):
    __test__ = False

    def test_single_bd_forward(self):
        # Mon Jan 6 → Tue Jan 7
        result = add_business_days(
            self._dt(Date(2025, 1, 6)), 1, BDC.Following, TEST_CAL_NAME
        )
        self._assert_dt_equal(result, Date(2025, 1, 7))

    def test_zero_bd(self):
        result = add_business_days(
            self._dt(Date(2025, 1, 6)), 0, BDC.Following, TEST_CAL_NAME
        )
        self._assert_dt_equal(result, Date(2025, 1, 6))


class TestAddBusinessDaysNp(_NpBase, _TestAddBusinessDays):
    __test__ = True

    def test_cross_weekend(self):
        # Thu Jan 2 + 5 BD → Thu Jan 9
        result = add_business_days(
            self._dt(Date(2025, 1, 2)), 5, BDC.Following, TEST_CAL_NAME
        )
        self._assert_dt_equal(result, Date(2025, 1, 9))

    def test_skip_holiday(self):
        # Wed Dec 24 + 1 BD: Thu Dec 25 is holiday → Fri Dec 26
        result = add_business_days(
            self._dt(Date(2025, 12, 24)), 1, BDC.Following, TEST_CAL_NAME
        )
        self._assert_dt_equal(result, Date(2025, 12, 26))

    def test_negative_bd(self):
        # Mon Jan 6 - 1 BD → Fri Jan 3
        result = add_business_days(
            self._dt(Date(2025, 1, 6)), -1, BDC.Following, TEST_CAL_NAME
        )
        self._assert_dt_equal(result, Date(2025, 1, 3))

    def test_array_input(self):
        starts = self._dts([Date(2025, 1, 2), Date(2025, 1, 6)])
        result = add_business_days(starts, 1, BDC.Following, TEST_CAL_NAME)
        self._assert_dts_equal(result, [Date(2025, 1, 3), Date(2025, 1, 7)])

    def test_no_adjustment_adds_calendar_days(self):
        # With NoAdjustment the np backend returns dt + Term(days, Days) raw
        result = add_business_days(
            self._dt(Date(2025, 1, 3)), 2, BDC.NoAdjustment, TEST_CAL_NAME
        )
        self._assert_dt_equal(result, Date(2025, 1, 5))


class TestAddBusinessDaysDt(_DtBase, _TestAddBusinessDays):
    __test__ = True

    def test_within_single_week(self):
        # Mon Jan 6 + 4 BD → Fri Jan 10 (no weekend crossing in dt-impl's loop)
        result = add_business_days(
            self._dt(Date(2025, 1, 6)), 4, BDC.Following, TEST_CAL_NAME
        )
        self._assert_dt_equal(result, Date(2025, 1, 10))


# ---------------------------------------------------------------------------
# add_term
# ---------------------------------------------------------------------------

@pytest.mark.datemath
class _TestAddTerm(_BaseUtilsTest):
    __test__ = False

    def test_add_months_lands_on_weekday(self):
        # Jan 15 2025 + 3M = Apr 15 2025 (Tue); MF doesn't move
        result = add_term(
            self._dt(Date(2025, 1, 15)),
            Term(3, TermType.Months),
            BDC.ModifiedFollowing,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2025, 4, 15))

    def test_add_years(self):
        # Jan 15 2025 + 1Y = Jan 15 2026 (Thu, bd)
        result = add_term(
            self._dt(Date(2025, 1, 15)),
            Term(1, TermType.Years),
            BDC.Following,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2026, 1, 15))

    def test_add_term_lands_on_weekend_mf_adjusts(self):
        # Feb 1 + 1M = Mar 1 (Sat); MF → Mon Mar 3 (same month → keep)
        result = add_term(
            self._dt(Date(2025, 2, 1)),
            Term(1, TermType.Months),
            BDC.ModifiedFollowing,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2025, 3, 3))

    def test_add_business_days_term_delegates(self):
        # Term with BusinessDays term type delegates to add_business_days
        result = add_term(
            self._dt(Date(2025, 1, 6)),
            Term(1, TermType.BusinessDays),
            BDC.Following,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2025, 1, 7))


class TestAddTermNp(_NpBase, _TestAddTerm):
    __test__ = True

    def test_array_input(self):
        starts = self._dts([Date(2025, 1, 15), Date(2025, 4, 15)])
        result = add_term(
            starts, Term(3, TermType.Months), BDC.ModifiedFollowing, TEST_CAL_NAME
        )
        self._assert_dts_equal(result, [Date(2025, 4, 15), Date(2025, 7, 15)])


class TestAddTermDt(_DtBase, _TestAddTerm):
    __test__ = True


# ---------------------------------------------------------------------------
# subtract_term
# ---------------------------------------------------------------------------

@pytest.mark.datemath
class _TestSubtractTerm(_BaseUtilsTest):
    __test__ = False

    def test_subtract_months(self):
        result = subtract_term(
            self._dt(Date(2025, 4, 15)),
            Term(3, TermType.Months),
            BDC.Following,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2025, 1, 15))

    def test_subtract_years(self):
        result = subtract_term(
            self._dt(Date(2026, 1, 15)),
            Term(1, TermType.Years),
            BDC.Following,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2025, 1, 15))


class TestSubtractTermNp(_NpBase, _TestSubtractTerm):
    __test__ = True

    def test_array_input(self):
        starts = self._dts([Date(2025, 4, 15), Date(2025, 7, 15)])
        result = subtract_term(
            starts, Term(3, TermType.Months), BDC.Following, TEST_CAL_NAME
        )
        self._assert_dts_equal(result, [Date(2025, 1, 15), Date(2025, 4, 15)])


class TestSubtractTermDt(_DtBase, _TestSubtractTerm):
    __test__ = True


# ---------------------------------------------------------------------------
# add_frequency
# ---------------------------------------------------------------------------

@pytest.mark.datemath
class _TestAddFrequency(_BaseUtilsTest):
    __test__ = False

    def test_quarterly(self):
        result = add_frequency(
            self._dt(Date(2025, 1, 15)),
            Frequency.Quarterly,
            BDC.ModifiedFollowing,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2025, 4, 15))

    def test_semi_annually(self):
        result = add_frequency(
            self._dt(Date(2025, 1, 15)),
            Frequency.SemiAnnually,
            BDC.Following,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2025, 7, 15))

    def test_annually(self):
        result = add_frequency(
            self._dt(Date(2025, 1, 15)),
            Frequency.Annually,
            BDC.Following,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2026, 1, 15))


class TestAddFrequencyNp(_NpBase, _TestAddFrequency):
    __test__ = True

    def test_array_input(self):
        starts = self._dts([Date(2025, 1, 15), Date(2025, 4, 15)])
        result = add_frequency(
            starts, Frequency.Quarterly, BDC.ModifiedFollowing, TEST_CAL_NAME
        )
        self._assert_dts_equal(result, [Date(2025, 4, 15), Date(2025, 7, 15)])


class TestAddFrequencyDt(_DtBase, _TestAddFrequency):
    __test__ = True


# ---------------------------------------------------------------------------
# subtract_frequency
# ---------------------------------------------------------------------------

@pytest.mark.datemath
class _TestSubtractFrequency(_BaseUtilsTest):
    __test__ = False

    def test_quarterly(self):
        result = subtract_frequency(
            self._dt(Date(2025, 4, 15)),
            Frequency.Quarterly,
            BDC.ModifiedFollowing,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2025, 1, 15))

    def test_annually(self):
        result = subtract_frequency(
            self._dt(Date(2026, 1, 15)),
            Frequency.Annually,
            BDC.Following,
            TEST_CAL_NAME,
        )
        self._assert_dt_equal(result, Date(2025, 1, 15))


class TestSubtractFrequencyNp(_NpBase, _TestSubtractFrequency):
    __test__ = True

    def test_array_input(self):
        starts = self._dts([Date(2025, 4, 15), Date(2025, 7, 15)])
        result = subtract_frequency(
            starts, Frequency.Quarterly, BDC.Following, TEST_CAL_NAME
        )
        self._assert_dts_equal(result, [Date(2025, 1, 15), Date(2025, 4, 15)])


class TestSubtractFrequencyDt(_DtBase, _TestSubtractFrequency):
    __test__ = True


# ---------------------------------------------------------------------------
# next_imm_date / prior_imm_date
# ---------------------------------------------------------------------------

@pytest.mark.datemath
class _TestImmDates(_BaseUtilsTest):
    __test__ = False

    # 2025 quarterly IMM dates: Mar 19, Jun 18, Sep 17, Dec 17

    def test_next_quarterly(self):
        self._assert_dt_equal(next_imm_date(self._dt(Date(2025, 1, 2))), Date(2025, 3, 19))
        self._assert_dt_equal(next_imm_date(self._dt(Date(2025, 6, 17))), Date(2025, 6, 18))
        self._assert_dt_equal(next_imm_date(self._dt(Date(2025, 6, 19))), Date(2025, 9, 17))

    def test_next_is_strict_on_imm_date(self):
        self._assert_dt_equal(next_imm_date(self._dt(Date(2025, 6, 18))), Date(2025, 9, 17))

    def test_prior_quarterly(self):
        self._assert_dt_equal(prior_imm_date(self._dt(Date(2025, 6, 19))), Date(2025, 6, 18))
        self._assert_dt_equal(prior_imm_date(self._dt(Date(2025, 6, 17))), Date(2025, 3, 19))

    def test_prior_is_strict_on_imm_date(self):
        self._assert_dt_equal(prior_imm_date(self._dt(Date(2025, 6, 18))), Date(2025, 3, 19))

    def test_year_boundary(self):
        self._assert_dt_equal(next_imm_date(self._dt(Date(2025, 12, 18))), Date(2026, 3, 18))
        self._assert_dt_equal(prior_imm_date(self._dt(Date(2026, 1, 5))), Date(2025, 12, 17))

    def test_serial_monthly(self):
        self._assert_dt_equal(next_imm_date(self._dt(Date(2025, 1, 16)), Frequency.Monthly), Date(2025, 2, 19))
        self._assert_dt_equal(prior_imm_date(self._dt(Date(2025, 1, 16)), Frequency.Monthly), Date(2025, 1, 15))

    def test_rejects_non_month_frequency(self):
        for frequency in (Frequency.Weekly, Frequency.Daily, Frequency.Once, Frequency.TwoYearly):
            with self.assertRaises(ValueError):
                next_imm_date(self._dt(Date(2025, 1, 16)), frequency)
            with self.assertRaises(ValueError):
                prior_imm_date(self._dt(Date(2025, 1, 16)), frequency)


class TestImmDatesNp(_NpBase, _TestImmDates):
    __test__ = True

    def test_array_input(self):
        dts = self._dts([Date(2025, 6, 17), Date(2025, 6, 18), Date(2025, 6, 19)])
        self._assert_dts_equal(next_imm_date(dts), [Date(2025, 6, 18), Date(2025, 9, 17), Date(2025, 9, 17)])
        self._assert_dts_equal(prior_imm_date(dts), [Date(2025, 3, 19), Date(2025, 3, 19), Date(2025, 6, 18)])

    def test_sweep_matches_dt_backend(self):
        # every day across 2024-2026: the vectorized results agree with the scalar Date backend
        days = np.arange(np.datetime64("2024-01-01"), np.datetime64("2027-01-01"), dtype="datetime64[D]")
        dt_mod = date_based_utils_module()
        for frequency in (Frequency.Quarterly, Frequency.Monthly):
            nxt = next_imm_date(days, frequency).view(np.int64)
            pri = prior_imm_date(days, frequency).view(np.int64)
            for ordinal, expected_next, expected_prior in zip(days.view(np.int64), nxt, pri):
                d = Date.fromordinal(int(ordinal))
                self.assertEqual(dt_mod.next_imm_date(d, frequency).toordinal(), expected_next, msg=f"next {d}")
                self.assertEqual(dt_mod.prior_imm_date(d, frequency).toordinal(), expected_prior, msg=f"prior {d}")


class TestImmDatesDt(_DtBase, _TestImmDates):
    __test__ = True

    def test_returns_date_type(self):
        self.assertIsInstance(next_imm_date(Date(2025, 6, 17)), Date)
        self.assertIsInstance(prior_imm_date(Date(2025, 6, 17)), Date)


# ---------------------------------------------------------------------------
# dispatcher and helper error paths
# ---------------------------------------------------------------------------

@pytest.mark.datemath
class TestDispatcher(UnitTest):
    """Verify type-based routing of the façade."""
    COVERAGE = ["finance.dates.utils"]

    def setUp(self):
        Calendars()
        Calendars.register(_make_test_calendar())

    def test_date_routes_to_dt_backend(self):
        expected = date_based_utils_module().is_good_bd(Date(2025, 1, 6), TEST_CAL_NAME)
        out = is_good_bd(Date(2025, 1, 6), TEST_CAL_NAME)
        self.assertEqual(out, expected)
        self.assertIsInstance(out, bool)

    def test_np_scalar_routes_to_np_backend(self):
        d = Date(2025, 1, 6).to_numpy()
        expected = np_date_based_utils_module().is_good_bd(d, TEST_CAL_NAME)
        out = is_good_bd(d, TEST_CAL_NAME)
        self.assertEqual(out, expected)
        self.assertIsInstance(out, np.bool_)

    def test_np_array_routes_to_np_backend(self):
        arr = np.array(["2025-01-04", "2025-01-06"], dtype="datetime64[D]")
        expected = np_date_based_utils_module().is_good_bd(arr, TEST_CAL_NAME)
        out = is_good_bd(arr, TEST_CAL_NAME)
        assert_array_equal(out, expected)
        self.assertIsInstance(out, np.ndarray)

    def test_unsupported_input_type_raises(self):
        with self.assertRaises(KeyError):
            is_good_bd("2025-01-06", TEST_CAL_NAME)


@pytest.mark.datemath
class TestHelperErrors(UnitTest):
    """Error paths in finance.dates.utils.impl.helpers."""
    COVERAGE = ["finance.dates.utils.impl.helpers"]

    def setUp(self):
        Calendars()

    def test_invalid_calendar_type_raises(self):
        with self.assertRaises(TypeError):
            is_good_bd(Date(2025, 1, 6), 42)

    def test_invalid_bdc_type_raises(self):
        with self.assertRaises(TypeError):
            adjust_date(Date(2025, 1, 4), 42, "no_holidays")
