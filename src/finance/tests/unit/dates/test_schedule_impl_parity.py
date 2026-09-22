"""Parity tests: loop/numba/C++ schedule generators vs the vectorized reference"""
import datetime
import itertools
import pytest
import numpy as np
from common.testing import UnitTest
from finance._core import datemath, is_available as core_available
from finance.dates.date import Date
from finance.dates.enums import Frequency, Roll, Direction
from finance.dates.schedules import generate_schedule
from finance.dates.schedules.compiled import generate_schedule_cpp, _UNIX_EPOCH_PYORDINAL
from finance.dates.schedules.loop_based import generate_schedule_loop
from finance.dates.schedules.numba_based import generate_schedule_numba, is_available as numba_available


@pytest.mark.datemath
@pytest.mark.skipif(not core_available(), reason="finance._core not built; run bin/build_core.sh")
class TestScalarDateMathParity(UnitTest):
    COVERAGE = ["finance._core"]

    # leap/century edges plus the epoch neighborhood
    edge_dates = [
        (1899, 12, 31),
        (1900, 2, 28),
        (1969, 12, 31),
        (1970, 1, 1),
        (2000, 2, 29),
        (2024, 2, 29),
        (2100, 2, 28),
        (9999, 12, 31),
    ]

    def test_days_from_ymd_matches_date(self):
        for ymd in self.edge_dates:
            with self.subTest(msg=f"days_from_ymd{ymd}"):
                self.assertEqual(datemath.days_from_ymd(*ymd), Date(*ymd).toordinal())

    def test_roundtrip_matches_date_over_range(self):
        # every 97 days across ~1910-2090
        for days in range(-22_000, 44_000, 97):
            with self.subTest(msg=f"ordinal {days}"):
                ymd = datemath.ymd_from_days(days)
                self.assertEqual(ymd, Date.fromordinal(days).to_ymd())
                self.assertEqual(datemath.days_from_ymd(*ymd), days)

    def test_python_date_epoch_offset(self):
        # datetime.date ordinals are 0001-01-01 based; the boundary shift is 719_163
        self.assertEqual(datetime.date(1970, 1, 1).toordinal(), _UNIX_EPOCH_PYORDINAL)
        self.assertEqual(
            datetime.date(2026, 7, 7).toordinal() - _UNIX_EPOCH_PYORDINAL,
            datemath.days_from_ymd(2026, 7, 7),
        )

    def test_days_in_month_and_leap(self):
        for y, m, expected in [(2024, 2, 29), (2025, 2, 28), (2000, 2, 29), (1900, 2, 28), (2025, 4, 30), (2025, 1, 31)]:
            with self.subTest(msg=f"days_in_month({y}, {m})"):
                self.assertEqual(datemath.days_in_month(y, m), expected)
        self.assertTrue(datemath.is_leap_year(2000))
        self.assertFalse(datemath.is_leap_year(1900))


@pytest.mark.datemath
class TestScheduleImplParity(UnitTest):
    COVERAGE = [
        "finance.dates.schedules.compiled",
        "finance.dates.schedules.loop_based",
        "finance.dates.schedules.numba_based",
    ]

    frequencies = [
        Frequency.Once,
        Frequency.Daily,
        Frequency.Weekly,
        Frequency.Monthly,
        Frequency.Quarterly,
        Frequency.SemiAnnually,
        Frequency.Annually,
    ]
    rolls = [Roll.Empty, Roll.EOM, Roll.IMM, Roll.RollDay1, Roll.RollDay15, Roll.RollDay28, Roll.RollDay29, Roll.RollDay31]
    directions = [Direction.Forward, Direction.Backward]
    # (first_regular_date, last_regular_date): no stub, front, back, both
    stub_configs = [
        (None, None),
        (Date(2025, 3, 15), None),
        (None, Date(2030, 3, 15)),
        (Date(2025, 3, 15), Date(2030, 3, 15)),
    ]
    start, end = Date(2025, 1, 10), Date(2030, 7, 20)

    # Roll.Empty EOM-inference: anchors on Jan 31 / Feb 29 with no regular dates given
    eom_anchor_cases = [
        (Date(2025, 1, 31), Date(2027, 1, 31)),
        (Date(2024, 2, 29), Date(2026, 2, 28)),
        (Date(2024, 11, 30), Date(2026, 11, 30)),
    ]

    def _assert_grid_parity(self, impl):
        grid = itertools.product(self.frequencies, self.rolls, self.stub_configs, self.directions)
        for frequency, roll, (first, last), direction in grid:
            with self.subTest(msg=f"{frequency.name}/{roll.name}/{direction.name}/first={first}/last={last}"):
                expected = generate_schedule(self.start, self.end, frequency, first, last, roll, direction)
                actual = impl(self.start, self.end, frequency, first, last, roll, direction)
                np.testing.assert_array_equal(expected, actual)
        for anchor_start, anchor_end in self.eom_anchor_cases:
            for frequency, direction in itertools.product([Frequency.Monthly, Frequency.Quarterly], self.directions):
                with self.subTest(msg=f"EOM-infer {anchor_start}/{frequency.name}/{direction.name}"):
                    expected = generate_schedule(anchor_start, anchor_end, frequency, direction=direction)
                    actual = impl(anchor_start, anchor_end, frequency, direction=direction)
                    np.testing.assert_array_equal(expected, actual)

    # CME quarterly IMM dates (third Wednesday of Mar/Jun/Sep/Dec), Dec 2024 - Dec 2026
    imm_quarterly = np.array(
        [
            "2024-12-18", "2025-03-19", "2025-06-18", "2025-09-17", "2025-12-17",
            "2026-03-18", "2026-06-17", "2026-09-16", "2026-12-16",
        ],
        dtype="datetime64[D]",
    )
    # serial (monthly) IMM dates, Jan-Jun 2025
    imm_serial = np.array(
        ["2025-01-15", "2025-02-19", "2025-03-19", "2025-04-16", "2025-05-21", "2025-06-18"],
        dtype="datetime64[D]",
    )

    def _imm_impls(self):
        impls = [generate_schedule, generate_schedule_loop]
        if numba_available():
            impls.append(generate_schedule_numba)
        if core_available():
            impls.append(generate_schedule_cpp)
        return impls

    def test_imm_known_cme_dates(self):
        # absolute anchor: parity alone would propagate a shared bug, so pin real CME dates
        for impl in self._imm_impls():
            with self.subTest(msg=f"quarterly/{impl.__module__}"):
                actual = impl(Date(2024, 12, 18), Date(2026, 12, 16), Frequency.Quarterly, roll_convention=Roll.IMM)
                np.testing.assert_array_equal(actual, self.imm_quarterly)
            with self.subTest(msg=f"serial/{impl.__module__}"):
                actual = impl(Date(2025, 1, 15), Date(2025, 6, 18), Frequency.Monthly, roll_convention=Roll.IMM)
                np.testing.assert_array_equal(actual, self.imm_serial)

    def test_imm_front_stub(self):
        # spot-starting swap rolling on IMM: non-IMM start snaps in as a front stub
        expected = np.concatenate([np.array(["2024-12-02"], dtype="datetime64[D]"), self.imm_quarterly[:5]])
        for impl in self._imm_impls():
            with self.subTest(msg=impl.__module__):
                actual = impl(
                    Date(2024, 12, 2),
                    Date(2025, 12, 17),
                    Frequency.Quarterly,
                    first_regular_date=Date(2024, 12, 18),
                    roll_convention=Roll.IMM,
                )
                np.testing.assert_array_equal(actual, expected)

    def _assert_rejects_bad_order(self, impl):
        with self.assertRaises(ValueError):
            impl(Date(2025, 1, 10), Date(2030, 7, 20), Frequency.Monthly, first_regular_date=Date(2024, 1, 1))

    def test_loop_parity(self):
        self._assert_grid_parity(generate_schedule_loop)
        self._assert_rejects_bad_order(generate_schedule_loop)

    @pytest.mark.skipif(not numba_available(), reason="numba not installed")
    def test_numba_parity(self):
        self._assert_grid_parity(generate_schedule_numba)
        self._assert_rejects_bad_order(generate_schedule_numba)

    @pytest.mark.skipif(not core_available(), reason="finance._core not built; run bin/build_core.sh")
    def test_cpp_parity(self):
        self._assert_grid_parity(generate_schedule_cpp)
        self._assert_rejects_bad_order(generate_schedule_cpp)

    @pytest.mark.skipif(not core_available(), reason="finance._core not built; run bin/build_core.sh")
    def test_cpp_accepts_all_date_types(self):
        # Date, datetime.date, 1970-based int ordinal, and datetime64 must agree
        start, end = Date(2025, 1, 15), Date(2028, 1, 15)
        expected = generate_schedule_cpp(start, end, Frequency.Quarterly)
        variants = [
            (datetime.date(2025, 1, 15), datetime.date(2028, 1, 15)),
            (start.toordinal(), end.toordinal()),
            (start.to_numpy(), end.to_numpy()),
        ]
        for s, e in variants:
            with self.subTest(msg=f"input type {type(s).__name__}"):
                actual = generate_schedule_cpp(s, e, Frequency.Quarterly)
                np.testing.assert_array_equal(expected, actual)
        with self.assertRaises(TypeError):
            generate_schedule_cpp("2025-01-15", end, Frequency.Quarterly)
