"""HistoricalFixings validation and flat lookup behavior."""
import numpy as np

from common.testing import UnitTest
from finance.markets import HistoricalFixings


class TestHistoricalFixings(UnitTest):
    COVERAGE = ["finance.markets.fixings"]

    def test_flat_interpolation_and_extrapolation(self):
        history = HistoricalFixings(
            dates=np.array(["2026-05-27", "2026-05-29"], dtype="datetime64[D]"),
            values=np.array([0.0361, 0.0364]),
        )
        queries = np.array(
            ["2026-05-20", "2026-05-27", "2026-05-28", "2026-06-01"],
            dtype="datetime64[D]",
        )
        np.testing.assert_allclose(history.get_value(queries), [0.0361, 0.0361, 0.0364, 0.0364])

    def test_constructor_copies_input_arrays(self):
        dates = np.array(["2026-05-27", "2026-05-29"], dtype="datetime64[D]")
        values = np.array([0.0361, 0.0364])
        history = HistoricalFixings(dates, values)
        dates[0] = np.datetime64("2020-01-01")
        values[0] = 99.0
        self.assertEqual(history.dates[0], np.datetime64("2026-05-27"))
        self.assertEqual(history.values[0], 0.0361)

    def test_invalid_history_is_rejected(self):
        valid_dates = np.array(["2026-05-27", "2026-05-29"], dtype="datetime64[D]")
        with self.assertRaises(ValueError):
            HistoricalFixings(valid_dates[::-1], np.array([0.0364, 0.0361]))
        with self.assertRaises(ValueError):
            HistoricalFixings(valid_dates, np.array([0.0361]))
        with self.assertRaises(ValueError):
            HistoricalFixings(valid_dates, np.array([0.0361, np.nan]))

