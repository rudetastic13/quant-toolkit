"""Tests for YieldCurve registration and namespace versioning."""
import numpy as np

from common.testing import UnitTest
from finance.markets.curves import CurveNamespace, YieldCurve


def _make_curve(index: str = "SOFR", terminal_df: float = 0.95) -> YieldCurve:
    dates = np.array(["2026-01-01", "2031-01-01"], dtype="datetime64[D]")
    return YieldCurve.build(dates, np.array([1.0, terminal_df]), currency="USD", index_name=index)


class TestCurveNamespaceBind(UnitTest):
    COVERAGE = ["finance.markets.curves.namespace"]

    def test_bind_and_resolve_roundtrip(self):
        namespace = CurveNamespace()
        curve = _make_curve()
        namespace.bind(curve)
        self.assertIs(namespace.resolve("USD.SOFR"), curve)

    def test_initial_version_is_zero(self):
        namespace = CurveNamespace()
        namespace.bind(_make_curve())
        self.assertEqual(namespace.version("USD.SOFR"), 0)

    def test_bind_duplicate_raises(self):
        namespace = CurveNamespace()
        namespace.bind(_make_curve())
        with self.assertRaises(KeyError):
            namespace.bind(_make_curve())

    def test_contains(self):
        namespace = CurveNamespace()
        namespace.bind(_make_curve())
        self.assertIn("USD.SOFR", namespace)
        self.assertNotIn("USD.FEDFUND", namespace)


class TestCurveNamespaceRebind(UnitTest):
    COVERAGE = ["finance.markets.curves.namespace"]

    def test_rebind_replaces_curve(self):
        namespace = CurveNamespace()
        first = _make_curve(terminal_df=0.95)
        second = _make_curve(terminal_df=0.90)
        namespace.bind(first)
        namespace.rebind(second)
        self.assertIs(namespace.resolve("USD.SOFR"), second)

    def test_rebind_increments_version(self):
        namespace = CurveNamespace()
        namespace.bind(_make_curve())
        namespace.rebind(_make_curve(terminal_df=0.94))
        self.assertEqual(namespace.version("USD.SOFR"), 1)
        namespace.rebind(_make_curve(terminal_df=0.93))
        self.assertEqual(namespace.version("USD.SOFR"), 2)

    def test_rebind_without_prior_bind(self):
        namespace = CurveNamespace()
        namespace.rebind(_make_curve())
        self.assertEqual(namespace.version("USD.SOFR"), 0)

    def test_snapshot_captures_versions(self):
        namespace = CurveNamespace()
        namespace.bind(_make_curve("SOFR"))
        namespace.bind(_make_curve("FEDFUND"))
        namespace.rebind(_make_curve("SOFR", terminal_df=0.94))
        snapshot = namespace.snapshot()
        self.assertEqual(snapshot["USD.SOFR"][1], 1)
        self.assertEqual(snapshot["USD.FEDFUND"][1], 0)


class TestCurveNamespaceErrors(UnitTest):
    COVERAGE = ["finance.markets.curves.namespace"]

    def test_resolve_missing_raises(self):
        namespace = CurveNamespace()
        with self.assertRaises(KeyError):
            namespace.resolve("UNKNOWN")

    def test_version_missing_raises(self):
        namespace = CurveNamespace()
        with self.assertRaises(KeyError):
            namespace.version("UNKNOWN")
