"""Tests for CurveNamespace — bind/resolve/rebind and version counter."""
import numpy as np

from common.testing import UnitTest
from finance.markets.curves import ZeroCurve
from finance.pricing.namespace import CurveNamespace


def _make_curve() -> ZeroCurve:
    """Minimal flat ZeroCurve for testing — two nodes, DF=1 at origin."""
    dates = np.array(["2026-01-01", "2031-01-01"], dtype="datetime64[D]")
    dfs = np.array([1.0, 0.95])
    return ZeroCurve(dates, dfs)


class TestCurveNamespaceBind(UnitTest):
    COVERAGE = ["finance.pricing.namespace.curve_namespace"]

    def test_bind_and_resolve_roundtrip(self):
        ns = CurveNamespace()
        curve = _make_curve()
        ns.bind("USD.SOFR", curve)
        self.assertIs(ns.resolve("USD.SOFR"), curve)

    def test_initial_version_is_zero(self):
        ns = CurveNamespace()
        ns.bind("USD.SOFR", _make_curve())
        self.assertEqual(ns.version("USD.SOFR"), 0)

    def test_bind_duplicate_raises(self):
        ns = CurveNamespace()
        ns.bind("USD.SOFR", _make_curve())
        with self.assertRaises(KeyError):
            ns.bind("USD.SOFR", _make_curve())

    def test_contains(self):
        ns = CurveNamespace()
        ns.bind("USD.SOFR", _make_curve())
        self.assertIn("USD.SOFR", ns)
        self.assertNotIn("EUR.EURIBOR", ns)


class TestCurveNamespaceRebind(UnitTest):
    COVERAGE = ["finance.pricing.namespace.curve_namespace"]

    def test_rebind_replaces_curve(self):
        ns = CurveNamespace()
        c1 = _make_curve()
        c2 = _make_curve()
        ns.bind("USD.SOFR", c1)
        ns.rebind("USD.SOFR", c2)
        self.assertIs(ns.resolve("USD.SOFR"), c2)

    def test_rebind_increments_version(self):
        ns = CurveNamespace()
        ns.bind("USD.SOFR", _make_curve())
        ns.rebind("USD.SOFR", _make_curve())
        self.assertEqual(ns.version("USD.SOFR"), 1)
        ns.rebind("USD.SOFR", _make_curve())
        self.assertEqual(ns.version("USD.SOFR"), 2)

    def test_rebind_without_prior_bind(self):
        # rebind on unseen name should succeed and start version at 0
        ns = CurveNamespace()
        ns.rebind("USD.SOFR", _make_curve())
        self.assertEqual(ns.version("USD.SOFR"), 0)

    def test_snapshot_captures_versions(self):
        ns = CurveNamespace()
        c1 = _make_curve()
        c2 = _make_curve()
        ns.bind("A", c1)
        ns.bind("B", c2)
        ns.rebind("A", _make_curve())
        snap = ns.snapshot()
        self.assertEqual(snap["A"][1], 1)
        self.assertEqual(snap["B"][1], 0)


class TestCurveNamespaceErrors(UnitTest):
    COVERAGE = ["finance.pricing.namespace.curve_namespace"]

    def test_resolve_missing_raises(self):
        ns = CurveNamespace()
        with self.assertRaises(KeyError):
            ns.resolve("UNKNOWN")

    def test_version_missing_raises(self):
        ns = CurveNamespace()
        with self.assertRaises(KeyError):
            ns.version("UNKNOWN")
