"""Tests for the content-addressed handle cache."""
import numpy as np

from common.testing import UnitTest
from quant_toolkit_xl import cache


class TestCache(UnitTest):
    COVERAGE = ["quant_toolkit_xl.cache"]

    def setUp(self):
        cache.clear()

    def test_store_returns_kinded_handle(self):
        h = cache.store("Curve", object(), 1, 2, 3)
        self.assertTrue(h.startswith("Curve::"))

    def test_load_roundtrip(self):
        obj = object()
        h = cache.store("Swap", obj, "a", 1.0)
        self.assertIs(cache.load(h), obj)
        self.assertIs(cache.load(h, "Swap"), obj)

    def test_content_addressing_is_idempotent(self):
        # identical key parts (incl. ndarrays) -> identical handle, single stored object
        arr = np.array([1.0, 2.0, 3.0])
        h1 = cache.store("Curve", object(), arr, "LogLinearDF")
        h2 = cache.store("Curve", object(), np.array([1.0, 2.0, 3.0]), "LogLinearDF")
        self.assertEqual(h1, h2)

    def test_different_inputs_differ(self):
        h1 = cache.store("Curve", object(), 1)
        h2 = cache.store("Curve", object(), 2)
        self.assertNotEqual(h1, h2)

    def test_wrong_kind_raises_typeerror(self):
        h = cache.store("Swap", object(), 1)
        with self.assertRaises(TypeError):
            cache.load(h, "Curve")

    def test_missing_handle_raises_keyerror(self):
        with self.assertRaises(KeyError):
            cache.load("Curve::deadbeef")

    def test_malformed_handle_raises_valueerror(self):
        for bad in (123, None, "not-a-handle"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    cache.load(bad)

    def test_clear_returns_count_and_empties(self):
        cache.store("Curve", object(), 1)
        cache.store("Curve", object(), 2)
        self.assertEqual(cache.clear(), 2)
        with self.assertRaises(KeyError):
            cache.load("Curve::" + "0" * 12)
