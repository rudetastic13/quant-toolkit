"""Curve keys, index families and rate indices."""

from common.testing import UnitTest
from finance.conventions import OVERNIGHT, CurveKey, CurveKind, IndexFamily, RateIndex, parse_tenor, tenor_label
from finance.dates import Term
from finance.dates.enums import DayCountMethod, FixingType


def _sofr(**overrides) -> IndexFamily:
    defaults = dict(
        currency="USD",
        name="SOFR",
        day_count_method=DayCountMethod.Actual360,
        fixing_calendar="NYC",
    )
    defaults.update(overrides)
    return IndexFamily(**defaults)


def _term_sofr() -> IndexFamily:
    return _sofr(name="CME-TERM-SOFR", tenors=("1M", "3M", "6M", "12M"))


class TestCurveKey(UnitTest):
    COVERAGE = ["finance.conventions.keys"]

    def test_projection_key_parses_tenor_and_overnight(self):
        key = CurveKey.parse("usd.sofr.on")
        self.assertEqual((key.currency, key.name), ("USD", "SOFR"))
        self.assertEqual(key.tenor, OVERNIGHT)
        self.assertIs(key.kind, CurveKind.Projection)
        self.assertTrue(key.is_overnight)
        self.assertEqual(str(key), "USD.SOFR.ON")
        term = CurveKey.parse("USD.CME-TERM-SOFR.3M")
        self.assertEqual(term.tenor, Term.from_str("3M"))
        self.assertFalse(term.is_overnight)
        self.assertEqual(term.tenor_label, "3M")
        self.assertEqual(str(term), "USD.CME-TERM-SOFR.3M")

    def test_discount_key_has_no_tenor(self):
        key = CurveKey.parse("USD.STDCSA")
        self.assertIs(key.kind, CurveKind.Discount)
        self.assertIsNone(key.tenor)
        self.assertIsNone(key.tenor_label)
        self.assertTrue(key.is_discount)
        self.assertEqual(str(key), "USD.STDCSA")
        self.assertEqual(str(CurveKey.parse("GBP.USDSTDCSA")), "GBP.USDSTDCSA")

    def test_constructors(self):
        self.assertEqual(CurveKey.projection("USD", "SOFR"), CurveKey.parse("USD.SOFR.ON"))
        self.assertEqual(CurveKey.projection("USD", "CME-TERM-SOFR", "3M"), CurveKey.parse("USD.CME-TERM-SOFR.3M"))
        self.assertEqual(CurveKey.discount("USD", "UST"), CurveKey.parse("USD.UST"))
        self.assertIs(CurveKey.parse(CurveKey.discount("USD", "UST")).__class__, CurveKey)

    def test_value_equality_and_hash(self):
        a, b = CurveKey.parse("USD.SOFR.ON"), CurveKey("usd", "sofr", Term.from_str("1D"))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b, CurveKey.parse("USD.STDCSA")}), 2)
        self.assertNotEqual(a, CurveKey.parse("USD.SOFR.3M"))
        self.assertNotEqual(a, "USD.SOFR.ON")

    def test_rejects_bad_shapes(self):
        for text in ("USD", "USD.SOFR.ON.X", "USD.SOFR.0M", "USD.SOFR.XYZ", "US D.SOFR", "USD..ON"):
            with self.assertRaises(ValueError, msg=text):
                CurveKey.parse(text)
        with self.assertRaises(TypeError):
            CurveKey("USD", "SOFR", "3M")

    def test_tenor_helpers(self):
        self.assertEqual(tenor_label(OVERNIGHT), "ON")
        self.assertEqual(tenor_label(Term.from_str("6M")), "6M")
        self.assertEqual(parse_tenor("on"), OVERNIGHT)
        self.assertEqual(parse_tenor("3M"), Term.from_str("3M"))
        self.assertEqual(parse_tenor(Term.from_str("3M")), Term.from_str("3M"))
        with self.assertRaises(ValueError):
            parse_tenor("-3M")

    def test_repr(self):
        self.assertEqual(repr(CurveKey.parse("USD.SOFR.ON")), "CurveKey('USD.SOFR.ON')")


class TestIndexFamily(UnitTest):
    COVERAGE = ["finance.conventions.rate_index"]

    def test_overnight_family(self):
        fam = _sofr()
        self.assertTrue(fam.is_overnight)
        self.assertEqual(fam.tenor_labels, ("ON",))
        idx = fam.index()
        self.assertIsInstance(idx, RateIndex)
        self.assertTrue(idx.is_overnight)
        self.assertEqual(idx.label, "USD.SOFR.ON")
        self.assertEqual(idx.key, CurveKey.parse("USD.SOFR.ON"))
        self.assertEqual(fam.index("ON"), idx)

    def test_term_family_needs_a_tenor(self):
        fam = _term_sofr()
        self.assertFalse(fam.is_overnight)
        self.assertEqual(fam.tenor_labels, ("1M", "3M", "6M", "12M"))
        with self.assertRaises(KeyError):
            fam.index()
        with self.assertRaises(KeyError):
            fam.index("2M")
        idx = fam.index("3M")
        self.assertEqual(idx.tenor, Term.from_str("3M"))
        self.assertEqual(idx.label, "USD.CME-TERM-SOFR.3M")
        self.assertFalse(idx.is_overnight)
        self.assertTrue(fam.has_tenor(Term.from_str("12M")))
        self.assertFalse(fam.has_tenor("ON"))

    def test_index_passes_family_conventions_through(self):
        idx = _sofr(fixing_type=FixingType.Advance).index()
        self.assertEqual(idx.currency, "USD")
        self.assertEqual(idx.name, "SOFR")
        self.assertEqual(idx.day_count_method, DayCountMethod.Actual360)
        self.assertEqual(idx.fixing_calendar, "NYC")
        self.assertEqual(idx.fixing_type, FixingType.Advance)
        self.assertIsNone(idx.publication_lag)

    def test_rate_index_rejects_unpublished_tenor(self):
        with self.assertRaises(ValueError):
            RateIndex(family=_sofr(), tenor=Term.from_str("3M"))

    def test_family_validation(self):
        with self.assertRaises(ValueError):
            _sofr(tenors=())
        with self.assertRaises(ValueError):
            _sofr(tenors=("3M", "3M"))
        self.assertEqual(_sofr(currency="usd", name="sofr").index().label, "USD.SOFR.ON")
