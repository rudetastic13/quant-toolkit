from __future__ import annotations

from common.containers.enums import CaseInsensitiveEnum, SupportedIntEnum
from common.testing import UnitTest


class MySupported(SupportedIntEnum):
    Unsupported = -1
    Zero = 0
    One = 1


class TestSupportedIntEnum(UnitTest):
    COVERAGE = ["common.containers.enums"]

    def test_is_supported_positive(self):
        self.assertTrue(MySupported.One.is_supported())
        self.assertTrue(MySupported.Zero.is_supported())

    def test_is_supported_negative(self):
        self.assertFalse(MySupported.Unsupported.is_supported())


class MyCI(CaseInsensitiveEnum):
    Alpha = "alpha"
    Beta = "beta"
    Gamma = "gamma"


class TestCaseInsensitiveEnum(UnitTest):
    COVERAGE = ["common.containers.enums"]

    def test_getitem_exact_case(self):
        self.assertEqual(MyCI["Alpha"], MyCI.Alpha)

    def test_getitem_wrong_case(self):
        self.assertEqual(MyCI["ALPHA"], MyCI.Alpha)
        self.assertEqual(MyCI["alpha"], MyCI.Alpha)

    def test_call_case_insensitive_string(self):
        self.assertEqual(MyCI("BETA"), MyCI.Beta)
        self.assertEqual(MyCI("beta"), MyCI.Beta)

    def test_call_exact_value(self):
        self.assertEqual(MyCI("gamma"), MyCI.Gamma)

    def test_parse_via_key(self):
        self.assertEqual(MyCI.parse("Alpha"), MyCI.Alpha)

    def test_parse_via_value(self):
        self.assertEqual(MyCI.parse("beta"), MyCI.Beta)

    def test_duplicate_case_insensitive_members_raises(self):
        with self.assertRaises(RuntimeError):

            class Bad(CaseInsensitiveEnum):
                Foo = "foo"
                FOO = "bar"

    def test_parse_empty_members_raises(self):
        """Parsing on an enum with no members raises IndexError."""

        class Empty(CaseInsensitiveEnum):
            pass

        with self.assertRaises(IndexError):
            Empty.parse("anything")
