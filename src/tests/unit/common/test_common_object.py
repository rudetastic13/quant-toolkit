from __future__ import annotations

from dataclasses import dataclass

from common.object import (
    CommonObject,
    ValidationException,
    ValidationMessage,
    ValidationResult,
    ValidationType,
)
from common.testing import UnitTest


class TestValidationMessage(UnitTest):
    COVERAGE = ["common.object.validation"]

    def test_str_format(self):
        msg = ValidationMessage(ValidationType.ValidationFailure, "bad value")
        self.assertEqual(str(msg), "[ValidationFailure] bad value")

    def test_frozen(self):
        msg = ValidationMessage(ValidationType.ValidationWarning, "warn")
        with self.assertRaises(AttributeError):
            msg.message = "other"


class TestValidationResult(UnitTest):
    COVERAGE = ["common.object.validation"]

    def test_empty_result(self):
        result = ValidationResult()
        self.assertEqual(len(result), 0)
        self.assertFalse(result)
        self.assertFalse(result.has_failures)
        self.assertFalse(result.has_warnings)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.warnings, [])

    def test_add_failure(self):
        result = ValidationResult()
        result.add_failure("something broke")
        self.assertEqual(len(result), 1)
        self.assertTrue(result)
        self.assertTrue(result.has_failures)
        self.assertEqual(result.failures[0].message, "something broke")

    def test_add_warning(self):
        result = ValidationResult()
        result.add_warning("heads up")
        self.assertTrue(result.has_warnings)
        self.assertFalse(result.has_failures)
        self.assertEqual(result.warnings[0].message, "heads up")

    def test_add_assumption(self):
        result = ValidationResult()
        result.add_assumption("assuming 30/360")
        self.assertEqual(result.assumptions[0].message, "assuming 30/360")

    def test_add_conformance(self):
        result = ValidationResult()
        result.add_conformance("data conforms")
        self.assertEqual(result.conformances[0].message, "data conforms")

    def test_of_type_filters_correctly(self):
        result = ValidationResult()
        result.add_failure("fail")
        result.add_warning("warn")
        result.add_failure("fail2")
        self.assertEqual(len(result.of_type(ValidationType.ValidationFailure)), 2)
        self.assertEqual(len(result.of_type(ValidationType.ValidationWarning)), 1)
        self.assertEqual(len(result.of_type(ValidationType.DataAssumption)), 0)

    def test_add_combines_results(self):
        r1 = ValidationResult()
        r1.add_failure("f1")
        r2 = ValidationResult()
        r2.add_warning("w1")
        combined = r1 + r2
        self.assertEqual(len(combined), 2)
        self.assertTrue(combined.has_failures)
        self.assertTrue(combined.has_warnings)
        # originals unchanged
        self.assertEqual(len(r1), 1)
        self.assertEqual(len(r2), 1)

    def test_iadd_extends_in_place(self):
        r1 = ValidationResult()
        r1.add_failure("f1")
        r2 = ValidationResult()
        r2.add_warning("w1")
        r1 += r2
        self.assertEqual(len(r1), 2)

    def test_str_empty(self):
        result = ValidationResult()
        self.assertIn("no issues", str(result))

    def test_str_with_messages(self):
        result = ValidationResult()
        result.add_failure("bad")
        s = str(result)
        self.assertIn("1 issue(s)", s)
        self.assertIn("[ValidationFailure] bad", s)


class TestValidationException(UnitTest):
    COVERAGE = ["common.object.validation"]

    def test_exception_message_format(self):
        result = ValidationResult()
        result.add_failure("f1")
        result.add_failure("f2")
        exc = ValidationException(result, {ValidationType.ValidationFailure})
        self.assertIn("2 issue(s)", str(exc))
        self.assertIn("[ValidationFailure] f1", str(exc))

    def test_exception_preserves_full_result(self):
        result = ValidationResult()
        result.add_failure("fail")
        result.add_warning("warn")
        exc = ValidationException(result, {ValidationType.ValidationFailure})
        self.assertEqual(len(exc.result.messages), 2)
        self.assertEqual(len(exc.triggering_messages), 1)

    def test_exception_filters_triggering_messages(self):
        result = ValidationResult()
        result.add_failure("fail")
        result.add_warning("warn")
        result.add_assumption("assume")
        exc = ValidationException(result, {ValidationType.ValidationWarning, ValidationType.DataAssumption})
        self.assertEqual(len(exc.triggering_messages), 2)
        types = {m.validation_type for m in exc.triggering_messages}
        self.assertEqual(types, {ValidationType.ValidationWarning, ValidationType.DataAssumption})


class TestCommonObject(UnitTest):
    COVERAGE = ["common.object.common_object"]

    def test_validate_empty_passes(self):
        obj = CommonObject()
        result = obj.validate()
        self.assertFalse(result)

    def test_validate_raises_on_failure(self):
        @dataclass
        class Broken(CommonObject):
            def _validate_impl(self) -> ValidationResult:
                result = super()._validate_impl()
                result.add_failure("broken")
                return result

        with self.assertRaises(ValidationException) as ctx:
            Broken().validate()
        self.assertEqual(len(ctx.exception.triggering_messages), 1)

    def test_validate_does_not_raise_on_warning_by_default(self):
        @dataclass
        class Warned(CommonObject):
            def _validate_impl(self) -> ValidationResult:
                result = super()._validate_impl()
                result.add_warning("just a warning")
                return result

        result = Warned().validate()
        self.assertTrue(result.has_warnings)

    def test_validate_raises_on_custom_types(self):
        @dataclass
        class Assumed(CommonObject):
            def _validate_impl(self) -> ValidationResult:
                result = super()._validate_impl()
                result.add_assumption("assumed something")
                return result

        with self.assertRaises(ValidationException):
            Assumed().validate(raise_on={ValidationType.DataAssumption})

    def test_validate_empty_raise_on_never_raises(self):
        @dataclass
        class Broken(CommonObject):
            def _validate_impl(self) -> ValidationResult:
                result = super()._validate_impl()
                result.add_failure("broken")
                return result

        result = Broken().validate(raise_on=set())
        self.assertTrue(result.has_failures)

    def test_subclass_validation_chain(self):
        @dataclass(kw_only=True)
        class Parent(CommonObject):
            name: str = ""

            def _validate_impl(self) -> ValidationResult:
                result = super()._validate_impl()
                if not self.name:
                    result.add_failure("name is required")
                return result

        @dataclass(kw_only=True)
        class Child(Parent):
            value: float = 0.0

            def _validate_impl(self) -> ValidationResult:
                result = super()._validate_impl()
                if self.value < 0:
                    result.add_warning("value is negative")
                return result

        # Both validations fire
        child = Child(name="", value=-1.0)
        with self.assertRaises(ValidationException):
            child.validate()

        # Only warning, no raise with default raise_on
        child2 = Child(name="ok", value=-1.0)
        result = child2.validate()
        self.assertTrue(result.has_warnings)
        self.assertFalse(result.has_failures)

        # Clean
        child3 = Child(name="ok", value=1.0)
        result = child3.validate()
        self.assertFalse(result)
