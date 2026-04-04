from __future__ import annotations

from common.object.serializer import Serializable
from common.object.validation import ValidationException, ValidationResult, ValidationType


class CommonObject(Serializable):
    """Base class for domain objects providing structured validation and serialization."""

    def validate(self, raise_on: set[ValidationType] | None = None) -> ValidationResult:
        if raise_on is None:
            raise_on = {ValidationType.ValidationFailure}

        result = self._validate_impl()

        if raise_on and any(m.validation_type in raise_on for m in result.messages):
            raise ValidationException(result, raise_on)

        return result

    def _validate_impl(self) -> ValidationResult:
        return ValidationResult()
