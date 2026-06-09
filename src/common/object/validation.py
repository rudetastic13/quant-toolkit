from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto


class ValidationType(IntEnum):
    ValidationFailure = auto()
    ValidationWarning = auto()
    DataAssumption = auto()
    DataConformance = auto()


@dataclass(slots=True, frozen=True)
class ValidationMessage:
    validation_type: ValidationType
    message: str

    def __str__(self) -> str:
        return f"[{self.validation_type.name}] {self.message}"


@dataclass(slots=True)
class ValidationResult:
    messages: list[ValidationMessage] = field(default_factory=list)

    def of_type(self, validation_type: ValidationType) -> list[ValidationMessage]:
        return [m for m in self.messages if m.validation_type == validation_type]

    @property
    def failures(self) -> list[ValidationMessage]:
        return self.of_type(ValidationType.ValidationFailure)

    @property
    def warnings(self) -> list[ValidationMessage]:
        return self.of_type(ValidationType.ValidationWarning)

    @property
    def assumptions(self) -> list[ValidationMessage]:
        return self.of_type(ValidationType.DataAssumption)

    @property
    def conformances(self) -> list[ValidationMessage]:
        return self.of_type(ValidationType.DataConformance)

    @property
    def has_failures(self) -> bool:
        return any(m.validation_type == ValidationType.ValidationFailure for m in self.messages)

    @property
    def has_warnings(self) -> bool:
        return any(m.validation_type == ValidationType.ValidationWarning for m in self.messages)

    def add(self, validation_type: ValidationType, message: str) -> None:
        self.messages.append(ValidationMessage(validation_type=validation_type, message=message))

    def add_failure(self, message: str) -> None:
        self.add(ValidationType.ValidationFailure, message)

    def add_warning(self, message: str) -> None:
        self.add(ValidationType.ValidationWarning, message)

    def add_assumption(self, message: str) -> None:
        self.add(ValidationType.DataAssumption, message)

    def add_conformance(self, message: str) -> None:
        self.add(ValidationType.DataConformance, message)

    def __add__(self, other: ValidationResult) -> ValidationResult:
        return ValidationResult(messages=self.messages + other.messages)

    def __iadd__(self, other: ValidationResult) -> ValidationResult:
        self.messages.extend(other.messages)
        return self

    def __len__(self) -> int:
        return len(self.messages)

    def __bool__(self) -> bool:
        return len(self.messages) > 0

    def __str__(self) -> str:
        if not self.messages:
            return "ValidationResult: no issues"
        lines = [f"ValidationResult: {len(self.messages)} issue(s)"]
        for msg in self.messages:
            lines.append(f"  - {msg}")
        return "\n".join(lines)


class ValidationException(Exception):
    def __init__(self, result: ValidationResult, raise_types: set[ValidationType]) -> None:
        self.result = result
        self.raise_types = raise_types
        self.triggering_messages: list[ValidationMessage] = [
            m for m in result.messages if m.validation_type in raise_types
        ]
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        lines = [f"Validation failed with {len(self.triggering_messages)} issue(s):"]
        for msg in self.triggering_messages:
            lines.append(f"  - {msg}")
        return "\n".join(lines)


class Validatable:
    """Mixin that provides structured validation via an overridable _validate_impl hook.

    Subclasses override ``_validate_impl`` to return a populated ``ValidationResult``.
    Call ``validate(raise_on=...)`` to trigger validation and optionally surface problems
    as a ``ValidationException``.
    """

    def validate(self, raise_on: set[ValidationType] | None = None) -> ValidationResult:
        """Run validation and return the result.

        Args:
            raise_on: Set of ``ValidationType`` values that should cause a
                ``ValidationException`` to be raised.  Defaults to
                ``{ValidationFailure}`` when *None*.

        Returns:
            The ``ValidationResult`` produced by ``_validate_impl``.

        Raises:
            ValidationException: When any message in the result matches a type in ``raise_on``.
        """
        if raise_on is None:
            raise_on = {ValidationType.ValidationFailure}

        result = self._validate_impl()

        if raise_on and any(m.validation_type in raise_on for m in result.messages):
            raise ValidationException(result, raise_on)

        return result

    def _validate_impl(self) -> ValidationResult:
        """Override in subclasses to populate a ``ValidationResult``.

        Always call ``super()._validate_impl()`` first so parent-class rules are
        composed correctly up the hierarchy.
        """
        return ValidationResult()
